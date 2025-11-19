from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa

from osrl.common.net import mlp, CVAE, Classifier, \
    EnsembleQCritic, SquashedGaussianMLPActor, WeightsNet
    
from osrl.common.dataset import TransitionDataset

class CCAC(nn.Module):
    """
    Constriant-conditioned Actor-Cricie

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes 
                               of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes 
                               of the layers in the critic network.
        vae_hidden_sizes (int): Number of hidden units in the CVAE. 
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks. 
        beta (float): Weight of the KL divergence term.
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 state_min: np.ndarray,
                 state_max: np.ndarray,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 vae_hidden_sizes: int = 64,
                 vae_latent_sizes: int = 32,
                 cost_conditioned: bool = True,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 beta: float = 1.5,
                 num_q: int = 1,
                 num_qc: int = 1,
                 ood_threshold: float = 0.5,
                 max_lag: float = 5.0,
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.vae_hidden_sizes = vae_hidden_sizes
        self.cost_conditioned = cost_conditioned
        self.max_lag = max_lag

        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.num_q = num_q
        self.num_qc = num_qc
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = vae_latent_sizes
        self.episode_len = episode_len
        self.max_action = max_action

        self.device = device
        self.ood_threshold = ood_threshold

        ################ create actor critic model ###############
        self.actor = SquashedGaussianMLPActor(self.state_dim, self.action_dim,
                                              self.a_hidden_sizes, nn.ReLU,
                                              self.cost_conditioned).to(self.device)
        self.critic = EnsembleQCritic(self.state_dim,
                                      self.action_dim,
                                      self.c_hidden_sizes,
                                      nn.ReLU,
                                      self.cost_conditioned,
                                      num_q=self.num_q).to(self.device)
        self.cost_critic = EnsembleQCritic(self.state_dim,
                                           self.action_dim,
                                           self.c_hidden_sizes,
                                           nn.ReLU,
                                           self.cost_conditioned,
                                           num_q=self.num_qc).to(self.device)
        self.cvae = CVAE(self.state_dim, self.action_dim, 
                         self.vae_hidden_sizes, self.latent_dim, 
                         self.max_action, state_min, state_max,
                         self.device).to(self.device)
        self.classifier = Classifier(self.state_dim, 
                                     self.action_dim, 
                                     self.vae_hidden_sizes).to(self.device)

        self.qc_lag = WeightsNet(1, self.vae_hidden_sizes, 1).to(self.device) # k <= E_{s, a \sim v}[Q_c(s, a)]
        self.actor_lag = WeightsNet(1, self.vae_hidden_sizes, 1).to(self.device) # for policy

        self.actor_old = deepcopy(self.actor); self.actor_old.eval()
        self.critic_old = deepcopy(self.critic); self.critic_old.eval()
        self.cost_critic_old = deepcopy(self.cost_critic); self.cost_critic_old.eval()

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module 
        towards the parameters of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _actor_forward(self,
                       obs: torch.tensor,
                       deterministic: bool = False,
                       with_logprob: bool = True):
        """
        Return action distribution and action log prob [optional].
        """
        a, logp = self.actor.forward(obs, deterministic, with_logprob)
        return a * self.max_action, logp

    def vae_loss(self, observations, actions, cost_thresholds):
        recon_sa, mean, std = self.cvae.forward(
                observations, actions, cost_thresholds)
        true_sa = torch.cat([observations, actions], dim=1)
        recon_loss = nn.functional.mse_loss(recon_sa, true_sa)
            
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss_vae = recon_loss + self.beta * KL_loss

        self.cvae_optim.zero_grad()
        loss_vae.backward()
        self.cvae_optim.step()
        
        # binary class: [in-dist, out-of-dist]
        # if cost_thresholds > cost_threshold then [0, 1], else [1, 0]
        # 0 is in-dist and 1 is out-of-dist
        new_cost = cost_thresholds[0] * torch.ones_like(cost_thresholds)
        true_sac = torch.cat([true_sa, new_cost[:, None]], dim=1)
        recon_sac = torch.cat([recon_sa.detach(), new_cost[:, None]], dim=1)
        sac = torch.vstack([true_sac, recon_sac])
        
        labels = (cost_thresholds > cost_thresholds[0]).float()
        labels = torch.cat([labels, labels])
        
        logits = self.classifier.forward(sac)
        loss_class = F.binary_cross_entropy(logits, labels)
        self.classifier_optim.zero_grad()
        loss_class.backward()
        self.classifier_optim.step()
        
        with torch.no_grad():
            pred = torch.where(logits > self.ood_threshold, 1, 0)
            acc = (pred == labels).sum() / pred.shape[0]
        
        stats_vae = {"loss/recon_loss_sa": recon_loss.item(),
                     "loss/loss_class": loss_class.item(),
                     "loss/pred_acc": acc.cpu().numpy(),
                     "loss/kl_loss": KL_loss.item(),
                     "loss/loss_vae": loss_vae.item()}

        return loss_vae, stats_vae

    def critic_loss_vanilla(self, observations, next_observations, actions, 
                    rewards, cost_thresholds, next_cost_thresholds, done):
        
        sc = torch.cat([observations, cost_thresholds[:, None]], dim=1)
        next_sc = torch.cat([next_observations, next_cost_thresholds[:, None]], dim=1)
        _, qr_list = self.critic.predict(sc, actions)
        # Bellman backup for Q functions
        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_sc, False, True)
            qr_targ, _ = self.critic_old.predict(next_sc, next_actions)
            qr_backup = rewards + self.gamma * (1 - done) * qr_targ

        # MSE loss against Bellman backup
        loss_critic = self.critic.loss(qr_backup, qr_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        stats_critic = {"loss/critic_loss": loss_critic.item(),
                        "loss/qr": qr_targ.mean().item()}
        return loss_critic, stats_critic

    def cost_critic_loss_cvae(self, observations, next_observations, actions, 
                         costs, cost_thresholds, next_cost_thresholds, done):
        
        sc = torch.cat([observations, cost_thresholds[:, None]], dim=1)
        next_sc = torch.cat([next_observations, next_cost_thresholds[:, None]], dim=1)
        _, qc_list = self.cost_critic.predict(sc, actions)
        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_sc, False, True)
            qc_targ, _ = self.cost_critic_old.predict(next_sc, next_actions)
            qc_backup = costs + self.gamma * (1 - done) * qc_targ
            
            # generate ood (s, a) randomly
            batch_size = observations.shape[0]
            noise = torch.randn(batch_size, self.cvae.latent_dim, device=self.device)
            noise.data.clamp_(-1, 1)
            gen_sa = self.cvae.decode(sc, cost_thresholds[:, None], noise) # sc is not used
            gen_sac = torch.cat([gen_sa, cost_thresholds[:, None]], dim=1)
            
            # if logits > ood_threshold, then 1, else 0
            labels = self.classifier.predict(gen_sac, self.ood_threshold)
            ood_mask = labels == 1
            ood_sa = gen_sa[ood_mask, :]
            ood_cost = cost_thresholds[ood_mask]

        # bellman error
        qc_loss = self.cost_critic.loss(qc_backup, qc_list)

        if torch.sum(ood_mask) > 0:
            ood_s = ood_sa[:, :self.state_dim]
            ood_sc = torch.cat([ood_s, ood_cost[:, None]], dim=1)
            ood_a = ood_sa[:, self.state_dim:]
            qc_ood, _ = self.cost_critic.predict(ood_sc, ood_a)
            qc_ood_detach = qc_ood.detach()
            
            log_lmbda1 = self.qc_lag.forward(ood_cost[:, None]).squeeze()
            log_lmbda1.data.clamp_(min=-20, max=self.max_lag)
            lmbda1 = log_lmbda1.exp()
            lmbda1_detach = lmbda1.detach()

            loss_lag1 = - lmbda1_detach * (qc_ood - ood_cost)

            loss_cost_critic = qc_loss.mean() + loss_lag1.mean()
            self.cost_critic_optim.zero_grad()
            loss_cost_critic.backward()
            self.cost_critic_optim.step()
            
            # update lagrangian
            loss_lag1 = lmbda1 * (qc_ood_detach - ood_cost)
            loss_lagrangian = loss_lag1.mean()
            for param in self.qc_lag.parameters():
                loss_lagrangian += 0.01 * torch.norm(param)**2
            self.lagrangian_optim.zero_grad()
            loss_lagrangian.backward()
            self.lagrangian_optim.step()
            
            stats_cost_critic = {
                "loss/cost_critic_loss": loss_cost_critic.item(),
                "loss/qc": qc_targ.mean().item(),
                "loss/loss_qc_lag": loss_lagrangian.item(),
            }
        else:
            loss_cost_critic = qc_loss.mean()
            self.cost_critic_optim.zero_grad()
            loss_cost_critic.backward()
            self.cost_critic_optim.step()
            
            stats_cost_critic = {
                "loss/cost_critic_loss": loss_cost_critic.item(),
                "loss/qc": qc_targ.mean().item(),
                "loss/loss_qc_lag": 0,
            }
        return loss_cost_critic, stats_cost_critic
    
    def actor_loss(self, observations, cost_thresholds):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False

        sc = torch.cat([observations, cost_thresholds[:, None]], dim=1)
        actions, _ = self._actor_forward(sc, False, True)
        qr_pi, _ = self.critic.predict(sc, actions)
        qc_pi, _ = self.cost_critic.predict(sc, actions)
        qc_pi_detach = qc_pi.detach()
        
        log_lmbda3 = self.actor_lag.forward(cost_thresholds[:, None]).squeeze()
        log_lmbda3.data.clamp_(min=-20, max=self.max_lag)
        lmbda3 = log_lmbda3.exp()
        lmbda3_detach = lmbda3.detach()
        
        # update actor
        loss_qr = - qr_pi
        loss_lag3 = lmbda3_detach * (qc_pi - cost_thresholds)

        loss_actor = loss_qr.mean() + loss_lag3.mean()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        
        # update lagrangian
        loss_lag3 = - lmbda3 * (qc_pi_detach - cost_thresholds)
        loss_lagrangian = loss_lag3.mean()
        for param in self.actor_lag.parameters():
            loss_lagrangian += 0.01 * torch.norm(param)**2
        self.lagrangian_optim.zero_grad()
        loss_lagrangian.backward()
        self.lagrangian_optim.step()
        
        stats_actor = {"loss/actor_loss": loss_actor.item(),
                       "loss/loss_actor_lag": loss_lagrangian.item()}
        
        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        
        return loss_actor, stats_actor
    
    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def setup_optimizers(self, actor_lr, critic_lr, vae_lr):

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)

        self.lagrangian_optim = torch.optim.Adam(
            list(self.qc_lag.parameters()) + \
            list(self.actor_lag.parameters()), lr=actor_lr)
        
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        
        self.cost_critic_optim = torch.optim.Adam(
            self.cost_critic.parameters(), lr=critic_lr)

        self.cvae_optim = torch.optim.Adam(self.cvae.parameters(), lr=vae_lr)
        
        self.classifier_optim = torch.optim.Adam(self.classifier.parameters(), lr=vae_lr)

    def act(self,
            obs: np.ndarray,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single obs, return the action, logp.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self._actor_forward(obs, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu(
        ).numpy()

        # check monotonicity of critics
        # for thd in range(1, 100):
        #     sc = torch.cat([obs[:, :-1], torch.tensor([[thd]], device=obs.device)], dim=-1)
        #     qr = self.critic.predict(sc, a)
        #     qc = self.cost_critic.predict(sc, a)
        #     print(f"thd: {thd}, Q_r: {qr[0][0].item():.2f}, Q_c: {qc[0][0].item():.2f}")
            

        return np.squeeze(a, axis=0), np.squeeze(logp_a)


class CCACTrainer:
    """
    Constraints Penalized Q-learning Trainer
    
    Args:
        model (CPQ): The CPQ model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic
        omega_lr (float): learning rate for alpha
        vae_lr (float): learning rate for vae
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: CCAC,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            vae_lr: float = 1e-4,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            device="cpu") -> None:

        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        # self.model.setup_optimizers(actor_lr, critic_lr, omega_lr, vae_lr)
        self.model.setup_optimizers(actor_lr, critic_lr, vae_lr)

    def train_cvae_only(self, observations, actions, cost_thresholds):
        # update VAE
        cost_thresholds = cost_thresholds.to(torch.float32)
        loss_vae, stats_vae = self.model.vae_loss(
            observations, actions, cost_thresholds)

    def train_one_step_cvae(self, observations, next_observations, 
                       actions, rewards, costs, cost_thresholds, 
                       next_cost_thresholds, done):
        # update VAE
        cost_thresholds = cost_thresholds.to(torch.float32)
        next_cost_thresholds = next_cost_thresholds.to(torch.float32)
        loss_vae, stats_vae = self.model.vae_loss(
            observations, actions, cost_thresholds)
        # update critic
        loss_critic, stats_critic = self.model.critic_loss_vanilla(
            observations, next_observations, actions, 
            rewards, cost_thresholds, next_cost_thresholds, done)
        # update cost critic
        loss_cost_critic, stats_cost_critic = self.model.cost_critic_loss_cvae(
            observations, next_observations, actions, 
            costs, cost_thresholds, next_cost_thresholds, done)
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, cost_thresholds)

        self.model.sync_weight()

        self.logger.store(**stats_vae)
        self.logger.store(**stats_critic)
        self.logger.store(**stats_cost_critic)
        self.logger.store(**stats_actor)
        
    def evaluate(self, eval_episodes, target_cost=0):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        with torch.no_grad():
            episode_rets, episode_costs, episode_lens = [], [], []
            for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
                epi_ret, epi_len, epi_cost = self.rollout(target_cost)
                episode_rets.append(epi_ret)
                episode_lens.append(epi_len)
                episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(self, target_cost):
        """
        Evaluates the performance of the model on a single episode.
        """
        obs, info = self.env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        new_cost = target_cost
        for _ in range(self.model.episode_len):
            obs = np.append(obs, new_cost)
            act, _ = self.model.act(obs, True, True)

            obs_next, reward, terminated, truncated, info = self.env.step(act)
            cost = info["cost"] * self.cost_scale
            
            new_cost -= cost
            new_cost = np.maximum(0, new_cost)

            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost
