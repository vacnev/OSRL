from typing import Optional, Tuple
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from fsrl.utils import DummyLogger, WandbLogger
from torch.distributions.beta import Beta
from torch.nn import functional as F  # noqa
from tqdm.auto import trange  # noqa

from osrl.common.net import DiagGaussianActor, TransformerBlock, mlp, EnsembleQCritic, WeightsNet


class PDT(nn.Module):
    """
    Pareto-regularized constrained Decision Transformer (PDT)
    
    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        seq_len (int): The length of the sequence to process.
        episode_len (int): The length of the episode.
        embedding_dim (int): The dimension of the embeddings.
        num_layers (int): The number of transformer layers to use.
        num_heads (int): The number of heads to use in the multi-head attention.
        attention_dropout (float): The dropout probability for attention layers.
        residual_dropout (float): The dropout probability for residual layers.
        embedding_dropout (float): The dropout probability for embedding layers.
        time_emb (bool): Whether to include time embeddings.
        use_rew (bool): Whether to include return embeddings.
        use_cost (bool): Whether to include cost embeddings.
        cost_transform (bool): Whether to transform the cost values.
        add_cost_feat (bool): Whether to add cost features.
        mul_cost_feat (bool): Whether to multiply cost features.
        cat_cost_feat (bool): Whether to concatenate cost features.
        action_head_layers (int): The number of layers in the action head.
        stochastic (bool): Whether to use stochastic actions.
        init_temperature (float): The initial temperature value for stochastic actions.
        target_entropy (float): The target entropy value for stochastic actions.
        num_qr (int): Number of Q functions to use in the reward critic.
        num_qc (int): Number of Q functions to use in the cost critic.
        c_hidden_sizes (Tuple[int, ...]): Hidden layer sizes for the critics.
        tau (float): Target network update rate.
        gamma (float): Discount factor for future rewards.
        use_verification (bool): Whether to use verification critic.
        infer_q (bool): Whether to use Q value during inference.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        time_emb: bool = True,
        use_rew: bool = False,
        use_cost: bool = False,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        stochastic: bool = False,
        init_temperature=0.1,
        target_entropy=None,
        num_qr: int = 4,
        num_qc: int = 4,
        c_hidden_sizes: Tuple[int, ...] = (512, 512, 512),
        tau: float = 0.005,
        gamma: float = 0.99,
        cost_gamma: float = 0.99,
        use_verification: bool = True,
        infer_q: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = stochastic
        self.tau = tau
        self.gamma = gamma
        self.cost_gamma = cost_gamma
        self.use_verification = use_verification
        self.infer_q = infer_q

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)

        self.seq_repeat = 2
        self.use_rew = use_rew
        self.use_cost = use_cost
        if self.use_cost:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.blocks = nn.ModuleList([
            TransformerBlock(
                seq_len=dt_seq_len,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            ) for _ in range(num_layers)
        ])

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        if self.stochastic:
            if action_head_layers >= 2:
                self.action_head = nn.Sequential(
                    nn.Linear(action_emb_dim, action_emb_dim), nn.GELU(),
                    DiagGaussianActor(action_emb_dim, action_dim))
            else:
                self.action_head = DiagGaussianActor(action_emb_dim, action_dim)
        else:
            self.action_head = mlp([action_emb_dim] * action_head_layers + [action_dim],
                                   activation=nn.GELU,
                                   output_activation=nn.Identity)
        self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        # a classification problem
        self.cost_pred_head = nn.Linear(embedding_dim, 2)

        if self.stochastic:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

        self.apply(self._init_weights)

        self.critic = EnsembleQCritic(self.state_dim,
                                      self.action_dim,
                                      hidden_sizes=c_hidden_sizes,
                                      cost_conditioned=True,
                                      num_q=num_qr,
                                      activation=nn.Mish,
                                      )
        
        self.cost_critic = EnsembleQCritic(self.state_dim,
                                           self.action_dim,
                                           hidden_sizes=c_hidden_sizes,
                                           cost_conditioned=True,
                                           num_q=num_qc,
                                           activation=nn.Mish,
                                           )
        
        self.actor_lag = WeightsNet(1, 128, 1, activation=nn.Mish)
        
        self.critic_target = deepcopy(self.critic)
        self.critic_target.eval()
        self.cost_critic_target = deepcopy(self.cost_critic)
        self.cost_critic_target.eval()

    def actor_parameters(self, include_aux=True):
        params = []

        # embeddings
        params += list(self.state_emb.parameters())
        params += list(self.action_emb.parameters())
        if self.use_cost:
            params += list(self.cost_emb.parameters())
        if self.use_rew:
            params += list(self.return_emb.parameters())
        if self.time_emb:
            params += list(self.timestep_emb.parameters())

        # transformer
        params += list(self.blocks.parameters())

        # norms
        params += list(self.emb_norm.parameters())
        params += list(self.out_norm.parameters())

        # heads
        params += list(self.action_head.parameters())
        params += list(self.state_pred_head.parameters())
        params += list(self.cost_pred_head.parameters())

        # temperature
        if self.stochastic:
            params.append(self.log_temperature)

        return params

    def temperature(self):
        if self.stochastic:
            return self.log_temperature.exp()
        else:
            return None

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def ema_update(self, target, source):
        for tgt_param, param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def sync_target_networks(self):
        self.ema_update(self.critic_target, self.critic)
        self.ema_update(self.cost_critic_target, self.cost_critic)

    @torch.no_grad()
    def pred_targets(self, sc, actions):
        target_qr, _ = self.critic_target.predict(sc, actions)
        target_qc, _ = self.cost_critic_target.predict(sc, actions)
        return target_qr, target_qc
    
    def pred_critics(self, sc, actions):
        # turn off grad for efficiency in predictions
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False

        qr_preds, _ = self.critic.predict(sc, actions)
        qc_preds, _ = self.cost_critic.predict(sc, actions)

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True

        return qr_preds, qc_preds


    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            costs_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]

        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)

        # [batch_size, seq_len, 2-4, emb_dim], (c_0 s_0, a_0, c_1, s_1, a_1, ...)
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len,
                                    self.embedding_dim)

        if padding_mask is not None:
            # [batch_size, seq_len * self.seq_repeat], stack mask identically to fit the sequence
            padding_mask = torch.stack([padding_mask] * self.seq_repeat,
                                       dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len * self.seq_repeat, embedding_dim]
        out = self.out_norm(out)

        # [batch_size, seq_len, self.seq_repeat, embedding_dim]
        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        # [batch_size, self.seq_repeat, seq_len, embedding_dim]
        out = out.permute(0, 2, 1, 3)

        # [batch_size, seq_len, embedding_dim]
        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]

        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            # [batch_size, seq_len, 2 * embedding_dim]
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        # get predictions

        action_preds = self.action_head(
            state_feat
        )  # predict next action given state, [batch_size, seq_len, action_dim]
        # [batch_size, seq_len, 2]
        cost_preds = self.cost_pred_head(
            action_feature)  # predict next cost return given state and action
        cost_preds = F.log_softmax(cost_preds, dim=-1)

        state_preds = self.state_pred_head(
            action_feature)  # predict next state given state and action

        return action_preds, cost_preds, state_preds
    
    def act(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            costs_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
            logger: Optional[WandbLogger] = None,
            n_repeats: int = 50,
    ) -> torch.FloatTensor:
        batch_size = returns_to_go.shape[0]
        states = states.repeat_interleave(repeats=n_repeats, dim=0)
        actions = actions.repeat_interleave(repeats=n_repeats, dim=0)
        costs_to_go = costs_to_go.repeat_interleave(repeats=n_repeats, dim=0)
        time_steps = time_steps.repeat_interleave(repeats=n_repeats, dim=0)
        if padding_mask is not None:
            padding_mask = padding_mask.repeat_interleave(repeats=n_repeats, dim=0)

        batch_repeats = n_repeats // batch_size
        returns_to_go = returns_to_go.repeat_interleave(repeats=batch_repeats, dim=0)
        returns_to_go = torch.cat([returns_to_go, torch.randn((n_repeats - returns_to_go.shape[0], returns_to_go.shape[1]), device=returns_to_go.device)], dim=0)

        # Uniform distribution perturbation for diversity
        slack = 0.05
        for i in range(batch_size):
            target_return = returns_to_go[i * batch_repeats, -1]
            low, high = target_return * (1 - slack), target_return * (1 + slack)
            returns_to_go[i * batch_repeats + 1:(i + 1) * batch_repeats, -1] = torch.rand((batch_repeats - 1), device=returns_to_go.device) * (high - low) + low

        # Add Q RTG action
        # predict from second to last, last one is dummy, check that we are not on the first step
        if actions.shape[1] > 1:
            sc = torch.cat([states[-1:, -2], costs_to_go[-1:, -2].unsqueeze(-1)], dim=-1)
            last_rew = returns_to_go[0, -2] - returns_to_go[0, -1]
            returns_to_go[-1, -1] = (self.critic.predict(sc, actions[-1:, -2])[0] - last_rew) / self.gamma

        action_preds, _, _ = self.forward(states, actions, returns_to_go, costs_to_go, time_steps, padding_mask)

        if self.stochastic:
            action_preds = action_preds.mean

        states_rpt = states[:, -1, :]
        costs_to_go_rpt = costs_to_go[:, -1]
        sc_rpt = torch.cat([states_rpt, costs_to_go_rpt.unsqueeze(-1)], dim=-1)
        action_preds = action_preds[:, -1, :]

        qr_preds, qc_preds = self.pred_critics(sc_rpt, action_preds)

        # Verification filtering to filter out unsafe actions
        if self.use_verification:
            safe_mask = (qc_preds <= costs_to_go[0, -1])
            if safe_mask.sum() > 0:
                qr_preds = qr_preds[safe_mask]
                action_preds = action_preds[safe_mask]

        idx = torch.multinomial(F.softmax(qr_preds, dim=-1), 1)

        if self.infer_q:
            return action_preds[idx.item()]
        else:
            return action_preds[0]

        

class PDTTrainer:
    """
    Pareto-regularized Constrained Decision Transformer Trainer
    
    Args:
        model (PDT): A PDT model to train.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): The learning rate for the actor optimizer.
        critic_lr (float): The learning rate for the critic optimizer.
        weight_decay (float): The weight decay for the optimizer.
        betas (Tuple[float, ...]): The betas for the optimizer.
        clip_grad (float): The clip gradient value.
        clip_grad_critic (float): The clip gradient value for the critic.
        lr_warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        loss_cost_weight (float): The weight for the cost loss.
        loss_state_weight (float): The weight for the state loss.
        eta (float): Weight for the lagrangian loss.
        cost_reverse (bool): Whether to reverse the cost.
        no_entropy (bool): Whether to use entropy.
        n_step (bool): Whether to use n-step returns.
        device (str): The device to use for training (e.g. "cpu" or "cuda").

    """

    def __init__(
            self,
            model: PDT,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            weight_decay: float = 1e-4,
            betas: Tuple[float, ...] = (0.9, 0.999),
            clip_grad: float = 0.25,
            clip_grad_critic: float = 2.0,
            lr_warmup_steps: int = 10000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            loss_cost_weight: float = 0.0,
            loss_state_weight: float = 0.0,
            eta: float = 1.0,
            max_lag: float = 5.0,
            min_lag: float = -20.0,
            cost_reverse: bool = False,
            no_entropy: bool = False,
            n_step: bool = True,
            device="cpu") -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.clip_grad_critic = clip_grad_critic
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.eta = eta
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy
        self.n_step = n_step
        self.max_lag = max_lag
        self.min_lag = min_lag

        self.actor_optim = torch.optim.AdamW(
            self.model.actor_parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )
        self.stochastic = self.model.stochastic
        if self.stochastic:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4,
                betas=[0.9, 0.999],
            )
        self.max_action = self.model.max_action

        self.beta_dist = Beta(torch.tensor(2, dtype=torch.float, device=self.device),
                              torch.tensor(5, dtype=torch.float, device=self.device))
        
        self.critic_optim = torch.optim.Adam(
            self.model.critic.parameters(), lr=critic_lr
        )
        self.cost_critic_optim = torch.optim.Adam(
            self.model.cost_critic.parameters(), lr=critic_lr
        )
        
        # Add warmup schedulers for critics (consistent with actor)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )
        self.cost_critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.cost_critic_optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )

        self.lagrangian_optim = torch.optim.Adam(
            self.model.actor_lag.parameters(), lr=actor_lr
        )

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask,
                       costs):
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)
        action_preds, cost_preds, state_preds = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            costs_to_go=costs_return,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        seq_len = states.shape[1]
        batch_size = states.shape[0]
        # find last valid index for each sequence
        last_idxs = mask.sum(dim=1) - 1  # [batch_size]
        last_idxs = last_idxs.long()
        batch_idxs = torch.arange(len(last_idxs), device=self.device)

        if self.stochastic:
            action_preds_mean = action_preds.mean
        else:
            action_preds_mean = action_preds

        # CRITIC LOSS
        sc = torch.cat([states, costs_return.unsqueeze(-1)], dim=-1)
        current_qrs = self.model.critic(sc, actions)
        current_qrs = torch.stack(current_qrs, dim=0)  # [num_qr, batch_size, seq_len]
        current_qcs = self.model.cost_critic(sc, actions)
        current_qcs = torch.stack(current_qcs, dim=0)  # [num_qc, batch_size, seq_len]

        # compute discounted n-step returns
        rewards = returns[:, :-1] - returns[:, 1:]  # r_t = R_t - R_{t+1}
        rewards = torch.cat([rewards, torch.zeros(batch_size, 1, device=self.device)], dim=1) # [batch_size, seq_len]
        if self.n_step:
            last_sc = sc[batch_idxs, last_idxs]  # [batch_size, state_dim + 1]
            last_action = action_preds_mean[batch_idxs, last_idxs]  # [batch_size, action_dim]
            target_qr, target_qc = self.model.pred_targets(last_sc, last_action)  # [batch_size]

            # zero the last cost for n-step as it is already included in the target Q
            costs_ = costs.clone()
            costs_[batch_idxs, last_idxs] = 0.

            # vectorized discounting
            arange = torch.arange(seq_len, device=self.device)  # [seq_len]
            exp_mat = arange * mask  # [batch_size, seq_len]

            discount = self.model.gamma ** exp_mat.float()  # [batch_size, seq_len]
            discount_cost = self.model.cost_gamma ** exp_mat.float()  # [batch_size, seq_len]

            n_rews = torch.cumsum((rewards * discount).flip(dims=[1]), dim=1).flip(dims=[1]) / discount # [batch_size, seq_len]
            n_costs = torch.cumsum((costs_ * discount_cost).flip(dims=[1]), dim=1).flip(dims=[1]) / discount_cost  # [batch_size, seq_len]

            # add target Q for bootstrap
            valid_len = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            exp_mat = torch.maximum(valid_len - 1 - arange, torch.zeros(1, device=self.device))  # [batch_size, seq_len]
            discount = self.model.gamma ** exp_mat  # [batch_size, seq_len]
            discount_cost = self.model.cost_gamma ** exp_mat  # [batch_size, seq_len]

            target_qr = (n_rews + target_qr.unsqueeze(-1) * discount).detach()  # [batch_size, seq_len]
            target_qc = (n_costs + target_qc.unsqueeze(-1) * discount_cost).detach()  # [batch_size, seq_len]
        else:
            target_qr, target_qc = self.model.pred_targets(sc, action_preds_mean)  # [batch_size, seq_len], [batch_size, seq_len]
            target_qr = target_qr[:, 1:]
            target_qc = target_qc[:, 1:]
            target_qr = rewards[:, :-1] + self.model.gamma * target_qr  # [batch_size, seq_len - 1]
            target_qc = costs[:, :-1] + self.model.cost_gamma * target_qc  # [batch_size, seq_len - 1]
            target_qr = torch.cat([target_qr, torch.zeros(batch_size, 1, device=self.device)], dim=1).detach()
            target_qc = torch.cat([target_qc, torch.zeros(batch_size, 1, device=self.device)], dim=1).detach()

        # target_qr = target_qr.unsqueeze(0).expand_as(current_qrs)  # [num_qr, batch_size, seq_len]
        # target_qc = target_qc.unsqueeze(0).expand_as(current_qcs)  # [num_qc, batch_size, seq_len]

        # mask out last valid index in each sequence
        mask_ = mask.clone()
        mask_[batch_idxs, last_idxs] = 0
        # mask_r = mask_.unsqueeze(0).expand_as(current_qrs)  # [num_qr, batch_size, seq_len]
        # mask_c = mask_.unsqueeze(0).expand_as(current_qcs)  # [num_qc, batch_size, seq_len]

        # critic_loss = F.mse_loss(current_qrs[mask_r > 0], target_qr[mask_r > 0])
        # cost_critic_loss = F.mse_loss(current_qcs[mask_c > 0], target_qc[mask_c > 0])

        critic_loss, cost_critic_loss = 0.0, 0.0
        for i in range(current_qrs.shape[0]):
            critic_loss += F.mse_loss(current_qrs[i][mask_ > 0], target_qr[mask_ > 0])
            cost_critic_loss += F.mse_loss(current_qcs[i][mask_ > 0], target_qc[mask_ > 0])

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.clip_grad_critic)
        self.critic_optim.step()
        self.cost_critic_optim.zero_grad()
        cost_critic_loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.cost_critic.parameters(), self.clip_grad_critic)
        self.cost_critic_optim.step()

        self.model.sync_target_networks()

        # ACTOR LOSS

        # cost_preds: [batch_size * seq_len, 2], costs: [batch_size * seq_len]
        cost_preds = cost_preds.reshape(-1, 2)
        costs = costs.flatten().long().detach()
        cost_loss = F.nll_loss(cost_preds, costs, reduction="none")
        # cost_loss = F.mse_loss(cost_preds, costs.detach(), reduction="none")
        cost_loss = (cost_loss * mask.flatten()).mean()
        # compute the accuracy, 0 value, 1 indice, [batch_size, seq_len]
        pred = cost_preds.data.max(dim=1)[1]
        correct = pred.eq(costs.data.view_as(pred)) * mask.flatten()
        correct = correct.sum()
        total_num = mask.sum()
        acc = correct / total_num

        if self.stochastic:
            log_likelihood = action_preds.log_prob(actions)[mask > 0].mean()
            entropy = action_preds.entropy()[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            entropy_reg_item = entropy_reg.item()
            if self.no_entropy:
                entropy_reg = 0.0
                entropy_reg_item = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
            self.logger.store(tab="train",
                              nll=-log_likelihood.item(),
                              ent=entropy.item(),
                              ent_reg=entropy_reg_item)
        else:
            act_loss = F.mse_loss(action_preds, actions.detach(), reduction="none")
            # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()


        # [batch_size, seq_len, state_dim]
        state_loss = F.mse_loss(state_preds[:, :-1],
                                states[:, 1:].detach(),
                                reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss

        # PF improvement

        qr_preds, qc_preds = self.model.pred_critics(sc, action_preds_mean) # [batch_size, seq_len]

        # log_lambda = self.model.actor_lag(costs_return.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
        # log_lambda.data.clamp_(min=-20, max=self.max_lag)
        # lambd = log_lambda.exp()
        log_lambd_raw = self.model.actor_lag(costs_return.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
        log_lambd = F.tanh(log_lambd_raw)
        log_lambd = 0.5 * (log_lambd + 1.0) * (self.max_lag - self.min_lag) + self.min_lag
        lambd = torch.exp(log_lambd)
        # lambd = torch.exp(torch.tensor(self.max_lag)) * torch.sigmoid(lambd)
        lambd_detach = lambd.detach()

        qc_loss = lambd_detach * (qc_preds - costs_return)
        q_loss = -qr_preds + qc_loss
        q_loss = q_loss[mask > 0]
        pf_loss = q_loss.mean() / (q_loss.abs().mean().detach() + 1e-8)
        
        loss += self.eta * pf_loss

        self.actor_optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(), self.clip_grad)
        self.actor_optim.step()

        # update Lagrangian multiplier
        loss_lag = (-(lambd * (qc_preds.detach() - costs_return)))[mask > 0]
        loss_lag = loss_lag.mean()
        # for param in self.model.actor_lag.parameters():
        #     loss_lag += 0.01 * torch.norm(param)**2  # L2 regularization
        self.lagrangian_optim.zero_grad()
        loss_lag.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.actor_lag.parameters(), self.clip_grad_critic)
        self.lagrangian_optim.step()

        if self.stochastic:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (self.model.temperature() *
                                (entropy - self.model.target_entropy).detach())
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.cost_critic_scheduler.step()
        self.logger.store(
            tab="train",
            all_loss=loss.item(),
            act_loss=act_loss.item(),
            cost_loss=cost_loss.item(),
            cost_acc=acc.item(),
            state_loss=state_loss.item(),
            train_lr=self.actor_scheduler.get_last_lr()[0],
            critic_lr=self.critic_scheduler.get_last_lr()[0],
            cost_critic_lr=self.cost_critic_scheduler.get_last_lr()[0],
            critic_loss=critic_loss.item(),
            cost_critic_loss=cost_critic_loss.item(),
            q_loss=q_loss.mean().item(),
            loss_lag=loss_lag.item(),
        )

    def evaluate(self, num_rollouts, target_returns, target_cost):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.model, self.env,
                                                      target_returns, target_cost)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(
        self,
        model: PDT,
        env: gym.Env,
        target_returns: np.ndarray,
        target_cost: float,
    ) -> Tuple[float, int, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        states = torch.zeros(1,
                             model.episode_len + 1,
                             model.state_dim,
                             dtype=torch.float,
                             device=self.device)
        actions = torch.zeros(1,
                              model.episode_len,
                              model.action_dim,
                              dtype=torch.float,
                              device=self.device)
        returns = torch.zeros(len(target_returns),
                              model.episode_len + 1,
                              dtype=torch.float,
                              device=self.device)
        costs = torch.zeros(1,
                            model.episode_len + 1,
                            dtype=torch.float,
                            device=self.device)
        time_steps = torch.arange(model.episode_len,
                                  dtype=torch.long,
                                  device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_returns, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            s = states[:, :step + 1][:, -model.seq_len:]  # noqa
            a = actions[:, :step + 1][:, -model.seq_len:]  # noqa
            r = returns[:, :step + 1][:, -model.seq_len:]  # noqa
            c = costs[:, :step + 1][:, -model.seq_len:]  # noqa
            t = time_steps[:, :step + 1][:, -model.seq_len:]  # noqa

            act = model.act(s, a, r, c, t, None, self.logger)
            act = act.clamp(-self.max_action, self.max_action)
            # act = self.get_ensemble_action(1, model, s, a, r, c, t)

            obs_next, reward, terminated, truncated, info = env.step(act.cpu().numpy())
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = act
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost)

            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"] * self.cost_scale

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def get_ensemble_action(self, size: int, model, s, a, r, c, t):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len, act_dim]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)

        acts, _, _ = model(s, a, r, c, t, None)
        if self.stochastic:
            acts = acts.mean

        # [size, seq_len, act_dim]
        acts = torch.mean(acts, dim=0, keepdim=True)
        acts = acts.clamp(-self.max_action, self.max_action)
        act = acts[0, -1].cpu().numpy()
        return act

    def collect_random_rollouts(self, num_rollouts):
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                obs = obs_next
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)