import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG, MAX_EPISODE_COST
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger, DummyLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.ccac_configs import CCAC_DEFAULT_CONFIG, CCACTrainConfig
from osrl.algorithms import CCAC, CCACTrainer
from osrl.common import TransitionDataset
from osrl.common.dataset import set_cost_thresholds
from osrl.common.exp_util import auto_name, seed_all


@pyrallis.wrap()
def train(args: CCACTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(CCACTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(CCAC_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(CCAC_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "/cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = DummyLogger()
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym  # noqa
    env = gym.make(args.task)

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)
    state_min, state_max = set_cost_thresholds(data)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    # model & optimizer setup
    model = CCAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        state_min=state_min,
        state_max=state_max,
        a_hidden_sizes=args.a_hidden_sizes,
        c_hidden_sizes=args.c_hidden_sizes,
        vae_hidden_sizes=args.vae_hidden_sizes,
        vae_latent_sizes=args.vae_latent_sizes,
        cost_conditioned=args.cost_conditioned,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        num_q=args.num_q,
        num_qc=args.num_qc,
        ood_threshold=args.ood_threshold,
        max_lag=args.max_lag,
        episode_len=args.episode_len,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict(),
                "actor_optim_state": model.actor_optim.state_dict(),
                "critic_optim_state": model.critic_optim.state_dict(),
                "cost_critic_optim_state": model.cost_critic_optim.state_dict(),
                "cvae_optim_state": model.cvae_optim.state_dict(),
                "classifier_optim_state": model.classifier.state_dict(),
                "lagrangian_optim_state": model.lagrangian_optim.state_dict(),}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = CCACTrainer(model,
                         env,
                         logger=logger,
                         actor_lr=args.actor_lr,
                         critic_lr=args.critic_lr,
                         vae_lr=args.vae_lr,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         device=args.device)

    dataset = TransitionDataset(data,
                                reward_scale=args.reward_scale,
                                cost_scale=args.cost_scale)
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)
    
    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    for _ in trange(args.update_steps, desc="CVAE warmup"):
        batch = next(trainloader_iter)
        observations, next_observations, actions, rewards, costs, \
            cost_thresholds, next_cost_thresholds, done = [b.to(args.device) for b in batch]
        trainer.train_cvae_only(observations, actions, cost_thresholds)

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        observations, next_observations, actions, rewards, costs, \
            cost_thresholds, next_cost_thresholds, done = [b.to(args.device) for b in batch]
        trainer.train_one_step_cvae(observations, next_observations, actions, 
                        rewards, costs, cost_thresholds, next_cost_thresholds, done)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            ret, cost, length = trainer.evaluate(args.eval_episodes, 
                                                 args.cost_limit * args.cost_scale)
            logger.store(tab="eval", Cost=cost, Reward=ret, Length=length)

            # save the current weight
            logger.save_checkpoint()
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)
        
    
if __name__ == "__main__":
    train()
