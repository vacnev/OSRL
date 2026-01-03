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
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.pdt_configs import PDT_DEFAULT_CONFIG, PDTTrainConfig
from osrl.algorithms import PDT, PDTTrainer
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all

# torch.autograd.set_detect_anomaly(True)

@pyrallis.wrap()
def train(args: PDTTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(PDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(PDT_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(PDT_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "/cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
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

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    # model & optimizer & scheduler setup
    model = PDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=args.embedding_dim,
        seq_len=args.seq_len,
        episode_len=args.episode_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        time_emb=args.time_emb,
        use_rew=args.use_rew,
        use_cost=args.use_cost,
        cost_transform=args.cost_transform,
        add_cost_feat=args.add_cost_feat,
        mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat,
        action_head_layers=args.action_head_layers,
        stochastic=args.stochastic,
        init_temperature=args.init_temperature,
        target_entropy=-env.action_space.shape[0],
        num_qr=args.num_qr,
        num_qc=args.num_qc,
        c_hidden_sizes=args.c_hidden_sizes,
        tau=args.tau,
        gamma=args.gamma,
        use_verification=args.use_verification,
        infer_q=args.infer_q,
    ).to(args.device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # model.compile()

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = PDTTrainer(model,
                         env,
                         logger=logger,
                         actor_lr=args.actor_lr,
                         critic_lr=args.critic_lr,
                         weight_decay=args.weight_decay,
                         betas=args.betas,
                         clip_grad=args.clip_grad,
                         lr_warmup_steps=args.lr_warmup_steps,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         loss_cost_weight=args.loss_cost_weight,
                         loss_state_weight=args.loss_state_weight,
                         eta=args.eta,
                         cost_reverse=args.cost_reverse,
                         no_entropy=args.no_entropy,
                         n_step=args.n_step,
                         batch_size=args.batch_size,
                         seq_len=args.seq_len,
                         device=args.device)

    max_cost = min(args.max_cost + 1, 125)
    ct = lambda x: max_cost - x if args.linear else 1 / (x + 10)

    dataset = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        deg=args.deg,
        pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease,
        beta=args.beta,
        augment_percent=args.augment_percent,
        cost_reverse=args.cost_reverse,
        max_reward=args.max_reward,
        min_reward=args.min_reward,
        pf_only=args.pf_only,
        rmin=args.rmin,
        cost_bins=args.cost_bins,
        npb=args.npb,
        cost_sample=args.cost_sample,
        cost_transform=ct,
        start_sampling=args.start_sampling,
        prob=args.prob,
        random_aug=args.random_aug,
        aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax,
        aug_cmin=args.aug_cmin,
        aug_cmax=args.aug_cmax,
        cgap=args.cgap,
        rstd=args.rstd,
        cstd=args.cstd,
    )

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
    best_safe = False
    best_idx = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            b.to(args.device) for b in batch
        ]
        
        # with torch.cuda.amp.autocast():
        trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
                               costs)
        
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_profiler/pdt'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
        #     for _ in range(5):
        #         prof.step()
        #         trainer.train_one_step(states, actions, returns, costs_return, time_steps, mask,
        #                                costs)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}
            log_safe_cost, log_safe_reward = {}, {}
            safe = True
            for target_return in args.target_returns:
                reward_returns, cost_return = target_return
                if args.cost_reverse:
                    # critical step, rescale the return!
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, np.array(reward_returns) * args.reward_scale,
                        (args.episode_len - cost_return) * args.cost_scale)
                else:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, np.array(reward_returns) * args.reward_scale,
                        cost_return * args.cost_scale)
                average_cost.append(cost)
                average_reward.append(ret)
                safe &= (cost <= cost_return)

                name = "c_" + str(int(cost_return)) + "_r_" + str(int(reward_returns[0]))
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})

                if cost <= cost_return:
                    log_safe_cost.update({name: cost})
                    log_safe_reward.update({name: ret})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="length", **log_len)
            logger.store(tab="safe_cost", **log_safe_cost)
            logger.store(tab="safe_ret", **log_safe_reward)

            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            mean_ret = np.mean(average_reward)
            mean_cost = np.mean(average_cost)
            if (not safe and not best_safe and (mean_cost < best_cost or (mean_cost == best_cost and mean_ret > best_reward))) \
                or (safe and (mean_ret > best_reward or not best_safe)):
                best_cost = mean_cost
                best_reward = mean_ret
                best_safe = safe
                best_idx = step
                logger.save_checkpoint(suffix="best")

            # safe halfway model
            if (step + 1) == args.update_steps // 2:
                logger.save_checkpoint(suffix="halfway")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
    import wandb
    wandb.finish()