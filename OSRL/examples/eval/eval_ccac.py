import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dsrl.infos import DENSITY_CFG, MAX_EPISODE_COST

import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.common.dataset import set_cost_thresholds
from osrl.algorithms import CCAC, CCACTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    target_costs: List[float] = field(default=[5], is_mutable=True)
    noise_scale: List[float] = None
    eval_episodes: int = 20
    best: bool = False
    mode: str = "offline"
    device: str = "cpu"
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = gym.make(cfg["task"])
    data = env.get_dataset()
    state_min, state_max = set_cost_thresholds(data)
    env = wrap_env(
        env,
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
        
    ccac_model = CCAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        state_min=state_min,
        state_max=state_max,
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        vae_hidden_sizes=cfg["vae_hidden_sizes"],
        vae_latent_sizes=cfg["vae_latent_sizes"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        beta=cfg["beta"],
        num_q=cfg["num_q"],
        num_qc=cfg["num_qc"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    ccac_model.load_state_dict(model["model_state"])
    ccac_model.to(args.device)

    trainer = CCACTrainer(ccac_model,
                         env,
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         device=args.device)
    
    for target_cost in args.target_costs:
        seed_all(cfg["seed"])
        env.set_target_cost(target_cost)
        ret, cost, length = trainer.evaluate(args.eval_episodes, target_cost * cfg["cost_scale"])
        normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
        print(
            f"Eval reward {ret}, normalized reward: {normalized_ret}; target cost {target_cost}, real cost {cost}, normalized cost: {normalized_cost}"
        )


if __name__ == "__main__":
    eval()
