#!/usr/bin/env python3
import os
import subprocess
import csv
import re
import numpy as np
from dataclasses import dataclass
from typing import List

import pyrallis
from pyrallis import field

PDT_TARGETS= {
    "OfflineCarCircle-v0": [[10, 20, 40], [[450.0, 400.0, 350.0], [500.0, 450.0, 400.0], [550.0, 500.0, 450.0]]],
    "OfflineDroneCircle-v0": [[10, 20, 40], [[700.0, 650.0, 600.0], [750.0, 700.0, 650.0], [800.0, 750.0, 700.0]]],
    "OfflineAntCircle-v0": [[10, 20, 40], [[300.0, 250.0, 200.0], [350.0, 300.0, 250.0], [400.0, 350.0, 300.0]]],
    "OfflineBallCircle-v0": [[10, 20, 40], [[700.0, 650.0, 600.0], [750.0, 700.0, 650.0], [800.0, 750.0, 700.0]]],
    "OfflineCarButton1Gymnasium-v0": [[40, 80, 120], [[20.0, 15.0, 10.0], [20.0, 15.0, 10.0], [20.0, 15.0, 10.0]]],
    "OfflineCarButton2Gymnasium-v0": [[40, 80, 120], [[20.0, 15.0, 10.0], [20.0, 15.0, 10.0], [20.0, 15.0, 10.0]]],
    "OfflineCarGoal1Gymnasium-v0": [[40, 80, 120], [[40.0, 35.0, 25.0], [40.0, 35.0, 25.0], [40.0, 35.0, 25.0]]],
    "OfflineCarGoal2Gymnasium-v0": [[40, 80, 120], [[30.0, 25.0, 20.0], [30.0, 25.0, 20.0], [30.0, 25.0, 20.0]]],
    "OfflineCarPush1Gymnasium-v0": [[40, 80, 120], [[15.0, 12.0, 10.0], [15.0, 12.0, 10.0], [15.0, 12.0, 10.0]]],
    "OfflineCarPush2Gymnasium-v0": [[40, 80, 120], [[12.0, 10.0, 8.0], [12.0, 10.0, 8.0], [12.0, 10.0, 8.0]]],
    "OfflineHalfCheetahVelocityGymnasium-v1": [[20, 40, 80], [[3000.0, 2800.0, 2600.0], [3000.0, 2800.0, 2600.0], [3000.0, 2800.0, 2600.0]]],
    "OfflineHopperVelocityGymnasium-v1": [[20, 40, 80], [[2000.0, 1750.0, 1500.0], [2000.0, 1750.0, 1500.0], [2000.0, 1750.0, 1500.0]]],
    "OfflineSwimmerVelocityGymnasium-v1": [[20, 40, 80], [[200.0, 180.0, 160.0], [200.0, 180.0, 160.0], [200.0, 180.0, 160.0]]],
}

BULLET = ["OfflineCarCircle-v0", "OfflineAntCircle-v0", "OfflineDroneCircle-v0", "OfflineBallCircle-v0"]
VELO = ["OfflineHalfCheetahVelocityGymnasium-v1", "OfflineHopperVelocityGymnasium-v1", "OfflineSwimmerVelocityGymnasium-v1"]
NAV = ["OfflineCarButton1Gymnasium-v0", "OfflineCarButton2Gymnasium-v0", "OfflineCarGoal1Gymnasium-v0", "OfflineCarGoal2Gymnasium-v0", "OfflineCarPush1Gymnasium-v0", "OfflineCarPush2Gymnasium-v0"]
ALL = BULLET + VELO + NAV

# Regex patterns for different algorithms
cpq_bearl_pattern = re.compile(r'Eval reward: ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*cost: ([-\d.]+),\s*normalized cost: ([-\d.]+);\s*length: (\d+)')

@dataclass
class EvalBatchConfig:
    envs: str = "all"
    algo_name: str = "cpq"
    cost_limits: List[float] = field(default_factory=lambda: [5, 10, 15])
    best: bool = False
    device: str = field(default="cpu")

def parse_eval_output(output):
    results = []
    lines = output.split('\n')
    for line in lines:
        if 'Eval reward:' in line:
            match = cpq_bearl_pattern.search(line)
            if match:
                real_reward = float(match.group(1))
                normalized_reward = float(match.group(2))
                real_cost = float(match.group(3))
                normalized_cost = float(match.group(4))
                results.append({
                    'target_reward': '',
                    'real_reward': real_reward,
                    'normalized_reward': normalized_reward,
                    'target_cost': '',
                    'real_cost': real_cost,
                    'normalized_cost': normalized_cost
                })
    return results

@pyrallis.wrap()
def eval_batch(args: EvalBatchConfig):
    if args.envs.lower() == 'all':
        envs = ALL
    elif args.envs.lower() == 'bullet':
        envs = BULLET
    elif args.envs.lower() == 'velocity':
        envs = VELO
    elif args.envs.lower() == 'navigation':
        envs = NAV
    else:
        envs = [args.envs]

    for env_name in envs:
        base_path = f'logs/{env_name}'
        if not os.path.isdir(base_path):
            print(f"Error: {base_path} is not a directory.")
            return

        cost_limits, _ = PDT_TARGETS[env_name]

        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{env_name}', exist_ok=True)

        csv_filename = f'results/{env_name}/{args.algo_name}.csv'
        fieldnames = ['algo_name', 'target_reward', 'real_reward', 'normalized_reward', 'target_cost', 'real_cost', 'normalized_cost']

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            all_norm_ret = []
            all_norm_cost = []
            for cost in cost_limits:
                cost_dir = f"cost-{int(cost)}"
                cost_path = os.path.join(base_path, cost_dir)
                if not os.path.isdir(cost_path):
                    print(f"Cost dir {cost_path} not found, skipping.")
                    continue

                subdirs = [d for d in os.listdir(cost_path) if os.path.isdir(os.path.join(cost_path, d)) and args.algo_name.lower() in d.lower()]
                for subdir in subdirs:
                    full_path = os.path.join(cost_path, subdir)
                    eval_script = f'examples.eval.eval_{args.algo_name.lower()}'
                    cmd = ['python3', '-m', eval_script, '--path', full_path, '--device', args.device, '--best', str(args.best).lower()]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace/OSRL')
                        if result.returncode != 0:
                            print(f"Error running {cmd}: {result.stderr}")
                            continue

                        parsed_results = parse_eval_output(result.stdout)
                        if parsed_results:
                            res = parsed_results[0]  # Single result for CPQ/BEARL
                            all_norm_ret.append(res['normalized_reward'])
                            all_norm_cost.append(res['normalized_cost'])

                            row = {'algo_name': f"{cost_dir}/{subdir}", **res}
                            writer.writerow(row)

                        csvfile.flush()

                    except Exception as e:
                        print(f"Exception running {cmd}: {e}")

            if all_norm_ret:
                total_avg_norm_ret = np.mean(all_norm_ret)
                total_avg_norm_cost = np.mean(all_norm_cost)
                writer.writerow({
                    'algo_name': "total_avg",
                    'target_reward': '',
                    'real_reward': total_avg_norm_ret,
                    'normalized_reward': '',
                    'target_cost': '',
                    'real_cost': total_avg_norm_cost,
                    'normalized_cost': ''
                })

        print(f"Results saved to {csv_filename}")

if __name__ == '__main__':
    eval_batch()
