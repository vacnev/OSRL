#!/usr/bin/env python3
import os
import subprocess
import yaml
import csv
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

import pyrallis
from pyrallis import field

TARGETS= {
    "OfflineCarCircle-v0": [[10, 50], [450.0, 550.0]],
    "OfflineDroneCircle-v0": [[10, 50], [700.0, 800.0]],
    "OfflineAntCircle-v0": [[10, 50], [300.0, 400.0]],
    "OfflineBallCircle-v0": [[10, 50], [700.0, 800.0]],
    "OfflineHalfCheetahVelocityGymnasium-v1": [[20, 80], [3000.0, 3000.0]],
    "OfflineHopperVelocityGymnasium-v1": [[20, 80], [2000.0, 2000.0]],
    "OfflineSwimmerVelocityGymnasium-v1": [[20, 80], [200.0, 200.0]],
}

BULLET = ["OfflineCarCircle-v0", "OfflineAntCircle-v0", "OfflineDroneCircle-v0", "OfflineBallCircle-v0"]
VELO = ["OfflineHalfCheetahVelocityGymnasium-v1", "OfflineHopperVelocityGymnasium-v1", "OfflineSwimmerVelocityGymnasium-v1"]
ALL = BULLET + VELO

# Regex patterns for different algorithms
pdt_pattern = re.compile(r'Target reward ([-\d.]+),\s*real reward ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*target cost ([-\d.]+),\s*real cost ([-\d.]+),\s*normalized cost: ([-\d.]+)')
ccac_pattern = re.compile(r'Eval reward ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*target cost ([-\d.]+),\s*real cost ([-\d.]+),\s*normalized cost: ([-\d.]+)')
cdt_pattern = re.compile(r'Target reward ([-\d.]+),\s*real reward ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*target cost ([-\d.]+),\s*real cost ([-\d.]+),\s*normalized cost: ([-\d.]+)')

@dataclass
class PFConfig:
    algo_name: str = "pdt"
    envs: str = "all"
    folder: str = None
    best: bool = False
    device: str = "cuda"

def parse_eval_output(output):
    results = []
    lines = output.split('\n')
    for line in lines:
        if 'Target reward' in line:
            if '(' in line:
                # PDT
                match = pdt_pattern.search(line)
                if match:
                    target_reward_str = match.group(1).strip('()')
                    # Take the first value if multiple
                    first_value = target_reward_str.split(',')[0]
                    target_reward = float(first_value)
                    real_reward = float(match.group(2))
                    normalized_reward = float(match.group(3))
                    target_cost = float(match.group(4))
                    real_cost = float(match.group(5))
                    normalized_cost = float(match.group(6))
                    results.append({
                        'target_reward': target_reward,
                        'real_reward': real_reward,
                        'normalized_reward': normalized_reward,
                        'target_cost': target_cost,
                        'real_cost': real_cost,
                        'normalized_cost': normalized_cost
                    })
            else:
                # CDT
                match = cdt_pattern.search(line)
                if match:
                    target_reward = float(match.group(1))
                    real_reward = float(match.group(2))
                    normalized_reward = float(match.group(3))
                    target_cost = float(match.group(4))
                    real_cost = float(match.group(5))
                    normalized_cost = float(match.group(6))
                    results.append({
                        'target_reward': target_reward,
                        'real_reward': real_reward,
                        'normalized_reward': normalized_reward,
                        'target_cost': target_cost,
                        'real_cost': real_cost,
                        'normalized_cost': normalized_cost
                    })
        elif 'Eval reward' in line:
            if 'target cost' in line:
                # CCAC
                match = ccac_pattern.search(line)
                if match:
                    real_reward = float(match.group(1))
                    normalized_reward = float(match.group(2))
                    target_cost = float(match.group(3))
                    real_cost = float(match.group(4))
                    normalized_cost = float(match.group(5))
                    results.append({
                        'target_reward': '',
                        'real_reward': real_reward,
                        'normalized_reward': normalized_reward,
                        'target_cost': target_cost,
                        'real_cost': real_cost,
                        'normalized_cost': normalized_cost
                    })
    return results

@pyrallis.wrap()
def eval_pf(args: PFConfig):
    if args.envs.lower() == 'all':
        envs = ALL
    elif args.envs.lower() == 'bullet':
        envs = BULLET
    elif args.envs.lower() == 'velocity':
        envs = VELO
    else:
        envs = [args.envs]

    if args.folder is None:
        if args.algo_name.lower() == 'ccac':
            folder = 'cost-10'
        else:
            folder = 'cost-5'
    else:
        folder = args.folder

    for env_name in envs:
        base_path = f'logs/{env_name}/{folder}'

        os.makedirs(f'results/pf/{env_name}', exist_ok=True)

        detailed_csv = f'results/pf/{env_name}/{folder}_{args.algo_name}_detailed.csv'
        fieldnames = ['algo_name', 'target_reward', 'real_reward', 'normalized_reward', 'target_cost', 'real_cost', 'normalized_cost']

        with open(detailed_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            thds = TARGETS[env_name][0]
            min_thd, max_thd = thds
            results_per_cost = {cost: {'rewards': [], 'costs': []} for cost in range(min_thd, max_thd + 1)}

            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and args.algo_name.lower() in d.lower()]
            for subdir in subdirs:
                full_path = os.path.join(base_path, subdir)
        
                cost_list = [c for c in range(min_thd, max_thd + 1)]
                returns_list = []
                if args.algo_name.lower() in ['pdt', 'cdt']:
                    rets = TARGETS[env_name][1]
                    min_ret, max_ret = rets
                    returns_list = np.linspace(min_ret, max_ret, len(cost_list)).astype(int).tolist()

                cmd = ['python3', '-m', f'examples.eval.eval_{args.algo_name.lower()}', '--path', full_path, '--device', args.device, '--best', str(args.best).lower()]

                if args.algo_name.lower() == 'ccac':
                    cmd += ['--target_costs'] + [str(cost_list)]
                else:
                    cmd += ['--costs'] + [str(cost_list)]
                    if returns_list:
                        if args.algo_name.lower() == 'pdt':
                            returns_list = [[ret, ret - 50, ret - 100] for ret in returns_list]
                            cmd += ['--returns'] + [str(returns_list)]
                        else:
                            cmd += ['--returns'] + [str(returns_list)]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace/OSRL')
                    if result.returncode != 0:
                        print(f"Error running {cmd}: {result.stderr}")
                        continue

                    parsed_results = parse_eval_output(result.stdout)
                    for res in parsed_results:
                        cost = int(float(res['target_cost']))  # Assuming target_cost is present
                        if cost in results_per_cost:
                            results_per_cost[cost]['rewards'].append(res['real_reward'])
                            results_per_cost[cost]['costs'].append(res['real_cost'])
                            writer.writerow({'algo_name': subdir, **res})

                except Exception as e:
                    print(f"Exception: {e}")

        # Save averaged results
        avg_csv = f'results/pf/{env_name}/{folder}_{args.algo_name}.csv'
        with open(avg_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['target_cost', 'cost_mean', 'cost_std', 'reward_mean', 'reward_std'])
            for cost, data in results_per_cost.items():
                if data['rewards'] and data['costs']:
                    reward_mean = np.mean(data['rewards'])
                    reward_std = np.std(data['rewards'])
                    cost_mean = np.mean(data['costs'])
                    cost_std = np.std(data['costs'])
                    writer.writerow([cost, cost_mean, cost_std, reward_mean, reward_std])

        print(f"Detailed results saved to {detailed_csv}")
        print(f"Averaged results saved to {avg_csv}")

if __name__ == '__main__':
    eval_pf()