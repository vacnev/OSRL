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
pdt_pattern = re.compile(r'Target reward ([-\d.]+),\s*real reward ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*target cost ([-\d.]+),\s*real cost ([-\d.]+),\s*normalized cost: ([-\d.]+)')
ccac_pattern = re.compile(r'Eval reward ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*target cost ([-\d.]+),\s*real cost ([-\d.]+),\s*normalized cost: ([-\d.]+)')
cdt_pattern = re.compile(r'Target reward ([-\d.]+),\s*real reward ([-\d.]+),\s*normalized reward: ([-\d.]+);\s*target cost ([-\d.]+),\s*real cost ([-\d.]+),\s*normalized cost: ([-\d.]+)')

@dataclass
class EvalBatchConfig:
    envs: str = "all"
    folder: Optional[str] = None
    algo_name: str = "pdt"
    use_verification: Optional[bool] = None
    infer_q: Optional[bool] = None
    best: bool = False
    device: str = field(default="cuda")

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

    if args.folder is None:
        if args.algo_name.lower() == 'ccac':
            folder = 'cost-10'
        else:
            folder = 'cost-5'
    else:
        folder = args.folder

    for env_name in envs:
        base_path = f'logs/{env_name}/{folder}'

        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{env_name}', exist_ok=True)

        csv_filename = f'results/{env_name}/{folder}_{args.algo_name}.csv'
        fieldnames = ['algo_name', 'target_reward', 'real_reward', 'normalized_reward', 'target_cost', 'real_cost', 'normalized_cost']

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            seed_avgs_ret = []
            seed_avgs_cost = []
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and args.algo_name.lower() in d.lower()]
            for subdir in subdirs:
                full_path = os.path.join(base_path, subdir)
                config_path = os.path.join(full_path, 'config.yaml')
                if not os.path.exists(config_path):
                    print(f"Config file not found in {full_path}, skipping.")
                    continue

                with open(config_path, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)

                algo_name_raw = config.get('prefix', 'unknown')
                algo_name_lower = algo_name_raw.lower()

                eval_script = f'examples.eval.eval_{args.algo_name.lower()}'
                cmd = ['python3', '-m', eval_script, '--path', full_path, '--device', args.device, '--best', str(args.best).lower()]

                # Add optional overrides for PDT
                if args.algo_name.lower() == 'pdt':
                    if args.use_verification is not None:
                        cmd += ['--use_verification', str(args.use_verification).lower()]
                    if args.infer_q is not None:
                        cmd += ['--infer_q', str(args.infer_q).lower()]
                    targets = PDT_TARGETS[env_name]
                    costs, rets = targets
                    cmd += ['--returns'] + [str(rets)]
                    cmd += ['--costs'] + [str(costs)]

                # Add cost limits for CCAC
                if args.algo_name.lower() == 'ccac':
                    cost_limits, _ = PDT_TARGETS[env_name]
                    cmd += ['--target_costs'] + [str(cost_limits)]

                if args.algo_name.lower() == 'cdt':
                    costs, rets = PDT_TARGETS[env_name]
                    rets = [r[0] for r in rets]  # CDT uses single values
                    cmd += ['--returns'] + [str(rets)]
                    cmd += ['--costs'] + [str(costs)]

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace/OSRL')
                    if result.returncode != 0:
                        print(f"Error running {cmd}: {result.stderr}")
                        continue

                    parsed_results = parse_eval_output(result.stdout)
                    if parsed_results:
                        avg_norm_ret = sum(r['normalized_reward'] for r in parsed_results) / len(parsed_results)
                        avg_norm_cost = sum(r['normalized_cost'] for r in parsed_results) / len(parsed_results)
                        seed_avgs_ret.append(avg_norm_ret)
                        seed_avgs_cost.append(avg_norm_cost)

                        for res in parsed_results:
                            row = {'algo_name': subdir, **res}
                            writer.writerow(row)

                        writer.writerow({
                            'algo_name': f"{subdir}_avg",
                            'target_reward': '',
                            'real_reward': '',
                            'normalized_reward': avg_norm_ret,
                            'target_cost': '',
                            'real_cost': '',
                            'normalized_cost': avg_norm_cost
                        })

                    csvfile.flush()

                except Exception as e:
                    print(f"Exception running {cmd}: {e}")

            if seed_avgs_ret:
                total_avg_norm_ret = np.mean(seed_avgs_ret)
                total_std_norm_ret = np.std(seed_avgs_ret)
                total_avg_norm_cost = np.mean(seed_avgs_cost)
                total_std_norm_cost = np.std(seed_avgs_cost)
                writer.writerow({
                    'algo_name': "total_norm_avg",
                    'target_reward': '',
                    'real_reward': total_avg_norm_ret,
                    'normalized_reward': total_std_norm_ret,
                    'target_cost': '',
                    'real_cost': total_avg_norm_cost,
                    'normalized_cost': total_std_norm_cost
                })

        print(f"Results saved to {csv_filename}")

if __name__ == '__main__':
    eval_batch()
