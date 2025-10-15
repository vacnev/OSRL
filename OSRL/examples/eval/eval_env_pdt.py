#!/usr/bin/env python3
import os
import subprocess
import yaml
import csv
import re
from dataclasses import dataclass
from typing import Optional

import pyrallis
from pyrallis import field

@dataclass
class EvalBatchConfig:
    base_path: str = "logs/OfflineAntCircle-v0/cost-5"
    use_verification: Optional[bool] = None
    infer_q: Optional[bool] = None
    best: bool = True
    device: str = field(default="cuda")

def parse_eval_output(output):
    results = []
    lines = output.split('\n')
    for line in lines:
        if 'Target reward' in line:
            match = re.search(r'Target reward\s*(.+?),\s*real reward ([-\d.]+), normalized reward: ([-\d.]+); target cost (\d+), real cost ([-\d.]+), normalized cost: ([-\d.]+)', line)
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
    return results

@pyrallis.wrap()
def eval_batch(args: EvalBatchConfig):
    base_path = args.base_path
    if not os.path.isdir(base_path):
        print(f"Error: {base_path} is not a directory.")
        return

    # Extract env_name: from logs/OfflineAntCircle-v0/cost-5, env_name = OfflineAntCircle-v0
    env_name = os.path.basename(os.path.dirname(base_path))
    exp_name = os.path.basename(base_path)

    os.makedirs('results', exist_ok=True)

    csv_filename = f'results/{env_name}_{exp_name}.csv'
    fieldnames = ['algo_name', 'target_reward', 'real_reward', 'normalized_reward', 'target_cost', 'real_cost', 'normalized_cost']

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
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

            eval_script = f'examples.eval.eval_pdt'
            cmd = ['python3', '-m', eval_script, '--path', full_path, '--device', args.device, '--best', str(args.best).lower()]

            # Add optional overrides
            if args.use_verification is not None:
                cmd += ['--use_verification', str(args.use_verification).lower()]
            if args.infer_q is not None:
                cmd += ['--infer_q', str(args.infer_q).lower()]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd='/workspace/OSRL')
                if result.returncode != 0:
                    print(f"Error running {cmd}: {result.stderr}")
                    continue

                parsed_results = parse_eval_output(result.stdout)
                norm_ret, norm_cost = 0, 0
                for res in parsed_results:
                    row = {'algo_name': subdir, **res}
                    writer.writerow(row)
                    norm_ret += res['normalized_reward']
                    norm_cost += res['normalized_cost']

                if parsed_results:
                    avg_norm_ret = norm_ret / len(parsed_results)
                    avg_norm_cost = norm_cost / len(parsed_results)
                    print(f"Avg normalized reward for {subdir}: {avg_norm_ret}, Avg normalized cost: {avg_norm_cost}")

                csvfile.flush()

            except Exception as e:
                print(f"Exception running {cmd}: {e}")

    print(f"Results saved to {csv_filename}")

if __name__ == '__main__':
    eval_batch()
