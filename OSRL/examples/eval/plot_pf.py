#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def plot_pf():
    envs = ['OfflineAntCircle-v0', 'OfflineDroneCircle-v0', 'OfflineCarCircle-v0', 'OfflineBallCircle-v0']
    algos = ['ccac', 'cdt', 'pdt']
    colors = ['blue', 'green', 'red']

    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    # fig.suptitle('Pareto Front Comparison', fontsize=16)

    for i, env in enumerate(envs):
        for j, (algo, color) in enumerate(zip(algos, colors)):
            csv_path = f'results/pf/{env}/{algo}.csv'
            if not os.path.exists(csv_path):
                print(f"CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            env_clean = env.replace('Offline', '')
            env_clean = env_clean.replace('-v0', '')

            # Reward plot
            axes[i, 0].plot(df['target_cost'], df['reward_mean'], label=f'{algo.upper()}' if i == 0 else "", color=color, alpha=0.7)
            axes[i, 0].fill_between(df['target_cost'], 
                                     df['reward_mean'] - df['reward_std'], 
                                     df['reward_mean'] + df['reward_std'], 
                                     color=color, alpha=0.2)
            if j == 0:  # Only set labels once
                axes[i, 0].set_xlabel('Cost budget', fontsize=16)
                axes[i, 0].set_ylabel('Eval reward', fontsize=16)
                axes[i, 0].set_title(f'{env_clean}', fontsize=16)

            # Cost plot
            axes[i, 1].plot(df['target_cost'], df['cost_mean'], label=f'{algo.upper()}' if i == 0 else "", color=color, alpha=0.7)
            axes[i, 1].fill_between(df['target_cost'], 
                                     df['cost_mean'] - df['cost_std'], 
                                     df['cost_mean'] + df['cost_std'], 
                                     color=color, alpha=0.2)
            axes[i, 1].plot(df['target_cost'], df['target_cost'], 'k--', label='Cost budget == Eval cost' if i == 0 and j == 0 else "")
            if j == 0:
                axes[i, 1].set_xlabel('Cost budget', fontsize=16)
                axes[i, 1].set_ylabel('Eval cost', fontsize=16)
                axes[i, 1].set_title(f'{env_clean}', fontsize=16)

    # Legend under the plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    handles += handles2
    labels += labels2
    # Remove duplicates
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=16)

    plt.tight_layout()
    plt.savefig('results/pf/pf_plt.svg', bbox_inches='tight')
    print("Plot saved to results/pf/pf_plt.svg")

if __name__ == '__main__':
    plot_pf()
