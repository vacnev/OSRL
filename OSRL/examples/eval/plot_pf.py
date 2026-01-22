#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def plot_pf():
    envs = ['OfflineAntCircle-v0', 'OfflineDroneCircle-v0',
            'OfflineCarCircle-v0', 'OfflineBallCircle-v0', 
            'OfflineHalfCheetahVelocityGymnasium-v1',
            'OfflineHopperVelocityGymnasium-v1',
            'OfflineSwimmerVelocityGymnasium-v1',
            ]

    algos = ['cost-10_ccac', 'cost-5_cdt', 'cost-1_pdt', 'trebi']

    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # nice color scheme

    """
        SETUP FIGURE HERE
    """
    fig_width = 8
    fig_height = 2.5

    axis_fontsize = 8
    title_fontsize = 8

    label_ticksize = 6
    tick_padding = -2.5



    fig, axes = plt.subplots(2, len(envs), figsize=(fig_width, fig_height), 
                             sharex='col')

    for row in axes:
        for ax in row:
            ax.tick_params(axis='both', labelsize=label_ticksize, pad=tick_padding)

    for i, env in enumerate(envs):
        env_clean = env.replace('Offline','').split('-')[0].replace("VelocityGymnasium", "-Vel")

        for j, (algo, color) in enumerate(zip(algos, colors)):
            print(algo)
            csv_path = f'results/pf/{env}/{algo}.csv'

            if not os.path.exists(csv_path):
                print(f"CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            df = df.sort_values(by='target_cost').reset_index(drop=True)

            """
                EMA:
            """
            smooth_alpha = 0.3
            df['reward_mean'] = df['reward_mean'].ewm(alpha=smooth_alpha, adjust=False).mean()
            df['cost_mean']   = df['cost_mean'].ewm(alpha=smooth_alpha, adjust=False).mean()
            df['reward_std']  = df['reward_std'].ewm(alpha=smooth_alpha, adjust=False).mean()
            df['cost_std']    = df['cost_std'].ewm(alpha=smooth_alpha, adjust=False).mean()

            """
                Window mean
            window_size = 10
            df['reward_mean'] = df['reward_mean'].rolling(window=window_size,
                                                          min_periods=1).mean()
            df['cost_mean']   = df['cost_mean'].rolling(window=window_size,
                                                        min_periods=1).mean()
            df['reward_std']  = df['reward_std'].rolling(window=window_size,
                                                        min_periods=1).mean()
            df['cost_std']    = df['cost_std'].rolling(window=window_size,
                                                       min_periods=1).mean()
            """

            # transparency of error bars
            alpha = 0.2

            # ---------- Reward row (row 0) ----------
            axes[0, i].plot(
                df['target_cost'], df['reward_mean'],
                label=f'{algo.upper()}' if i == 0 else "",
                color=color, alpha=0.7
            )
            axes[0, i].fill_between(
                df['target_cost'],
                df['reward_mean'] - df['reward_std'],
                df['reward_mean'] + df['reward_std'],
                color=color, alpha=alpha
            )

            # ---------- Cost row (row 1) ----------
            axes[1, i].plot(
                df['target_cost'], df['cost_mean'],
                label=f'{algo.upper()}' if i == 0 else "",
                color=color, alpha=0.7
            )
            axes[1, i].fill_between(
                df['target_cost'],
                df['cost_mean'] - df['cost_std'],
                df['cost_mean'] + df['cost_std'],
                color=color, alpha=alpha
            )

            if j == 0:
                axes[1, i].plot(
                    df['target_cost'], df['target_cost'],
                    'k--',
                    label='Cost budget == Eval cost' if i == 0 else ""
                )

        # Column titles (envs)
        axes[0, i].set_title(env_clean, fontsize=axis_fontsize)

        # clip y axis to max_cost + 20
        if not df.empty:
            max_cost = df['target_cost'].max()
            axes[1, i].set_ylim(0, max_cost + 30)


    # Row labels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    axes[0, 0].set_ylabel('Eval reward', fontsize=title_fontsize)
    axes[1, 0].set_ylabel('Eval cost', fontsize=title_fontsize)

    # X labels
    #for i in range(len(envs)):
        #axes[1, i].set_xlabel('Cost budget', fontsize=16)

    # shared x-axis label
    fig.text(0.48, 0.0, 'Cost budget', ha='center', fontsize=title_fontsize)

    # Legend
    unique = dict(zip(labels + labels2, handles + handles2))

    # Global legend below the figure


    # rename keys in unique, remove underscores
    unique_renamed = {}
    for key in unique.keys():
        split = key.split('_')
        if len(split) > 1:
            new_key = split[1]
        else:
            new_key = split[0]
        unique_renamed[new_key] = unique[key]
    unique = unique_renamed

    legend = fig.legend(
        handles=list(unique.values()),
        labels=list(unique.keys()),
        loc='upper center',
        ncol=len(algos),               # one column per algorithm
        frameon=True,
        fancybox=True,
        framealpha=0.9,                # semi-transparent box
        columnspacing=1.0,             # spacing between columns
        borderpad=0.75,                # padding inside the box
        fontsize=8
    )

    fig.subplots_adjust(
        left=0,    # reduce left margin
        right=0.95,   # reduce right margin
        top=0.78,
        bottom=0.10,  # make space for shared x-label
        hspace=0.1,  # vertical spacing between rows
        wspace=0.25   # horizontal spacing between columns
    )
    plt.savefig('pf_plt.svg', bbox_inches='tight')

if __name__ == '__main__':
    plot_pf()
