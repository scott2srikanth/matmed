"""
phase4_plot.py — Phase 4: Ablation Study Plotting
===================================================
Plots the results of the Phase 4 ablation studies (Full MATMED vs No-Safety vs No-Reaction vs No-Binding).
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_phase4_ablations():
    ablation_files = {
        'Full MATMED':        'phase4_metrics_full_matmed.csv',
        'No Safety Agent':    'phase4_metrics_no_safety_agent.csv',
        'No Reaction Agent':  'phase4_metrics_no_reaction_agent.csv',
        'No Binding Agent':   'phase4_metrics_no_binding_agent.csv',
        'No Vision Agent':    'phase4_metrics_no_vision_agent.csv'
    }

    colors = {
        'Full MATMED':        'steelblue',
        'No Safety Agent':    'tomato',
        'No Reaction Agent':  'darkorange',
        'No Binding Agent':   'mediumseagreen',
        'No Vision Agent':    'purple'
    }

    plt.figure(figsize=(10, 6))
    
    for name, filepath in ablation_files.items():
        try:
            df = pd.read_csv(filepath)
            if 'episode' in df.columns and 'avg_reward' in df.columns:
                # Plot raw data with low alpha (transparency)
                plt.plot(df['episode'], df['avg_reward'], color=colors[name], alpha=0.2, linewidth=1)
                
                # Plot smoothed moving average for clearer trends
                window_size = 5
                smoothed = df['avg_reward'].rolling(window=window_size, min_periods=1).mean()
                plt.plot(df['episode'], smoothed, label=f"{name} (MA={window_size})", color=colors[name], linewidth=2.5)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Skipping {name}.")

    plt.title('MATMED Phase 4 Ablations: Multi-Agent Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('RL Iteration (Episode)', fontsize=12)
    plt.ylabel('Average Multi-Objective Reward', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    save_path = 'phase4_ablation_plot.png'
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_phase4_ablations()
