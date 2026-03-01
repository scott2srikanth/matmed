import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(csv_path: str = 'matmed_metrics.csv', output_path: str = 'matmed_training_plot.png'):
    """Plot training metrics from the CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run train_matmed.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # We expect columns: iteration, avg_reward, best_reward, valid_pct, diversity, loss
    if 'iteration' not in df.columns or len(df) == 0:
        print("CSV appears empty or malformed.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('MATMED RL Training Metrics')

    # 1. Rewards
    axs[0, 0].plot(df['iteration'], df['avg_reward'], label='Avg Reward', color='blue', alpha=0.7)
    axs[0, 0].plot(df['iteration'], df['best_reward'], label='Best Reward', color='green', linewidth=2)
    axs[0, 0].set_title('Reward Optimization')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Validity
    axs[0, 1].plot(df['iteration'], df['valid_pct'], color='purple', linewidth=2)
    axs[0, 1].set_title('SMILES Validity %')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Valid %')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Diversity
    axs[1, 0].plot(df['iteration'], df['diversity'], color='orange', linewidth=2)
    axs[1, 0].set_title('Chemical Diversity (Tanimoto)')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Diversity Score')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Generator Loss
    if 'loss' in df.columns:
        axs[1, 1].plot(df['iteration'], df['loss'], color='red')
        axs[1, 1].set_title('Generator Generator Loss')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Loss')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    plot_metrics()
