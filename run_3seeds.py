import pandas as pd
import matplotlib.pyplot as plt
from phase4_main import load_pretrained_runner
import os

def main():
    seeds = [42, 123, 999]
    all_dfs = []
    
    for seed in seeds:
        print(f"Running Seed {seed}...")
        runner = load_pretrained_runner({}, seed=seed)
        csv_path = f"seed_{seed}_metrics.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)
        runner.train(num_episodes=50, steps_per_episode=64, save_csv=csv_path)
        
        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_dfs.append(df)
        
    final_df = pd.concat(all_dfs)
    
    # Calculate means over seeds
    mean_df = final_df.groupby('episode', as_index=False).mean(numeric_only=True)
    
    print("\n\n--- FINAL RESULTS ---")
    print(f"Final % valid SMILES (mean): {mean_df['pct_valid'].iloc[-1]:.2f}%")
    print(f"Final Mean Reward (mean): {mean_df['avg_reward'].iloc[-1]:.2f}")
    if 'approx_kl' in mean_df.columns:
        print(f"Final KL Divergence (mean): {mean_df['approx_kl'].iloc[-1]:.4f}")
    if 'policy_entropy' in mean_df.columns:
        print(f"Final Policy Entropy (mean): {mean_df['policy_entropy'].iloc[-1]:.4f}")
    
    # Plot KL divergence curve
    plt.figure(figsize=(9, 5))
    for seed in seeds:
        seed_df = final_df[final_df['seed'] == seed]
        plt.plot(seed_df['episode'], seed_df['approx_kl'], label=f"Seed {seed}")
        
    plt.plot(mean_df['episode'], mean_df['approx_kl'], color='black', linewidth=2, label='Mean KL')
    plt.axhline(0.02, color='red', linestyle='--', label='Target KL')
    plt.title("KL Divergence Curve")
    plt.xlabel("PPO Iteration")
    plt.ylabel("Approx KL")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kl_divergence_curve.png")
    print("Saved kl_divergence_curve.png")

    # Plot validity over time
    plt.figure(figsize=(9, 5))
    for seed in seeds:
        seed_df = final_df[final_df['seed'] == seed]
        plt.plot(seed_df['episode'], seed_df['pct_valid'], label=f"Seed {seed}", alpha=0.7)
    plt.plot(mean_df['episode'], mean_df['pct_valid'], color='black', linewidth=2, label='Mean Validity')
    plt.axhline(60.0, color='green', linestyle='--', label='Target 60%')
    plt.title("Valid SMILES Rate")
    plt.xlabel("PPO Iteration")
    plt.ylabel("% Valid SMILES")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig("validity_curve.png")
    print("Saved validity_curve.png")

if __name__ == "__main__":
    main()
