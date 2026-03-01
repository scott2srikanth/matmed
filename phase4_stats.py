import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_stats():
    # Map the trial names to the actual CSV output files written by Phase 4
    files = {
        'Full MATMED': 'phase4_metrics_full_matmed.csv',
        'No Vision': 'phase4_metrics_no_vision_agent.csv',
        'No Safety': 'phase4_metrics_no_safety_agent.csv',
        'No Reaction': 'phase4_metrics_no_reaction_agent.csv',
        'No Binding': 'phase4_metrics_no_binding_agent.csv'
    }
    
    results = []
    pareto_data = {}
    
    for name, f in files.items():
        if not os.path.exists(f): 
            print(f"Skipping {name} - file {f} not found.")
            continue
            
        df = pd.read_csv(f)
        
        # Calculate stats over the last 15 episodes (steady state) to reflect final convergence
        steady_state = df.tail(15)
        
        avg_rew = steady_state['avg_reward'].mean()
        std_rew = steady_state['avg_reward'].std()
        
        # invalid smiles = 100 - pct_valid
        pct_invalid = 100.0 - steady_state['pct_valid'].mean()
        
        # Low feasibility proxy (Higher diversity + valid often implies better exploration/stability)
        stability = steady_state['diversity'].mean() * (steady_state['pct_valid'].mean() / 100.0)
        low_feasibility = (1.0 - stability) * 100.0
        
        results.append({
            'Model': name,
            'Final Avg Reward': f"{avg_rew:.2f} \u00B1 {std_rew:.2f}",
            'Std Dev': f"{std_rew:.3f}",
            '% Invalid SMILES': f"{pct_invalid:.1f}%",
            '% Low Feasibility': f"{low_feasibility:.1f}%"
        })
        
        # Save pareto metrics (Valid % vs Reward)
        pareto_data[name] = {
            'x': steady_state['pct_valid'].mean(), # Proxy for structural/reaction feasibility
            'y': avg_rew                           # Overall reward (binding + safety)
        }
        
    df_res = pd.DataFrame(results)
    print("\n--- STATISTICAL RESULTS FOR PAPER ---")
    print(df_res.to_markdown(index=False))
    
    # Generate Pareto Plot
    plt.figure(figsize=(8,6))
    colors = {'Full MATMED': 'blue', 'No Vision': 'purple', 'No Safety': 'red', 'No Reaction': 'orange', 'No Binding': 'green'}
    
    for name, metrics in pareto_data.items():
        if name in colors:
            plt.scatter(metrics['x'], metrics['y'], s=150, label=name, color=colors[name], alpha=0.8, edgecolors='black')
        
    plt.title('MATMED Pareto Frontier: Structural Validity vs. Target Optimization', fontsize=13, fontweight='bold')
    plt.xlabel('Structural Validity (% Valid SMILES Output)', fontsize=11)
    plt.ylabel('Overall Multi-Objective Reward', fontsize=11)
    
    # Add a dashed line showing the ideal Pareto frontier direction
    plt.annotate('Ideal Domain\n(High Validity, High Reward)', xy=(95, -0.6), xytext=(70, -0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=1))

    plt.grid(alpha=0.3)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('phase4_pareto.png', dpi=200)
    print("\nSaved Pareto plot to phase4_pareto.png")

if __name__ == '__main__':
    compute_stats()
