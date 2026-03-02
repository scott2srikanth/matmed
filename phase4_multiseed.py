"""
phase4_multiseed.py — Phase 4: Multi-Seed Ablation Runner + Statistical Analysis
==================================================================================
Runs all 5 ablation configurations across 3 independent random seeds.
Aggregates results into a master CSV and produces:
  - Table 1: Mean ± Std steady-state rewards per ablation
  - Fig 1: Smoothed convergence plot (averaged over seeds with shaded ±1 std band)
  - Fig 2: Pareto scatter (Validity vs Reward)
  - Significance: Welch's t-test between Full MATMED and each lesion
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils import get_logger, set_seed
from reward import RewardConfig
from train_matmed import MATMEDRunner

logger = get_logger("Phase4-MultiSeed")

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
ABLATIONS = {
    "Full MATMED":       {},
    "No Vision Agent":   {"no_vision": True},
    "No Safety Agent":   {"no_safety": True},
    "No Reaction Agent": {"no_reaction": True},
    "No Binding Agent":  {"no_binding": True},
}

SEEDS      = [42, 0, 7]
N_EPISODES = 50
STEPS_EP   = 64

COLORS = {
    "Full MATMED":       "steelblue",
    "No Vision Agent":   "purple",
    "No Safety Agent":   "tomato",
    "No Reaction Agent": "darkorange",
    "No Binding Agent":  "mediumseagreen",
}

# ---------------------------------------------------------------------------
# Build and run a runner for one (ablation, seed) pair
# ---------------------------------------------------------------------------
def run_one(name: str, cfg: dict, seed: int) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_cfg = RewardConfig(
        alpha=1.0 if not cfg.get("no_binding")  else 0.0,
        beta =1.0 if not cfg.get("no_reaction") else 0.0,
        gamma=1.0 if not cfg.get("no_safety")   else 0.0,
        delta=1.0,
    )
    runner = MATMEDRunner(
        reward_config=reward_cfg,
        num_pretrain_epochs=0,
        seed=seed,
        use_chemberta=False,
        use_vision=not cfg.get("no_vision", False),
        uncertainty_lambda=0.1,
        lr_policy=1e-5,
        entropy_coeff=0.1,
        ppo_clip=0.1,
        kl_coef=0.1,
    )

    # Load the pretrained checkpoint if it exists (best effort, strict=False to skip arch mismatches)
    def _try_load(model, path):
        if os.path.exists(path):
            state = torch.load(path, map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"  ⚠ {path}: missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  ⚠ {path}: unexpected keys: {len(unexpected)}")
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    _try_load(runner.g_agent, "generator_pretrained.pt")
    if not cfg.get("no_binding"):
        _try_load(runner.e_agent, "binding_regressor.pt")
    if not cfg.get("no_safety") and os.path.exists("toxicity_model.pt"):
        runner.s_agent.encoder.load_state_dict(
            torch.load("toxicity_model.pt", map_location=device)
        )
        runner.s_agent.eval()
        for p in runner.s_agent.parameters():
            p.requires_grad = False
    if not cfg.get("no_reaction"):
        _try_load(runner.r_agent, "reaction_model.pt")

    csv_path = f"ms_{name.lower().replace(' ', '_')}_seed{seed}.csv"
    runner.train(num_episodes=N_EPISODES, steps_per_episode=STEPS_EP, save_csv=csv_path)
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------
def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def steady_state(df: pd.DataFrame, n: int = 15) -> pd.Series:
    return df["avg_reward"].tail(n)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    all_data: dict[str, list[pd.DataFrame]] = {k: [] for k in ABLATIONS}

    print("\n=== Phase 4 Multi-Seed Ablation Runner ===")
    for name, cfg in ABLATIONS.items():
        for seed in SEEDS:
            print(f"  Running: {name} | seed={seed}")
            df = run_one(name, cfg, seed)
            all_data[name].append(df)

    # ---------- Figure 1: Convergence with ±1σ band ----------
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, dfs in all_data.items():
        episodes = dfs[0]["episode"].values
        smoothed_runs = np.array([smooth(df["avg_reward"]).values for df in dfs])
        mean_curve = smoothed_runs.mean(axis=0)
        std_curve  = smoothed_runs.std(axis=0)

        color = COLORS[name]
        ax.plot(episodes, mean_curve, label=name, color=color, linewidth=2.5)
        ax.fill_between(episodes,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color, alpha=0.15)

    ax.set_title("MATMED Phase 4 Ablations — Convergence (3-Seed Mean ± 1σ)", fontsize=13, fontweight='bold')
    ax.set_xlabel("RL Iteration (Episode)", fontsize=12)
    ax.set_ylabel("Average Multi-Objective Reward", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("phase4_convergence_multiseed.png", dpi=200)
    print("\nSaved: phase4_convergence_multiseed.png")

    # ---------- Figure 2: Pareto Scatter ----------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    pareto_points = {}
    for name, dfs in all_data.items():
        validity_vals = [df["pct_valid"].tail(15).mean() for df in dfs]
        reward_vals   = [df["avg_reward"].tail(15).mean() for df in dfs]
        px = np.mean(validity_vals)
        py = np.mean(reward_vals)
        pe = np.std(reward_vals)
        pareto_points[name] = (px, py, pe)
        ax2.errorbar(px, py, yerr=pe, fmt='o', color=COLORS[name],
                     markersize=10, capsize=5, label=name, alpha=0.9)
    
    ax2.annotate("Ideal Domain\n(High Validity, High Reward)",
                 xy=(0.93 * max(p[0] for p in pareto_points.values()), 
                     max(p[1] for p in pareto_points.values()) + 0.01),
                 xytext=(40, -0.55),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    ax2.set_title("MATMED Pareto Frontier: Structural Validity vs. Target Optimization", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Structural Validity (% Valid SMILES)", fontsize=11)
    ax2.set_ylabel("Overall Multi-Objective Reward", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("phase4_pareto.png", dpi=200)
    print("Saved: phase4_pareto.png")

    # ---------- Table 1 + Welch's t-test ----------
    print("\n--- TABLE 1: Steady-State Statistical Summary ---")
    full_rewards = np.concatenate([steady_state(df).values for df in all_data["Full MATMED"]])

    rows = []
    for name, dfs in all_data.items():
        pooled = np.concatenate([steady_state(df).values for df in dfs])
        validity_pool = np.concatenate([df["pct_valid"].tail(15).values for df in dfs])

        pct_invalid  = 100.0 - validity_pool.mean()
        # Low feasibility proxy: 1 - (pct_valid * diversity)
        div_pool     = np.concatenate([df["diversity"].tail(15).values for df in dfs])
        stability    = (validity_pool / 100.0) * div_pool
        low_feas     = (1.0 - stability.mean()) * 100.0

        if name == "Full MATMED":
            t_stat, p_val = float("nan"), float("nan")
        else:
            t_stat, p_val = stats.ttest_ind(full_rewards, pooled, equal_var=False)

        sig = ""
        if not np.isnan(p_val):
            if   p_val < 0.001: sig = "***"
            elif p_val < 0.01:  sig = "**"
            elif p_val < 0.05:  sig = "*"
            else:               sig = "ns"

        rows.append({
            "Model":             name,
            "Avg Reward":        f"{pooled.mean():.3f} ± {pooled.std():.3f}",
            "% Invalid SMILES":  f"{pct_invalid:.1f}%",
            "% Low Feasibility": f"{low_feas:.1f}%",
            "vs Full (p-value)": f"{p_val:.4f} {sig}" if not np.isnan(p_val) else "—",
        })

    df_table = pd.DataFrame(rows)
    print(df_table.to_markdown(index=False))
    df_table.to_csv("phase4_table1.csv", index=False)
    print("\nSaved: phase4_table1.csv")


if __name__ == "__main__":
    main()
