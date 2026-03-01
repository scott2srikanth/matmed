# MATMED: Grounding Multi-Agent Language Models in Physical Reality via Physics-Inspired Visual Reaction Simulators

## Abstract
Recent advances in autoregressive generation of Simplified Molecular-Input Line-Entry System (SMILES) strings have produced deep learning models capable of hallucinating vast libraries of novel chemical matter. However, without grounding in physical reality, language-model-based generators frequently propose molecules that are fundamentally impossible or economically unviable to synthesize in a wet lab. We introduce **MATMED** (Multi-Agent Transformer for Molecular Evolution & Design), a modular reinforcement learning framework where a central Policy Agent coordinates four domain-specific expert critics. Crucially, MATMED incorporates a Vision-Temporal Transformer (VTT) that explicitly models the spatio-temporal dynamics of physics-inspired liquid-phase reaction trajectories (e.g., simulated colorimetry, turbidity gradients). By penalizing the Generator using multi-objective Proximal Policy Optimization (PPO) informed by these simulated reaction videos, MATMED significantly restricts the action space to synthetically accessible, high-binding, non-toxic drug candidates. Ablation studies demonstrate that dropping the temporal visual modality from the RL reward signal strictly degrades convergence smoothness and reduces the stable reward ceiling, indicating that physics-inspired visual heuristics contribute stabilizing regulatory information beyond structural graph-based heuristics alone.

---

## 1. Introduction
The design of novel therapeutics involves navigating a chemical space estimated to contain $10^{60}$ possible compounds. Deep generative models—most notably recurrent neural networks and Transformers trained on SMILES strings—have shown immense promise in sampling this distribution. However, the classical "generate-then-filter" pipeline suffers from severe sample inefficiency. Transformer language models trained on standard chemical databases capture the syntactic rules of SMILES generation but remain entirely blind to the physical realities of the laboratory. 

Often, a generative agent will optimize perfectly for a proxy metric (e.g., ADMET properties or target protein affinity) but exploit the reward function by proposing complex, highly strained polycyclic structures that take months of multi-step organic synthesis to realize, if they can be synthesized at all.

To address this, we propose grounding the generation process in physics-inspired empirical heuristics. Wet-lab chemists do not evaluate reaction success purely through thermodynamic equations; they observe empirical visual cues: the formation of a precipitate, rapid evolution of heat (bubbling), or intermediate colorimetric shifts. We hypothesize that chemical generative models equipped with similar sequential "vision" can produce highly practical molecules.

We present **MATMED**:
1. A multi-agent framework comprising a Generator, Binding Evaluator, Safety Critic, and Reaction Feasibility Critic.
2. A novel application of the **Vision-Temporal Transformer (VTT)** to encode synthetic liquid-phase reaction trajectories into the RL policy loop.
3. A multi-objective Proximal Policy Optimization (PPO) pipeline that orchestrates the simultaneous optimization of structural viability, receptor affinity, and toxicity.

---

## 2. Architecture & Methods

The MATMED architecture trains specialized Transformer bodies on domain-specific datasets (BindingDB, Tox21, USPTO, and ChEMBL) and freezes them during the closed RL loop. A lightweight Policy Agent then acts as the choreographer, aggregating the frozen embedding vectors from the critics and updating only the Generator's policy.

### 2.1 The Generator Agent (Language Model)
A causal Transformer trained on the ChEMBL database to predict the next SMILES token via standard Cross-Entropy Loss to internalize chemical syntax.

### 2.2 The Evaluator Agent (Graph Transformer)
A sparse Graph Transformer where atoms (nodes) exchange multi-head attention messages augmented by bond constraints (edge features). The global scattered mean-pooled representation of the graph is passed through an MLP regression head, pretrained on normalized pIC50 assays from BindingDB.

### 2.3 The Safety Critic (ChemBERTa ADMET)
The Safety Critic employs a pretrained ChemBERTa architecture. The $[CLS]$ token embedding is passed through a multi-task head trained on Tox21 binary classification endpoints.

### 2.4 The Reaction Feasibility Critic (Vision-Temporal Transformer)
The critical innovation of MATMED is the **R-Agent**. Standard approaches predict reaction yield via Graph Neural Networks analyzing the reactant structures. MATMED introduces a dual-modal architecture. 
The input consists of:
1. The structural graph of the proposed molecule.
2. A sequence of 16 simulated "video frames" representing a *physics-inspired reaction trajectory simulation* (e.g., temperature changes, phase shifts).

The Visual-Temporal Transformer tracks the evolution of these visual features across $T$ frames utilizing dense temporal self-attention. If the visual heuristic indicates a volatile or stagnant physical reaction (e.g., extreme high-frequency entropy changes or zero colorimetric evolution), the agent reduces the predicted synthetic feasibility score, directly penalizing the Generator.

### 2.5 Multi-Objective Proximal Policy Optimization (PPO)
During the RL loop, the generator produces a molecule $m$. The critics return continuous scores corresponding to their domains: Binding Score ($E_m$), Safety Score ($S_m$), and Reaction Yield Proxy ($R_m$).
The Reward is calculated as:
$$Reward(m) = \alpha E_m + \beta (\mu_{yield} - \lambda \sigma_{vision}) + \gamma S_m - \delta \cdot \text{KL}(Policy || Prior)$$

---

## 3. Results & Ablation Studies

To quantify the exact necessity of grounding chemical discovery in sequential visual heuristics, we performed ablation studies handling continuous batches of 16 molecules over 50 PPO iteratons.

### 3.1 Ablation Quantitative Analysis

We evaluated the framework against four targeted lesions:
1. **Full MATMED**: Intact VTT Reaction Agent, Evalulator, and Safety Critic.
2. **No Vision Agent**: The VTT was bypassed. The R-Agent predicted feasibility via pure SMILES structure parsing without the visual-physics temporal data.
3. **No Safety Agent**: Removed Tox21 ADMET signals.
4. **No Binding Agent**: Removed BindingDB pIC50 predictions.

All experiments were run with **3 independent random seeds** `{0, 7, 42}` to ensure reproducibility and enable computation of confidence intervals. The convergence curves are averaged with ±1σ shading (see Figure 1). Statistical significance between Full MATMED and each ablation setting was computed using a **Welch's t-test** (two-sided, unequal variance) over the last 15 steady-state reward values.

Table 1 details the statistical convergence of the varied architectures calculated over the final 15 PPO epochs (steady state).

| Model              | Avg Reward (3-seed) | Std Dev | % Invalid SMILES | % Low Feasibility | vs Full (p-value) |
| ------------------ | ------------------- | ------- | ---------------- | ------------------|-------------------|
| **Full MATMED**    | -0.78 ± 0.03        | 0.04    | 12.0%            | 18.2%             | —                 |
| **No Vision**      | -0.82 ± 0.05        | 0.07    | 19.4%            | 27.5%             | 0.023 *           |
| **No Safety**      | -0.96 ± 0.08        | 0.11    | 24.1%            | 29.1%             | 0.004 **          |
| **No Reaction**    | -0.91 ± 0.10        | 0.13    | 28.6%            | N/A               | 0.007 **          |
| **No Binding**     | -0.87 ± 0.04        | 0.06    | 15.2%            | 21.3%             | 0.018 *           |

\*p < 0.05; \*\*p < 0.01 — Welch's t-test, two-sided

**Finding 1: Physics-Inspired Vision Stabilizes the Reinforcement Trajectory.**
The *No Vision Agent* ablation did not cause a catastrophic failure of the policy, proving that textual features alone are functional. However, incorporating temporal visual reaction signals (Full MATMED) consistently improved the final reward ceiling and reduced convergence variance across PPO iterations (Welch's p=0.023, *significant*). This suggests that physics-inspired synthetic trajectory modeling contributes strictly complementary and stabilizing regulatory information beyond structural heuristics alone. Figure 1 shows the 3-seed mean ±1σ convergence band, confirming this smoothing effect is consistent across random initializations.

**Finding 2: Multi-Agent Pareto Frontiers Ensure Broad Viability (Figure 2).**
The inclusion of all critics prevents mode collapse into single-objective exploitation. As seen in the Pareto Analysis tradeoff chart, removing the Safety or Reaction agents explicitly drives the generator to output a significantly higher percentage of invalid or wildly unfeasible structures ($24\%$ and $28\%$, respectively) as the policy destructively manipulates syntax strings trying to max out the remaining singular metrics.

### 4. Conclusion
MATMED bridges the critical gap between computational hallucination and physical realization in *de novo* drug design. By breaking the monolithic generative model into an ensemble of specialized experts, and by explicitly encoding simulated sequential liquid-phase visual physics via the Vision-Temporal Transformer, MATMED successfully restricts chemical exploration to domains that are structurally valid and synthetically stable. 
