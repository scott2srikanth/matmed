# MATMED: Multi-Agent Transformer for Molecular Evolution & Design

![MATMED Architecture](https://img.shields.io/badge/Status-Research%20Prototype-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

MATMED is a **multi-agent reinforcement learning platform** designed for de novo drug discovery. Unlike traditional single-model generative approaches, MATMED utilizes a distributed architecture where specialized Transformer agents independently evaluate specific facets of a molecule (e.g. Binding Affinity, Safety/Toxicity, Synthesis Feasibility), while a central Policy Agent choreographs the generation process.

### 🧠 Core Innovation: The Vision-Reaction Agent
MATMED introduces the **Vision-Temporal Transformer (VTT)** into the molecular design loop. Real-world chemical synthesis often yields visual cues (turbidity, phase changes, color gradients) that deterministic reaction predictors miss. MATMED's R-Agent is capable of processing synthetic 16-frame "reaction videos" (simulating laboratory observations) to penalize or reward the Generator based on empirical synthesis complexity, fundamentally grounding the RL policy in physical reality.

---

## 🏗️ Architecture

The system consists of 5 collaborating agents:

1. **G-Agent (Generator)**: An autoregressive Transformer trained on ChEMBL that proposes novel SMILES strings token-by-token.
2. **E-Agent (Evaluator)**: A Graph Transformer (dense attention over atoms and bonds) trained on BindingDB to predict target binding affinity (pIC50).
3. **S-Agent (Safety Critic)**: A ChemBERTa-powered encoder (with character-level fallback) trained on the Tox21 dataset to predict off-target toxicity and ADMET profiles.
4. **R-Agent (Reaction Critic)**: A dual-modal Vision-Temporal Transformer that evaluates synthesis feasibility using both the SMILES text and simulated reaction video frames. Evaluates on USPTO data.
5. **P-Agent (Policy)**: The central RL coordinator (using Proximal Policy Optimization) that processes embeddings from all critic agents to compute a multi-objective reward function.

---

## 🚀 Getting Started

The exact MATMED pipeline is separated into phases. Run the included Google Colab notebook (`phase4_colab.ipynb`) for the simplest cloud-GPU execution.

### Local Setup
```bash
git clone https://github.com/scott2srikanth/matmed.git
cd matmed
pip install -r requirements.txt
```

### Pretraining the Agents (Phase 4)
Each critic agent must be pretrained on its domain-specific structural / synthetic data before the policy loop can optimize against them.

```bash
python pretrain_g_agent.py   # Autoregressive Language Model (ChEMBL)
python pretrain_e_agent.py   # Graph Transformer (BindingDB)
python pretrain_s_agent.py   # ADMET Encoder (Tox21)
python pretrain_r_agent.py   # Multi-Modal Vision Feasibility (USPTO)
```

### Running the Reinforcement Learning Ablations
The `phase4_main.py` script orchestrates PPO training and automatically runs component ablation studies (disabling one agent at a time) to scientifically validate the architecture's contribution to the convergence curve.

```bash
python phase4_main.py
python phase4_plot.py
```

This generates `phase4_ablation_plot.png`, tracking the Multi-Objective Reward against PPO iterations for the full MATMED ensemble versus targeted agent lesions (e.g., No Vision, No Safety).
