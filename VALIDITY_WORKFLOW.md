# MATMED validity-first implementation

## What is available

- `matmed_validity_colab.ipynb`: fail-fast setup, unit tests, one-step smoke run,
  optional real-corpus pretraining and three-seed/50-iteration experiment, results export.
- `artifacts/matmed_colab_bundle.zip`: matching source upload for Colab. No GitHub
  push is needed to use this bundle. Rebuild with `python scripts/build_colab.py`.
- `validity_pipeline.py`: corpus audit, scaffold-disjoint split, supervised training,
  best-validation checkpoint, raw-validity gate, token-level generator PPO, plots.
- `prototype/`: a new responsive dashboard plus authenticated Cloudflare Worker API.
- `progress_connector.py`: optional report uploads from Colab; no public Colab tunnel.

Cloudflare does not execute PyTorch. The frontend observes jobs started in Colab;
it does not remotely start training or serve molecular inference. Dataset and
checkpoint storage stays in Colab/Drive. Worker KV stores small reports only.

## Run locally

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-colab.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python validity_pipeline.py --mode smoke
node --test prototype/tests/worker.test.mjs
```

Real research requires a CSV with a `smiles` column and source/version metadata:

```sh
.venv/bin/python validity_pipeline.py --data /path/to/curated.csv \
  --source 'Dataset name / release / provenance identifier' \
  --epochs 20 --iterations 50 --seed 42 --device cuda
```

Research mode never downloads replacement data or silently substitutes fixtures.
It canonicalizes and deduplicates, excludes sequences that cannot fit rather than
truncating them, extends the vocabulary to observed tokens, and saves the vocabulary
and complete model configuration with checkpoints. Splits are whole scaffold groups;
the empty acyclic scaffold is one shared group. Too few groups fails explicitly.
Actual split sizes may differ from 80/10/10. The source string is user-provided
provenance, not automatic dataset-quality certification.

## PPO and measurement contract

The pretrained generator is a frozen reference. Only the upper half of generator
layers and LM head are updated by token-level clipped PPO. Rollout log probabilities
and centered advantages remain fixed across epochs. Dropout is disabled for rollout
and replay. Sampling and replay share the same special-token masks; there is no
top-k support mismatch. Explicit KL anchors to the frozen prior, and old-policy KL
is monitored separately for early stopping. A supervised replay term preserves
grammar. Critic architectures and the coordinator remain available unchanged.

Defaults: learning rate 1e-5, clip .1, four PPO epochs, entropy coefficient .1,
prior KL coefficient .1 with adaptive target .02, batch size 64. Research PPO starts
only if the selected pretrained checkpoint achieves at least 40% raw sample validity.
This is an engineering gate, not an experimental validation criterion. Sampling
uncertainty still applies; default evaluation size is 1,024 molecules.

This first stage uses **only** a validity reward (+1 valid, -2 invalid), not
binding/safety/reaction/vision reward. Existing critics are retained, not falsely
represented as calibrated on real data. Later multi-objective integration needs
real, validated critic checkpoints and a shared evaluation protocol.

All attempts count, including truncated or invalid strings; no retries select a
valid molecule before measurement. EOS/PAD masking excludes padded positions from
PPO. Validity is RDKit parsing, not synthesizability or therapeutic usefulness.
Uniqueness is among valid outputs; novelty is canonical-string absence from the
training partition, not patent novelty. Final reward metrics are computed on the
fresh evaluation sample; final KL/entropy remain diagnostics on the last PPO rollout
prefixes and are not estimates over the entire molecular distribution.

Each UUID run has `report.json`, `metrics.csv`, `curves.png`, `data_manifest.json`,
and model checkpoints. Failed gates produce `needs_pretraining`, not a false success.
Check `connector_status.json` separately: training can complete while upload fails.
Never merge repeated runs by episode number. The source SHA-256 identifies Python
source even when Colab is running an uncommitted upload without Git metadata.

## Legacy correctness fixes

- Pretraining no longer freezes randomly initialized lower generator layers.
- Legacy sampling clamps top-k and masks specials before truncating support;
  incorrect bans on ring-index reuse and consecutive ring digits were removed.
- Legacy episode validity denominator counts actual selected molecules. Its rejection
  selection remains different from raw validity; use the new pipeline for raw metrics.
- Binding targets already expressed as pIC50 are no longer log-transformed again;
  validation/test use training normalization bounds.
- Running variance now includes between-batch mean shifts, including singleton calls.
- Binding and ADMET-minus-toxicity retain their higher-is-better direction.
- Held-out calibration caches detached logits, uses positive per-endpoint temperatures,
  masks missing binary labels, and rejects empty validation input.

Temperature checkpoints now contain `log_temperature`, not the old unconstrained
`temperature`. Refit old scalers; do not load them silently. These mathematical fixes
do not validate the legacy synthetic critics or connect their scalers to all legacy
inference paths. The old Phase 4 experiments remain unsuitable for efficacy claims.

## Still required before drug-discovery claims

1. Curated target-specific binding and endpoint-specific toxicity datasets.
2. Real reaction outcomes and matched visual observations; no shuffled-SMILES or
   image-noise "failure" labels without experimental justification.
3. Trained heads and fitted scalers consistently loaded at inference, independently
   tested with untouched scaffold/temporal holdouts.
4. Common external evaluation across agents/ablations and sufficient independent seeds.
5. Prospective compound synthesis, identity/purity verification and biological assays.

Local smoke results prove software execution only. No real-data or GPU experiment
is inferred from the smoke test, and no Cloudflare deployment is performed automatically.
