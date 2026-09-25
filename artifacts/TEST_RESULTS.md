# Local verification results

Date: 2026-09-25. These are software tests, not a drug-discovery experiment.

## Checks completed

| Check | Result |
|---|---|
| Python unit/regression tests | 13 passed |
| Worker API tests | 4 passed |
| Python source compilation | Passed |
| Colab notebook schema and code-cell syntax | Passed |
| Colab source archive integrity | Passed |
| Cloudflare Wrangler deployment dry run | Passed; no deployment performed |
| Local Worker + KV integration | Three authenticated upload/readback roundtrips passed |
| Dashboard browser inspection | Runs visible; desktop and 390px mobile layouts checked |

See `python_tests.txt`, `api_tests.txt`, `cloudflare_dry_run.txt`, and
`smoke_results.json` for recorded output. The integration was performed against
Wrangler's local Worker/KV emulator, not a deployed Cloudflare account.

## Three-seed smoke measurements

Each run used the 18-molecule software fixture, a small randomly initialized
32-dimensional two-layer generator, two supervised epochs, and one PPO iteration
with up to four optimization epochs. Final evaluation sampled 16 molecules per seed.

| Seed | Final raw validity | Mean validity reward | Prior KL on PPO prefixes | PPO-prefix entropy |
|---|---:|---:|---:|---:|
| 42 | 0/16 (0%) | -2.000 | 3.706e-7 | 3.3707 |
| 123 | 0/16 (0%) | -2.000 | 2.945e-7 | 3.3698 |
| 999 | 0/16 (0%) | -2.000 | 2.311e-7 | 3.3948 |

The generator gradients were finite. All-invalid smoke batches have identical
rewards, so their centered PPO advantages are zero; these smoke updates get their
nonzero gradient from supervised replay and entropy, not a validity preference.
A separate passing unit test disables supervised replay and entropy, supplies
nonconstant rewards, and verifies that PPO changes generator weights while frozen
layers and the pretrained reference remain unchanged.

**Interpretation:** startup, optimization, persistence, transport and visualization
work. No improvement in generation quality has been demonstrated by this fixture.
The research-mode gate correctly blocks PPO when measured prior validity is below
40%; smoke mode deliberately bypasses it to exercise the update code.

## Not tested or established

- A real corpus, a Colab GPU run, or three seeds of 50 research iterations.
- Calibrated real-data binding, safety, reaction or vision performance.
- Statistical superiority of any ablation or experimental drug efficacy.
- A live Cloudflare deployment or production-scale security/load behavior.

Use `matmed_validity_colab.ipynb` with a curated corpus for the next experiment.
The old `phase4_colab.ipynb` remains a legacy synthetic-data workflow.
