"""Validity-first experiment. This is NOT a validated multi-objective drug screen."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase

from generator_agent import GeneratorAgent
from generator_ppo import GeneratorPPO, rollout
from progress_connector import publish_report
from research_data import load_corpus, split_corpus, write_manifest
from utils import SMILESTokenizer, is_valid_smiles, set_seed

SMOKE_SMILES = ['CCO', 'CCN', 'CCC', 'CC(=O)O', 'c1ccccc1', 'Oc1ccccc1',
                'C1CCCCC1', 'OC1CCCCC1', 'c1ccncc1', 'Cc1ccncc1',
                'C1CCOC1', 'CC1CCOC1', 'C1CCC1', 'CC1CCC1',
                'c1ccc2ccccc2c1', 'C1CC1', 'CC1CC1', 'CC(C)O']


def encode(model, smiles):
    return torch.tensor([model.tokenizer.encode(s, max_len=model.max_len) for s in smiles],
                        device=next(model.parameters()).device, dtype=torch.long)


@torch.no_grad()
def evaluate(model, count, batch_size, training_smiles):
    generated = []
    for start in range(0, count, batch_size):
        ids = rollout(model, min(batch_size, count - start))
        generated.extend(model.tokenizer.decode(row.tolist()) for row in ids)
    valid = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in generated if is_valid_smiles(s)]
    unique = set(valid)
    rewards = np.array([1. if is_valid_smiles(s) else -2. for s in generated])
    return dict(attempts=count, valid_count=len(valid),
                raw_validity_pct=100 * len(valid) / count,
                mean_reward=float(rewards.mean()), reward_variance=float(rewards.var()),
                uniqueness_pct=100 * len(unique) / max(1, len(valid)),
                novelty_pct=100 * len(unique - set(training_smiles)) / max(1, len(unique))), generated


@torch.no_grad()
def validation_nll(model, smiles, batch_size):
    model.eval()
    total, tokens = 0., 0
    for start in range(0, len(smiles), batch_size):
        ids = encode(model, smiles[start:start + batch_size])
        n = int(ids[:, 1:].ne(model.tokenizer.pad_idx).sum())
        total += float(model.compute_loss(ids[:, :-1], ids[:, 1:])) * n
        tokens += n
    return total / max(1, tokens)


def write_outputs(directory, report):
    report['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    data = json.dumps(report, indent=2, allow_nan=False)
    temp = directory / 'report.tmp'
    temp.write_text(data)
    temp.replace(directory / 'report.json')
    history = report['history']
    if history:
        fields = list(dict.fromkeys(k for row in history for k in row))
        with (directory / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
    try:
        outcome = publish_report(report)
        (directory / 'connector_status.json').write_text(json.dumps(outcome))
    except Exception as exc:
        # Preserve local results if the remote service is unavailable.
        (directory / 'connector_status.json').write_text(json.dumps(
            {'uploaded': False, 'error': type(exc).__name__}))
        print(f'Progress upload failed ({type(exc).__name__}); results retained locally', flush=True)


def plot_results(directory, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['history']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, field, label in zip(axes, ['raw_validity_pct', 'prior_kl', 'mean_reward'],
                              ['Raw validity (%)', 'KL to pretrained prior', 'Validity reward']):
        points = [(i, r[field]) for i, r in enumerate(rows) if field in r]
        if points:
            ax.plot(*zip(*points), marker='o')
        ax.set(xlabel='Recorded checkpoint', ylabel=label)
        ax.grid(alpha=.2)
    axes[0].set_ylim(0, 100)
    fig.suptitle('SMOKE TEST - not scientific evidence' if report['mode'] == 'smoke'
                 else 'Validity-first experiment - no drug efficacy claim')
    fig.tight_layout()
    fig.savefig(directory / 'curves.png', dpi=150)
    plt.close(fig)


def run(args):
    if args.mode == 'smoke':
        args.epochs, args.iterations, args.batch_size, args.eval_samples = 2, 1, 4, 16
        args.max_len = 32
    if min(args.epochs, args.batch_size, args.eval_samples) < 1 or args.iterations < 0:
        raise ValueError('epochs, batch-size and eval-samples must be positive')
    if args.max_len < 4 or not 0 <= args.min_validity <= 100:
        raise ValueError('Invalid max-len or validity gate')
    set_seed(args.seed)
    if args.device == 'cpu':
        torch.set_num_threads(2)
    run_id = f'matmed-{args.seed}-{uuid.uuid4().hex[:12]}'
    directory = Path(args.output) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    report = dict(schema_version=1, run_id=run_id, mode=args.mode, status='running',
                  stage='data', seed=args.seed, history=[], metrics={},
                  objective='structural_validity_only',
                  limitations=['Not experimentally validated',
                               'Binding, safety, reaction and vision are not evaluated in this stage'],
                  provenance={'commit': commit, 'python': platform.python_version(),
                              'source_sha256': source_hash.hexdigest(),
                              'torch': torch.__version__, 'rdkit': rdBase.rdkitVersion},
                  config=vars(args))
    write_outputs(directory, report)
    try:
        if args.mode == 'smoke':
            corpus, tokenizer = SMOKE_SMILES, SMILESTokenizer()
            meta = {'source': 'synthetic/software-test-fixture', 'retained': len(corpus)}
        else:
            if not args.data or not args.source:
                raise ValueError('Research mode requires --data real.csv and --source dataset/version')
            corpus, tokenizer, meta = load_corpus(args.data, args.source, args.max_len)
        split = split_corpus(corpus)
        write_manifest(directory / 'data_manifest.json', meta, split)
        report['dataset'] = {**meta, 'split_counts': {k: len(v) for k, v in split.items()}}
        config = dict(d_model=32 if args.mode == 'smoke' else 256,
                      nhead=4, num_layers=2 if args.mode == 'smoke' else 4,
                      d_ff=64 if args.mode == 'smoke' else 512, max_len=args.max_len)
        model = GeneratorAgent(tokenizer=tokenizer, **config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        best = float('inf')
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = np.random.permutation(len(split['train']))
            for start in range(0, len(order), args.batch_size):
                ids = encode(model, [split['train'][i] for i in order[start:start + args.batch_size]])
                loss = model.compute_loss(ids[:, :-1], ids[:, 1:])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite supervised loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            nll = validation_nll(model, split['validation'], args.batch_size)
            if nll < best:
                best = nll
                torch.save({'state_dict': model.state_dict(), 'config': config,
                            'vocab': tokenizer.vocab, 'data': meta}, directory / 'prior.pt')
            report['stage'] = 'pretraining'
            report['history'].append(dict(stage='pretraining', iteration=epoch, validation_nll=nll))
            write_outputs(directory, report)
            print(f'Pretraining {epoch}/{args.epochs}: validation NLL={nll:.4f}', flush=True)
        checkpoint = torch.load(directory / 'prior.pt', map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        baseline, _ = evaluate(model, args.eval_samples, args.batch_size, split['train'])
        report['baseline'] = baseline
        report['metrics'] = baseline
        report['history'].append(dict(stage='prior', iteration=0, **baseline))
        if args.mode != 'smoke' and baseline['raw_validity_pct'] < args.min_validity:
            report.update(status='needs_pretraining', stage='validity_gate')
            report['limitations'].append('Raw prior validity below PPO entry gate; PPO was not run')
        else:
            trainer = GeneratorPPO(model)
            for iteration in range(1, args.iterations + 1):
                ids = rollout(model, args.batch_size)
                smiles = [tokenizer.decode(row.tolist()) for row in ids]
                valid = [is_valid_smiles(s) for s in smiles]
                rewards = torch.tensor([1.0 if v else -2.0 for v in valid], device=args.device)
                replay = encode(model, [split['train'][i] for i in
                    np.random.randint(len(split['train']), size=args.batch_size)])
                update = trainer.update(ids, rewards, replay)
                metrics = dict(raw_validity_pct=100 * sum(valid) / len(valid),
                               attempts=len(valid), valid_count=sum(valid),
                               mean_reward=float(rewards.mean()),
                               reward_variance=float(rewards.var(unbiased=False)), **update)
                report['history'].append(dict(stage='ppo', iteration=iteration, **metrics))
                report.update(stage='ppo', metrics=metrics)
                write_outputs(directory, report)
                print(f'PPO {iteration}: validity={metrics["raw_validity_pct"]:.1f}% '
                      f'prior KL={metrics["prior_kl"]:.5f}', flush=True)
            final, samples = evaluate(model, args.eval_samples, args.batch_size, split['train'])
            report['metrics'] = {**report['metrics'], **final}
            report['history'].append(dict(stage='final', iteration=args.iterations, **final))
            report['test_nll'] = validation_nll(model, split['test'], args.batch_size)
            with (directory / 'generated.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles', 'valid'])
                writer.writerows((s, is_valid_smiles(s)) for s in samples)
            torch.save({'state_dict': model.state_dict(), 'config': config,
                        'vocab': tokenizer.vocab, 'data': meta}, directory / 'generator_final.pt')
            report.update(status='completed', stage='final')
        plot_results(directory, report)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_outputs(directory, report)
        raise
    write_outputs(directory, report)
    print(f'REPORT_PATH={directory / "report.json"}', flush=True)
    return directory, report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['research', 'smoke'], default='research')
    parser.add_argument('--data')
    parser.add_argument('--source', default='')
    parser.add_argument('--output', default='runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--eval-samples', type=int, default=1024)
    parser.add_argument('--min-validity', type=float, default=40.)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
