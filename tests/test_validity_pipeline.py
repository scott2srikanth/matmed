import copy
import argparse
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from generator_agent import GeneratorAgent
from generator_ppo import GeneratorPPO, rollout, token_distribution
from research_data import load_corpus, split_corpus
from progress_connector import publish_report
from validity_pipeline import encode, evaluate, SMOKE_SMILES
from utils import SMILESTokenizer, set_seed


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        set_seed(42)
        self.model = GeneratorAgent(d_model=32, nhead=4, num_layers=2,
                                    d_ff=64, max_len=24, dropout=.1)

    def test_ppo_changes_generator_and_preserves_frozen_prior(self):
        trainer = GeneratorPPO(self.model, entropy_coef=0, sft_coef=0)
        ids = encode(self.model, ['CCO', 'CCC', 'CCN', 'CO'])
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        reference = copy.deepcopy(trainer.reference.state_dict())
        result = trainer.update(ids, torch.tensor([1., -2., 1., -2.]), ids)
        self.assertGreater(result['grad_norm'], 0)
        self.assertTrue(torch.isfinite(torch.tensor(list(result.values()))).all())
        self.assertFalse(torch.equal(before['output_proj.weight'], self.model.output_proj.weight))
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, before[name]), name)
        for name, p in trainer.reference.state_dict().items():
            self.assertTrue(torch.equal(p, reference[name]), name)

    def test_rollout_replay_probabilities_agree(self):
        self.model.eval()
        ids = rollout(self.model, 3)
        full = token_distribution(self.model, ids[:, :-1]).probs
        for t in range(ids.size(1) - 1):
            prefix = token_distribution(self.model, ids[:, :t + 1]).probs[:, -1]
            torch.testing.assert_close(prefix, full[:, t], atol=2e-6, rtol=1e-5)
        self.assertEqual(float(full[..., self.model.tokenizer.unk_idx].detach().sum()), 0)

    def test_singleton_and_all_invalid_batches_are_finite(self):
        for smiles in [['CCO'], ['CCO', 'CCC']]:
            trainer = GeneratorPPO(self.model)
            ids = encode(self.model, smiles)
            result = trainer.update(ids, torch.full((len(smiles),), -2.), ids)
            self.assertTrue(torch.isfinite(torch.tensor(list(result.values()))).all())

    def test_real_corpus_dedup_vocab_and_scaffold_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.csv'
            path.write_text('smiles\nCCO\nOCC\nc1ccccc1\nC1CCCCC1\n[nH]1cccc1\ninvalid\n')
            smiles, tokenizer, metadata = load_corpus(path, 'test-fixture', 64)
            self.assertEqual(metadata['invalid_rows'], 1)
            self.assertEqual(len(smiles), 4)
            self.assertIn('H', tokenizer.vocab)
            self.assertNotIn(tokenizer.unk_idx, tokenizer.encode('[nH]1cccc1'))
            split = split_corpus(smiles)
            self.assertTrue(all(split.values()))
            from rdkit.Chem.Scaffolds import MurckoScaffold
            groups = [{MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in part}
                      for part in split.values()]
            self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])

    def test_no_random_split_fallback(self):
        with self.assertRaises(ValueError):
            split_corpus(['CCO', 'CCC', 'CCN'])

    def test_topk_larger_than_vocabulary_and_special_tokens(self):
        for k in [1, 500]:
            samples, _ = self.model.generate(batch_size=2, top_k=k)
            self.assertEqual(len(samples), 2)
            self.assertTrue(all('<' not in s for s in samples))

    def test_raw_validity_denominator(self):
        stats, samples = evaluate(self.model, 7, 3, SMOKE_SMILES)
        self.assertEqual(stats['attempts'], 7)
        self.assertEqual(len(samples), 7)
        self.assertEqual(stats['raw_validity_pct'], 100 * stats['valid_count'] / 7)

    def test_connector_rejects_insecure_remote(self):
        with self.assertRaises(ValueError):
            publish_report({'run_id': 'matmed-test'}, 'http://example.com', 'secret')

    def test_calibration_freezes_critic_and_reduces_heldout_nll(self):
        from calibration_utils import calibrate_model
        from torch.utils.data import TensorDataset, DataLoader
        model = torch.nn.Linear(1, 2)
        model.weight.data.fill_(8.)
        model.bias.data.zero_()
        x = torch.tensor([[-1.], [1.], [-1.], [1.]])
        y = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
        before = copy.deepcopy(model.state_dict())
        raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        scaler = calibrate_model(model, DataLoader(TensorDataset(x, y), batch_size=3), 'cpu')
        after_loss = torch.nn.functional.binary_cross_entropy_with_logits(scaler(model(x)), y)
        self.assertLessEqual(float(after_loss.detach()), float(raw_loss.detach()))
        self.assertTrue((scaler.temperature > 0).all())
        self.assertTrue(model.training)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[name]))
            self.assertIsNone(p.grad)
        with self.assertRaises(ValueError):
            calibrate_model(model, [], 'cpu')

    def test_normalizer_retains_between_batch_variance(self):
        from reward_normalizer import RunningRewardNormalizer
        norm = RunningRewardNormalizer(momentum=.5)
        norm.normalize('x', torch.tensor([0., 0.]))
        norm.normalize('x', torch.tensor([2., 2.]))
        self.assertAlmostEqual(float(norm.mean['x']), 1.)
        self.assertAlmostEqual(float(norm.var['x']), 1.)
        for x in [0., 1., 0., 1.]:
            value = norm.normalize('singleton', torch.tensor([x]))
            self.assertTrue(torch.isfinite(value).all())
            self.assertLess(abs(float(value[0])), 5.)

    def test_binding_and_safety_reward_directions(self):
        from critic_calibration import CriticCalibrator
        calibrator = CriticCalibrator()
        for transform in [calibrator.calibrate_binding, calibrator.calibrate_safety]:
            values = transform(torch.tensor([.1, .9]))
            self.assertGreater(float(values[1]), float(values[0]))

    def test_pic50_is_not_double_transformed(self):
        from pretrain_e_agent import BindingDataset
        train = BindingDataset([('CCO', 5.), ('CCC', 9.)])
        validation = BindingDataset([('CCN', 7.)], (train.y_min, train.y_max))
        self.assertEqual(train.y.tolist(), [5., 9.])
        self.assertAlmostEqual(validation[0][1], .5)

    def test_research_gate_blocks_ppo_below_raw_validity_threshold(self):
        from validity_pipeline import run
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'fixture.csv'
            data.write_text('smiles\n' + '\n'.join(SMOKE_SMILES) + '\n')
            args = argparse.Namespace(mode='research', data=str(data), source='unit-test-only',
                output=tmp, seed=42, epochs=1, iterations=1, batch_size=4,
                eval_samples=4, max_len=32, min_validity=40., device='cpu')
            baseline = dict(raw_validity_pct=0., attempts=4, valid_count=0)
            with patch.dict(os.environ, {'MATMED_API_URL': ''}), \
                 patch('validity_pipeline.evaluate', return_value=(baseline, [])), \
                 patch('validity_pipeline.GeneratorPPO') as ppo, \
                 patch('validity_pipeline.plot_results'):
                directory, report = run(args)
            ppo.assert_not_called()
            self.assertEqual(report['status'], 'needs_pretraining')
            self.assertTrue((directory / 'prior.pt').is_file())
            self.assertNotIn('test_nll', report)


if __name__ == '__main__':
    unittest.main()
