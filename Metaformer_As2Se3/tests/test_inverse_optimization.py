"""Check train-only preprocessing, equivalent attention, and held-out evaluation."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from fdtd_pipeline.pipeline import atomic_npz
from metaformer import EncoderBlock, InverseMetaformer, MetaformerConfig, MultiHeadSelfAttention
from train_metaformer import TrainingConfig, evaluate, train
from test_training import TINY_MODEL, tiny_data


class InverseOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_fast_attention_matches_map_path_outputs_and_gradients(self):
        torch.manual_seed(3)
        module = MultiHeadSelfAttention(8, 3, 4, 0).double()
        x = torch.randn(2, 7, 8, dtype=torch.double, requires_grad=True)
        fast = module(x)
        reference, maps = module(x, True)
        torch.testing.assert_close(fast, reference, atol=1e-12, rtol=1e-12)
        variables = (x, *module.parameters())
        fast_grad = torch.autograd.grad(fast.square().sum(), variables, retain_graph=True)
        map_grad = torch.autograd.grad(reference.square().sum(), variables)
        for actual, expected in zip(fast_grad, map_grad):
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        self.assertEqual(maps.shape, (2, 3, 7, 7))

    def test_post_norm_is_add_then_norm_for_both_sublayers(self):
        config = replace(TINY_MODEL, norm_order='post', activation='relu', dropout=0)
        block = EncoderBlock(config).eval()
        x = torch.randn(2, 10, 8)
        expected = block.attention_norm(x + block.attention(x))
        expected = block.mlp_norm(expected + block.mlp(expected))
        torch.testing.assert_close(block(x), expected)
        self.assertIsInstance(block.mlp[1], torch.nn.ReLU)

    def test_preprocessing_buffers_reload_and_legacy_has_no_new_state_keys(self):
        plain = InverseMetaformer(TINY_MODEL)
        self.assertFalse(any('spectrum_' in key or 'fitted' in key for key in plain.state_dict()))
        config = replace(TINY_MODEL, input_normalization='train_standard', dropout=0)
        model = InverseMetaformer(config).eval()
        values = torch.stack((torch.zeros(1000), torch.full((1000,), .2)))
        with self.assertRaisesRegex(RuntimeError, 'Fit input'):
            model(values)
        model.fit_input_normalization(values)
        torch.testing.assert_close(model.spectrum_mean, torch.full((1000,), .1))
        torch.testing.assert_close(model.spectrum_scale, torch.full((1000,), .1))
        other = InverseMetaformer(config).eval()
        other.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(model(values), other(values))
        model.fit_input_normalization(torch.zeros(3, 1000))
        torch.testing.assert_close(model.spectrum_scale, torch.full((1000,), .02))

    def test_validation_trial_never_evaluates_test_and_normalizes_train_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = tiny_data()
            data['spectra'][2:] = 100  # Deliberately distinct held-out statistics.
            atomic_npz(root / 'data.npz', **data)
            config = TrainingConfig(root / 'data.npz', root / 'runs', allow_validation_data=True,
                device='cpu', epochs=1, batch_size=2, validation_only=True, optimizer='adam',
                model=replace(TINY_MODEL, input_normalization='train_standard'))
            evaluated_ids = []
            def spy(model, loader, device):
                evaluated_ids.extend(loader.dataset.sample_ids.tolist())
                return evaluate(model, loader, device)
            with patch('train_metaformer.evaluate', side_effect=spy), patch('plot_training.plot_training') as plot:
                run_dir, report = train(config)
            self.assertNotIn(4, evaluated_ids)
            self.assertNotIn(5, evaluated_ids)
            self.assertNotIn('test', report['metrics'])
            self.assertFalse((run_dir / 'test_predictions.csv').exists())
            plot.assert_not_called()
            checkpoint = torch.load(run_dir / 'best.pt', weights_only=True)
            torch.testing.assert_close(checkpoint['model_state']['spectrum_mean'],
                                       torch.from_numpy(data['spectra'][:2].mean(0)))
            self.assertEqual(report['input_normalization_fit_sample_ids'], [0, 1])
            self.assertEqual(json.loads((run_dir / 'status.json').read_text())['state'], 'completed')

    def test_seven_output_parameter_counts_and_unchanged_capacity(self):
        base = MetaformerConfig(geometry_dim=7, angle_symmetry='square_periodic')
        counts = []
        for config in (base, replace(base, activation='relu', norm_order='post',
                                    input_normalization='train_standard'), replace(base, patch_count=200)):
            counts.append(sum(p.numel() for p in InverseMetaformer(config).parameters()))
        self.assertEqual(counts, [3998087, 3998087, 3997447])


if __name__ == '__main__':
    unittest.main()
