"""Rotation geometry, mixed units, equivalent-angle learning and legacy compatibility."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from fdtd_pipeline.config import Geometry, PARAMETER_NAMES, PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import generate_dataset
from metaformer import InverseMetaformer, MetaformerConfig, geometry_mse
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import TrainingConfig, load_training_data, regression_metrics, train
from test_fdtd_pipeline import FakeSimulator


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_self_center_rotation_and_mirror_vertices(self):
        original = Geometry(3200, 900, 1000, 200, 300)
        rotated = replace(original, rotation_deg=20)
        self.assertEqual(rotated.centers(), original.centers())
        for vertices, center in zip(rotated.vertices(), original.centers()):
            np.testing.assert_allclose(vertices.mean(axis=0), center, atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1), 900)
        plus = np.vstack(rotated.vertices())
        minus = np.vstack(replace(original, rotation_deg=-20).vertices())
        reflected = plus * np.array([1, -1])
        np.testing.assert_allclose(reflected[np.lexsort(reflected.T)], minus[np.lexsort(minus.T)], atol=1e-10)
        np.testing.assert_array_equal(rotated.array(), original.array())
        self.assertEqual(rotated.array(include_rotation=True)[-1], 20)

    def test_overlap_and_periodic_cell_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap|touch'):
            Geometry(3200, 900, 1000, 150, 200, 45)
        with self.assertRaisesRegex(ValueError, 'periodic cell'):
            Geometry(2200, 900, 1000, 300, 300, 25)
        with self.assertRaisesRegex(ValueError, 'finite'):
            Geometry(3200, 900, 1000, 200, 300, float('nan'))

    def test_fdtd_sets_native_angle_without_moving_centers(self):
        simulator = object.__new__(FDTDSimulator)
        simulator.settings = load_config(PROJECT_ROOT / 'configs/rotation_pilot.json').simulation
        simulator.fdtd = MagicMock()
        geometry = Geometry(3200, 900, 1000, 200, 300, 20)
        simulator.build(geometry)
        square_calls = [c.kwargs for c in simulator.fdtd.addrect.call_args_list if c.kwargs.get('name', '').startswith('si_')]
        self.assertEqual(len(square_calls), 4)
        np.testing.assert_allclose([(p['x'], p['y']) for p in square_calls], np.asarray(geometry.centers()) * 1e-9)
        rotations = [c.args for c in simulator.fdtd.set.call_args_list if c.args[0] == 'rotation 1']
        self.assertEqual(rotations, [('rotation 1', -20.0)] * 4)

    def test_mirror_equivalent_loss_has_correct_error_and_gradient(self):
        target = torch.full((1, 6), .5, dtype=torch.float64)
        target[0, 5] = (-20 + 45) / 90
        prediction = target.clone()
        prediction[0, 5] = (20 + 45) / 90
        self.assertLess(geometry_mse(prediction, target, 'square_mirror').item(), 1e-28)
        self.assertGreater(geometry_mse(prediction, target).item(), .01)
        prediction[0, 5] = (10 + 45) / 90
        prediction.requires_grad_()
        loss = geometry_mse(prediction, target, 'square_mirror')
        self.assertAlmostEqual(loss.item(), ((10 / 45) ** 2) / 6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLess(prediction.grad[0, 5].item(), 0)  # Increasing this candidate toward +20 reduces error.

    def test_degree_metrics_are_separate_from_nanometers(self):
        target = np.full((1, 6), .5)
        target[0, 5] = (-25 + 45) / 90
        prediction = target.copy()
        prediction[0, 5] = (25 + 45) / 90
        score = regression_metrics(prediction, target, [3000, 850, 800, 150, 150, -45],
                                   [3400, 1050, 1100, 350, 350, 45], angle_symmetry='square_mirror')
        self.assertEqual(set(score['mae_nm']), set(PARAMETER_NAMES))
        self.assertAlmostEqual(score['mae_deg']['rotation_deg'], 0)
        self.assertAlmostEqual(score['raw_signed_mae_deg'], 50)

    def test_mixed_unit_dataset_resume_and_six_output_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(load_config(PROJECT_ROOT / 'configs/rotation_pilot.json'), output_dir=root / 'data')
            FakeSimulator.calls = 0
            FakeSimulator.fail_first = False
            with patch('fdtd_pipeline.pipeline.FDTDSimulator', FakeSimulator):
                generate_dataset(config, limit=2)
                summary = generate_dataset(config)
                self.assertEqual(summary['geometry_shape'], [6, 6])
                generate_dataset(config)
                self.assertEqual(FakeSimulator.calls, 6)
            dataset = SpectrumGeometryDataset(config.output_dir / 'dataset.npz', 'all', True)
            self.assertEqual(dataset.geometry_nm.shape, (6, 5))
            self.assertEqual(dataset.parameter_units, ('nm',) * 5 + ('deg',))
            np.testing.assert_allclose(dataset.denormalize(dataset.labels), dataset.geometry_values, atol=.001)
            # 即使六列归一化值一致，混合单位边界也必须与独立的 nm 边界相符。
            with np.load(config.output_dir / 'dataset.npz', allow_pickle=False) as archive:
                malformed = {key: archive[key].copy() for key in archive.files}
            malformed['lower_nm'][0] += 1
            np.savez(root / 'wrong_bounds.npz', **malformed)
            with self.assertRaisesRegex(ValueError, 'bounds disagree'):
                SpectrumGeometryDataset(root / 'wrong_bounds.npz', 'all', True)
            model_config = MetaformerConfig(geometry_dim=6, angle_symmetry='square_mirror', patch_count=10,
                embedding_dim=8, attention_heads=2, head_dim=4, encoder_layers=1, mlp_dim=16)
            training = TrainingConfig(config.output_dir / 'dataset.npz', root / 'runs',
                                      allow_validation_data=True, epochs=2, batch_size=2, model=model_config)
            with patch('plot_training.plot_training'):
                run_dir, report = train(training)
            self.assertEqual(report['parameter_units'][-1], 'deg')
            self.assertIn('rotation_deg', report['metrics']['test']['mae_deg'])
            header = (run_dir / 'test_predictions.csv').read_text().splitlines()[0]
            self.assertIn('true_deg_rotation_deg', header)
            self.assertNotIn('true_nm_rotation_deg', header)
            checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
            model = InverseMetaformer(model_config)
            model.load_state_dict(checkpoint['model_state'])
            self.assertEqual(model(dataset.spectra[:2]).shape, (2, 6))
            with self.assertRaisesRegex(ValueError, 'parameter count'):
                load_training_data(replace(training, model=MetaformerConfig()))

    def test_legacy_configuration_fingerprint_excludes_rotation_default(self):
        config = load_config(PROJECT_ROOT / 'configs/smoke.json')
        # The preserved historical fingerprint includes the original absolute output path.
        config = replace(config, output_dir=Path(r'D:\projects\LumOpt_test\data\smoke_100'))
        self.assertNotIn('rotation_bounds_deg', config.metadata())
        fingerprint = hashlib.sha256(json.dumps(config.metadata(), sort_keys=True).encode()).hexdigest()
        # 此值来自新增旋转字段之前保存的五参数 manifest，不依赖本地数据文件存在。
        self.assertEqual(fingerprint, '3505271c78c9f60d9fa639998ac3d9401315fa4f8bb5415a0fdace90f70aa5a7')


if __name__ == '__main__':
    unittest.main()
