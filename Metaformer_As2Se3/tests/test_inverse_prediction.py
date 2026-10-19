"""Standalone inverse checkpoint and physical-unit input/output contract."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from metaformer import InverseMetaformer, MetaformerConfig
from predict_inverse import load_inverse, predict


class InversePredictionTests(unittest.TestCase):
    def test_raw_spectrum_checkpoint_round_trip_and_physical_units(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            config = MetaformerConfig(patch_count=10, embedding_dim=8, attention_heads=2,
                head_dim=4, encoder_layers=1, mlp_dim=16, geometry_dim=7,
                angle_symmetry='square_periodic', input_normalization='train_standard', dropout=0)
            model = InverseMetaformer(config).eval()
            model.fit_input_normalization(torch.stack((torch.zeros(1000), torch.ones(1000))))
            wavelength = np.linspace(700, 1700, 1000)
            lower = np.array([620, 160, 240, 90, 90, -45, -45])
            upper = np.array([690, 210, 380, 150, 150, 45, 45])
            names = ['p', 'l', 'h', 'alpha_x', 'alpha_y', 'theta_odd_deg', 'theta_even_deg']
            metadata = {'config': {'model': asdict(config), 'seed': 42},
                        'wavelength_nm': wavelength.tolist(), 'lower_values': lower.tolist(),
                        'upper_values': upper.tolist(), 'parameter_names': names,
                        'parameter_units': ['nm'] * 5 + ['deg'] * 2, 'quality_level': 'pipeline_validation'}
            spectrum = np.linspace(.1, .9, 1000, dtype=np.float32)
            with TemporaryDirectory() as directory:
                path = Path(directory) / 'best.pt'
                torch.save({'model_state': model.state_dict(), 'metadata': metadata}, path)
                reloaded, saved = load_inverse(path)
                actual = predict(reloaded, saved, spectrum, wavelength)
            with torch.inference_mode():
                expected = model(torch.from_numpy(spectrum[None])).numpy()[0]
            np.testing.assert_allclose(actual['normalized'], expected)
            np.testing.assert_allclose([actual['parameters'][p]['value'] for p in names],
                                       lower + expected * (upper - lower))
            with self.assertRaisesRegex(ValueError, 'grid differs'):
                predict(model, metadata, spectrum, wavelength + 1)
            with self.assertRaisesRegex(ValueError, '1000 wavelength'):
                predict(model, metadata, spectrum[:-1], wavelength[:-1])
            spectrum[1] = np.nan
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                predict(model, metadata, spectrum, wavelength)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
