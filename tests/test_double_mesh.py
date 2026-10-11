"""Reject a mesh-convergence claim when either tracked trough changes."""
import unittest
import numpy as np

from compare_double_mesh import compare_curves
from fdtd_pipeline.lumerical import validate_power_integration
from fdtd_pipeline.config import load_config, PROJECT_ROOT
from dataclasses import replace


class DoubleMeshTests(unittest.TestCase):
    def test_reduced_monitor_storage_requires_and_passes_independent_power_checks(self):
        config = load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json')
        with self.assertRaisesRegex(ValueError, 'independent'):
            replace(config.simulation, monitor_downsample_xy=2)
        settings = replace(config.simulation, monitor_downsample_xy=2,
                           power_validation_wavelengths_nm=(700, 839, 927, 1700))
        self.assertEqual(settings.spectrum_points, 1000)
        w = np.linspace(700, 1700, 3000)
        r, t = np.full(3000, .8), np.full(3000, 1e-6)
        checks = np.array([1700, 927, 839, 700])
        report, sorted_w, _, _ = validate_power_integration(w, r, t, checks, np.full(4, .8001), np.full(4, 1e-6))
        self.assertLess(report['maximum_R_error'], .002)
        self.assertTrue(np.all(np.diff(sorted_w) > 0))
        with self.assertRaisesRegex(ValueError, 'failed'):
            validate_power_integration(w, r, t, checks, np.full(4, .81), np.full(4, 1e-6))
        with self.assertRaisesRegex(ValueError, 'failed'):
            validate_power_integration(w, r, t, checks, np.full(4, .8), np.full(4, .001))
        self.assertNotIn('monitor_downsample_xy', config.metadata()['simulation'])
        self.assertNotIn('power_validation_wavelengths_nm', config.metadata()['simulation'])

    def test_tracks_both_dips_even_when_prominence_rank_changes(self):
        w = np.linspace(700, 1700, 3000)
        def curve(depths, shifts=(0, 0), widths=(3, 3)):
            return .95-sum(depth*np.exp(-.5*((w-center-shift)/width)**2)
                           for depth, center, shift, width in zip(depths, (838, 924), shifts, widths))
        reference = curve((.90, .91))
        fine = curve((.91, .90), shifts=(.3, -.3))
        report = compare_curves(w, reference, fine)
        self.assertTrue(report['both_dips_stable'])
        self.assertEqual(len(report['matched_dips']), 2)
        for changed in (curve((.90, .80)), curve((.90, .91), shifts=(0, 4)),
                        curve((.90, .91), widths=(3, 4)), curve((.90, 0))):
            self.assertFalse(compare_curves(w, reference, changed)['both_dips_stable'])


if __name__ == '__main__':
    unittest.main()
