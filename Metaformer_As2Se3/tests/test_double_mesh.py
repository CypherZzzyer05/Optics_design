"""Reject a mesh-convergence claim when either tracked trough changes."""
import unittest
import numpy as np

from compare_double_mesh import compare_curves


class DoubleMeshTests(unittest.TestCase):
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
