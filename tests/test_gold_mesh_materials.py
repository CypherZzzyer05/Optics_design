"""Material passivity is not a substitute for fitting its absorption."""
import unittest
import numpy as np
from audit_gold_mesh_materials import fit_metrics


class MaterialDiagnosticsTests(unittest.TestCase):
    def test_matching_k_does_not_hide_lossless_gold_fit(self):
        measured=np.array([.2+5j]);fitted=np.array([1e-8+5j])
        result=fit_metrics(measured,fitted)
        self.assertEqual(result['max_k_error'],0.)
        self.assertAlmostEqual(result['max_n_error'],.19999999)
        self.assertAlmostEqual(result['maximum_imaginary_permittivity_error'],1.9999999)

    def test_nonfinite_gain_and_mismatched_axes_are_rejected(self):
        for fitted in (np.array([np.nan+1j]),np.array([1-1j]),np.array([1+1j,1+1j])):
            with self.assertRaises(ValueError):fit_metrics(np.array([1+1j]),fitted)


if __name__=='__main__':unittest.main()
