import unittest

from train_metaformerpro_refine import select_validation


class RefinementSelectionTests(unittest.TestCase):
    def test_keep_baseline_for_tiny_gain_and_reject_worse_dy(self):
        records = []
        for name, mse, dy in [('baseline', .02, 13.), ('tiny_gain', .0199, 13.),
                              ('worse_dy', .018, 14.)]:
            for seed in (42, 43, 44):
                records.append({'variant': name, 'seed': seed, 'parameters': 100,
                    'validation': {'mirror_equivalent': {'mse': mse, 'length_mse': .02,
                        'mae_nm': [4., 1., 4., 6., dy], 'mae_deg': [12., 13.]}}})
        selection = select_validation(records, 'fixed rule')
        self.assertEqual(selection['selected'], 'baseline')
        bad = next(r for r in selection['ranking'] if r['variant'] == 'worse_dy')
        self.assertFalse(bad['eligible'])
        for seed in (42, 43, 44):
            records.append({'variant': 'useful', 'seed': seed, 'parameters': 120,
                'validation': {'mirror_equivalent': {'mse': .019, 'length_mse': .02,
                    'mae_nm': [4., 1., 4., 6., 12.], 'mae_deg': [12., 13.]}}})
        selection = select_validation(records, 'fixed rule')
        self.assertEqual(selection['selected'], 'useful')
        self.assertEqual(selection['representative_seed'], 43)
        self.assertFalse(selection['test_evaluated_at_selection'])


if __name__ == '__main__':
    unittest.main()
