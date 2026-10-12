import unittest
import numpy as np
from gold_mesh_research import geometry_source
from gold_mesh_metrics import axis_weights,weighted_statistics,physical_masks,sampling_plan


class FieldMetricTests(unittest.TestCase):
    def test_constant_field_average_and_area_are_independent_of_grid_size(self):
        for count in (9,17):
            coordinates=np.linspace(-40,40,count)
            weight=np.outer(axis_weights(coordinates,-40,40),axis_weights(coordinates,-40,40))
            result=weighted_statistics(np.full(weight.shape,7.),weight)
            self.assertAlmostEqual(result['sampled_area_nm2'],6400)
            self.assertAlmostEqual(result['mean_intensity_enhancement'],7.)
            self.assertEqual(result['p95_intensity_enhancement'],7.)

    def test_quantile_is_area_weighted_not_a_count_of_dense_grid_nodes(self):
        result=weighted_statistics(np.array([[1.,100.]]),np.array([[99.,1.]]))
        self.assertEqual(result['p95_intensity_enhancement'],1.)
        self.assertAlmostEqual(result['mean_intensity_enhancement'],1.99)
        self.assertEqual(result['maximum_intensity_enhancement'],100.)

    def test_physical_masks_include_pillar_centres_and_exclude_sharp_edges(self):
        g=geometry_source(); centre=np.asarray(g.centers()[0])
        masks=physical_masks([centre[0],0.],[centre[1],0.],g)
        self.assertTrue(masks['pillar_bulk'][0,0])
        self.assertFalse(masks['air_away_from_edges'][0,0])
        self.assertTrue(masks['air_away_from_edges'][1,1])
        edge=np.array([g.l/2,0.]) @ g.rotation_matrix(g.angles()[0]).T+centre
        masks=physical_masks([edge[0]],[edge[1]],g)
        self.assertFalse(masks['pillar_bulk'][0,0])
        self.assertFalse(masks['air_away_from_edges'][0,0])

    def test_nonfinite_negative_and_empty_regions_are_rejected(self):
        for bad in (np.array([[np.nan]]),np.array([[-1.]])):
            with self.assertRaises(ValueError):weighted_statistics(bad,np.ones((1,1)))
        with self.assertRaises(ValueError):weighted_statistics(np.ones((1,1)),np.ones((1,1)),np.zeros((1,1),bool))

    def test_selected_frequencies_are_real_common_points_with_dense_target_windows(self):
        w=np.linspace(700,1700,3000)
        gold_r=.95-.9*np.exp(-.5*((w-839)/3)**2)-.9*np.exp(-.5*((w-927)/3)**2)
        bare_r=.1+.7*np.exp(-.5*((w-837)/3)**2)+.7*np.exp(-.5*((w-920)/3)**2)
        gold={'reflectance':gold_r,'absorptance':1-gold_r,'transmittance':np.zeros_like(w)}
        bare={'reflectance':bare_r,'absorptance':np.zeros_like(w),'transmittance':1-bare_r}
        plan=sampling_plan(w,gold,bare)
        selected=np.array(plan['spectral_wavelengths_nm'])
        self.assertTrue(np.isin(selected,w).all())
        self.assertLessEqual(np.max(np.diff(selected)),1.001)
        for center in plan['gold_target_wavelengths_nm']:
            self.assertTrue(np.isin(w[abs(w-center)<=24],selected).all())
        self.assertTrue(np.isin(plan['electric_wavelengths_nm'],selected).all())
        self.assertTrue(np.isin(plan['power_check_wavelengths_nm'],selected).all())


if __name__=='__main__':unittest.main()
