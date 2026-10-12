"""Keep measured-frequency provenance and enforce the accepted tolerances."""
import unittest
from unittest.mock import patch
import numpy as np
from analyze_gold_mesh_results import compare_cases,sampled_field_extrema,fields_near_spectral_features


def stats(mean):
    return {'mean_intensity_enhancement':mean,'p95_intensity_enhancement':mean*2,
            'maximum_intensity_enhancement':mean*3,'sampled_area_nm2':1,'sampled_nodes':1}


def profiles(mean):
    return {name:[{'wavelength_nm':800.,'regions':{r:stats(mean) for r in
             (('all',) if '_top_' in name else ('pillar_bulk','air_away_from_edges'))}}]
            for name in ('nf_top_E','nf_mid_E','nf_top_H','nf_mid_H')}


def case(mean):
    return {'state':{'recipe':{'case_id':'fixture'}},'data':{'wavelength_nm':np.array([800.,801.]),
            'reflectance':np.array([.1,.2]),'transmittance':np.zeros(2),'absorptance':np.array([.9,.8])},
            'profiles':profiles(mean)}


class AnalysisTests(unittest.TestCase):
    def test_field_difference_alone_prevents_substitution(self):
        dip={'wavelength_nm':800.,'reflectance_min':.1,'half_prominence_width_nm':5.}
        tracked={'retained':True,'tracked_dips':[dip,dip]}
        criteria={'center_nm':2.,'absolute_R_change':.03,'relative_width_change':.1,
                  'relative_field_mean_change':.1,'relative_field_p95_change':.1}
        with patch('analyze_gold_mesh_results.track_pair',return_value=tracked):
            good=compare_cases(case(1.05),case(1.),[dip,dip],[800.],criteria)
            bad=compare_cases(case(1.2),case(1.),[dip,dip],[800.],criteria)
        self.assertTrue(good['small_effect_under_current_criteria'])
        self.assertFalse(bad['small_effect_under_current_criteria'])
        self.assertTrue(all(row['passes'] for row in bad['tracked_spectral_features']))

    def test_field_extrema_keep_actual_frequency(self):
        rows=profiles(1.)
        rows['nf_top_E'].append({'wavelength_nm':900.,'regions':{'all':stats(5.)}})
        result=sampled_field_extrema(rows)['nf_top_E']['all']
        self.assertEqual(result['p95_intensity_enhancement'],{'wavelength_nm':900.,'value':10.})

    def test_nearby_field_is_not_fabricated_at_spectral_center(self):
        result=fields_near_spectral_features(profiles(1.),[{'center_nm':803.}])[0]
        field=result['nearest_measured_fields']['nf_top_H']
        self.assertEqual(field['actual_wavelength_nm'],800.)
        self.assertEqual(field['offset_nm'],3.)


if __name__=='__main__':unittest.main()
