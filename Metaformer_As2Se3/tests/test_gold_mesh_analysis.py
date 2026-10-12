"""Keep measured-frequency provenance and enforce the accepted tolerances."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from analyze_gold_mesh_results import compare_cases,sampled_field_extrema,fields_near_spectral_features,verify_calibration,compare_own_peaks,gold_loss_interpretation


class GoldLossInterpretationTests(unittest.TestCase):
    def diagnostic(self,fit_n):
        rows=[{'wavelength_nm':w,'data_n':.2,'data_k':5.,'fit_n':fit_n,'fit_k':5.} for w in (839.,927.)]
        return {'cases':{'Au4':{'materials':{'Au (Gold) - Palik':{'at_target_wavelengths':rows}}}}}

    def test_positive_k_does_not_hide_missing_loss(self):
        result=gold_loss_interpretation(self.diagnostic(1e-8))
        self.assertTrue(result['near_lossless_fit_at_any_target'])
        self.assertAlmostEqual(result['target_loss_diagnostics'][0]['data_imaginary_permittivity'],2.)
        self.assertFalse(result['material_accuracy_certified'])

    def test_matching_loss_does_not_certify_experimental_accuracy(self):
        result=gold_loss_interpretation(self.diagnostic(.2))
        self.assertFalse(result['near_lossless_fit_at_any_target'])
        self.assertFalse(result['material_accuracy_certified'])
from gold_mesh_research import audit_has_complete_fields


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
    def test_bare_peaks_compare_their_own_grid_response(self):
        wavelength=np.arange(700.,1100.,.25)
        def signal(shift):return .05+.7*np.exp(-((wavelength-800-shift)/3)**2)+.4*np.exp(-((wavelength-900-shift)/5)**2)
        def fixture(shift):return {'data':{'wavelength_nm':wavelength,'reflectance':signal(shift)}}
        criteria={'center_nm':2.,'absolute_R_change':.03,'relative_width_change':.1}
        good=compare_own_peaks(fixture(.5),fixture(0),criteria)
        bad=compare_own_peaks(fixture(3),fixture(0),criteria)
        self.assertTrue(good['both_peaks_stable']);self.assertFalse(bad['both_peaks_stable'])
        self.assertEqual([row['second']['center_nm'] for row in good['tracked_peaks']],[800.,900.])
        self.assertTrue(all(abs(row['absolute_R_change'])<1e-9 for row in bad['tracked_peaks']))

    def test_calibration_flag_cannot_hide_current_normalization_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'vacuum8';directory.mkdir()
            record={name:{'passed':True,'maximum_mean_or_p95_deviation_from_unity':0.} for name in profiles(1.)}
            (directory/'normalization_check.json').write_text(json.dumps(record))
            fixture={'state':{'recipe':{'vacuum_reference':True}},'data':{}}
            bad_rows=[{'regions':{'all':{'mean_intensity_enhancement':1.05,'p95_intensity_enhancement':1.05}}}]
            with patch('analyze_gold_mesh_results.load_audited',return_value=fixture),patch('analyze_gold_mesh_results.profile_statistics',return_value=bad_rows):
                with self.assertRaisesRegex(ValueError,'current raw fields'):verify_calibration(root,'vacuum8',None)

    def test_old_audit_requires_every_field_and_coordinate(self):
        errors={f'nf_{plane}_{vector}_{suffix}':0. for plane in ('mid','top') for vector in ('E','H')
                for suffix in ('enhancement','wavelength_nm','incident_squared','x_nm','y_nm','z_nm')}
        audit={'native_fields_match_export':True,'native_vs_export_field_errors':errors}
        self.assertTrue(audit_has_complete_fields(audit))
        self.assertFalse(audit_has_complete_fields(audit|{'field_monitor_count':0}))
        errors.pop('nf_top_H_enhancement')
        self.assertFalse(audit_has_complete_fields(audit))

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
