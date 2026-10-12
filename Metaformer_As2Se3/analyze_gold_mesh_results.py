"""Evaluate the two requested comparisons from completed, audited real cases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks,peak_widths

from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json,checksum
from gold_mesh_research import geometry_source,validate_saved_case,audit_has_complete_fields
from gold_mesh_metrics import profile_statistics
from explore_double_dips import double_pair
from analyze_angle_controls import track_pair

CASES=('Au8','bare8','Au4','bare4')


def load_audited(output,case_id):
    directory=output/case_id;state=validate_saved_case(directory)
    audit=json.loads((directory/'native_audit.json').read_text())
    if (audit['sample_sha256']!=state['sample_sha256'] or audit['project_sha256']!=checksum(directory/'simulation.fsp')
            or audit['status']!=2 or not audit['native_geometry_verified']
            or not audit['native_spectra_match_export'] or not audit_has_complete_fields(audit)):
        raise ValueError('Current native audit does not verify the complete spectral/field case')
    with np.load(directory/'sample.npz') as z:data={key:z[key] for key in z.files}
    return {'state':state,'audit':audit,'data':data,'directory':directory}


def spectral_features(wavelength,signal):
    found=[]
    for sign,kind in ((1,'maximum'),(-1,'minimum')):
        peaks,properties=find_peaks(sign*signal,prominence=.02)
        width,_,left,right=peak_widths(sign*signal,peaks,rel_height=.5)
        for i,index in enumerate(peaks):
            found.append({'kind':kind,'center_nm':float(wavelength[index]),'value':float(signal[index]),
                          'prominence':float(properties['prominences'][i]),
                          'half_prominence_width_nm':float(np.interp(right[i],np.arange(len(wavelength)),wavelength)
                               -np.interp(left[i],np.arange(len(wavelength)),wavelength))})
    return sorted(found,key=lambda x:x['prominence'],reverse=True)[:8]


def relative_change(fine,reference):return float(abs(fine-reference)/max(abs(reference),1e-12))


def nearest_profile(rows,wavelength):
    row=min(rows,key=lambda r:abs(r['wavelength_nm']-wavelength))
    if abs(row['wavelength_nm']-wavelength)>1e-5:raise ValueError('A required common field wavelength is missing')
    return row


def sampled_field_extrema(profiles):
    """Keep frequency/region provenance for maxima over measured samples only."""
    result={}
    for name,rows in profiles.items():
        result[name]={}
        for region in rows[0]['regions']:
            result[name][region]={}
            for metric in ('mean_intensity_enhancement','p95_intensity_enhancement','maximum_intensity_enhancement'):
                selected=max(rows,key=lambda row:row['regions'][region][metric])
                result[name][region][metric]={'wavelength_nm':selected['wavelength_nm'],
                                             'value':selected['regions'][region][metric]}
    return result


def fields_near_spectral_features(profiles,features):
    """Show each stack's own response, retaining the actual field frequency."""
    result=[]
    for feature in features[:4]:
        fields={}
        for name,rows in profiles.items():
            selected=min(rows,key=lambda row:abs(row['wavelength_nm']-feature['center_nm']))
            fields[name]={'actual_wavelength_nm':selected['wavelength_nm'],
                          'offset_nm':abs(selected['wavelength_nm']-feature['center_nm']),
                          'regions':selected['regions']}
        result.append({'spectral_feature':feature,'nearest_measured_fields':fields})
    return result


def verify_calibration(output,case_id,geometry):
    """Recompute calibration from audited raw fields; do not trust a stale flag."""
    case=load_audited(output,case_id)
    if not case['state']['recipe']['vacuum_reference']:raise ValueError('Calibration is not a vacuum experiment')
    path=output/case_id/'normalization_check.json'
    stored=json.loads(path.read_text())
    names=('nf_mid_E','nf_top_E','nf_mid_H','nf_top_H')
    if set(stored)!=set(names):raise ValueError('A vacuum profile is missing')
    calculated={}
    for name in names:
        rows=profile_statistics(case['data'],name,geometry)
        deviation=max(abs(row['regions']['all'][metric]-1) for row in rows for metric in
                      ('mean_intensity_enhancement','p95_intensity_enhancement'))
        if (not stored[name]['passed'] or deviation>.02 or
            not np.isclose(deviation,stored[name]['maximum_mean_or_p95_deviation_from_unity'],rtol=0,atol=1e-12)):
            raise ValueError('Stored vacuum normalization does not verify current raw fields')
        calculated[name]=float(deviation)
    return {'sample_sha256':case['state']['sample_sha256'],'native_audit_sha256':checksum(case['directory']/'native_audit.json'),
            'normalization_check_sha256':checksum(path),'maximum_deviations':calculated,'passed':True}


def compare_own_peaks(first,second,criteria):
    """Auxiliary bare-stack grid check; absent Au dips are not a mesh error."""
    reference=[feature for feature in spectral_features(second['data']['wavelength_nm'],second['data']['reflectance'])
               if feature['kind']=='maximum'][:2]
    candidates=[feature for feature in spectral_features(first['data']['wavelength_nm'],first['data']['reflectance'])
                if feature['kind']=='maximum']
    rows=[];used=set()
    for peak in reference:
        options=[(i,value) for i,value in enumerate(candidates) if i not in used and abs(value['center_nm']-peak['center_nm'])<=25]
        if not options:
            rows.append({'second':peak,'first':None,'passes':False});continue
        index,selected=min(options,key=lambda pair:abs(pair[1]['center_nm']-peak['center_nm']));used.add(index)
        shift=abs(selected['center_nm']-peak['center_nm']);height=abs(selected['value']-peak['value'])
        width=relative_change(selected['half_prominence_width_nm'],peak['half_prominence_width_nm'])
        rows.append({'first':selected,'second':peak,'center_shift_nm':shift,'absolute_R_change':height,
                     'relative_width_change':width,'passes':bool(shift<=criteria['center_nm'] and height<=criteria['absolute_R_change']
                                                                and width<=criteria['relative_width_change'])})
    return {'reference_peaks':reference,'tracked_peaks':rows,'both_peaks_stable':bool(len(rows)==2 and all(row['passes'] for row in rows)),
            'scope':'Auxiliary reflection-peak comparison with the same numerical tolerances; it does not replace the accepted double-dip substitution gate.'}


def compare_fields(first,second,targets,criteria):
    field=[]
    for name in ('nf_top_E','nf_mid_E','nf_top_H','nf_mid_H'):
        regions=('all',) if '_top_' in name else ('pillar_bulk','air_away_from_edges')
        for wavelength in targets:
            left=nearest_profile(first['profiles'][name],wavelength)
            right=nearest_profile(second['profiles'][name],wavelength)
            for region in regions:
                r1,r2=left['regions'][region],right['regions'][region]
                mean=relative_change(r1['mean_intensity_enhancement'],r2['mean_intensity_enhancement'])
                p95=relative_change(r1['p95_intensity_enhancement'],r2['p95_intensity_enhancement'])
                field.append({'profile':name,'region':region,'wavelength_nm':wavelength,'first':r1,'second':r2,
                              'mean_ratio_first_over_second':r1['mean_intensity_enhancement']/max(r2['mean_intensity_enhancement'],1e-12),
                              'p95_ratio_first_over_second':r1['p95_intensity_enhancement']/max(r2['p95_intensity_enhancement'],1e-12),
                              'relative_mean_change':mean,'relative_p95_change':p95,
                              'passes':bool(mean<=criteria['relative_field_mean_change'] and p95<=criteria['relative_field_p95_change'])})
    return field


def compare_cases(first,second,reference_dips,targets,criteria):
    a,b=first['data'],second['data']
    if not np.array_equal(a['wavelength_nm'],b['wavelength_nm']):raise ValueError('Main spectral grids differ')
    first_pair=track_pair(a['wavelength_nm'],a['reflectance'],reference_dips)
    second_pair=track_pair(b['wavelength_nm'],b['reflectance'],reference_dips)
    spectral=[]
    if first_pair['tracked_dips'] and second_pair['tracked_dips']:
        for left,right in zip(first_pair['tracked_dips'],second_pair['tracked_dips']):
            shift=abs(left['wavelength_nm']-right['wavelength_nm'])
            depth=abs(left['reflectance_min']-right['reflectance_min'])
            width=relative_change(left['half_prominence_width_nm'],right['half_prominence_width_nm'])
            spectral.append({'first':left,'second':right,'center_shift_nm':shift,'absolute_R_change':depth,
                             'relative_width_change':width,'passes':bool(shift<=criteria['center_nm']
                             and depth<=criteria['absolute_R_change'] and width<=criteria['relative_width_change'])})
    field=compare_fields(first,second,targets,criteria)
    return {'first_case':first['state']['recipe']['case_id'],'second_case':second['state']['recipe']['case_id'],
            'reference_pair_preserved':bool(first_pair['retained'] and second_pair['retained']),
            'tracked_spectral_features':spectral,'field_at_common_wavelengths':field,
            'maximum_absolute_R_difference':float(np.max(abs(a['reflectance']-b['reflectance']))),
            'maximum_absolute_T_difference':float(np.max(abs(a['transmittance']-b['transmittance']))),
            'maximum_absolute_A_difference':float(np.max(abs(a['absorptance']-b['absorptance']))),
            'small_effect_under_current_criteria':bool(first_pair['retained'] and second_pair['retained']
                and len(spectral)==2 and all(x['passes'] for x in spectral) and all(x['passes'] for x in field))}


def analyze(output,destination):
    geometry=geometry_source();plan=json.loads((output/'sampling_plan.json').read_text())
    cases={name:load_audited(output,name) for name in CASES}
    calibrations={mesh:verify_calibration(output,'vacuum'+str(mesh),geometry) for mesh in (8,4)}
    from audit_gold_mesh_materials import audit_materials
    material_diagnostics=audit_materials(output,destination/'materials',plan['gold_target_wavelengths_nm'])
    if not material_diagnostics['shared_fits_match']:
        raise ValueError('Shared material fits differ across controls; investigate before a final substitution decision')
    base=cases['Au4']['state']['recipe']
    for name,case in cases.items():
        recipe=case['state']['recipe']
        if recipe['geometry_values']!=base['geometry_values'] or recipe['vacuum_reference']:
            raise ValueError('Physical geometry changed between the four controls')
        for key,value in base['simulation'].items():
            if key not in ('mesh_nm','maximum_mesh_nm','monitor_downsample_xy','power_validation_wavelengths_nm') and recipe['simulation'][key]!=value:
                raise ValueError('Another physical/spectral setting changed: '+key)
        for key in ('electric_wavelengths_nm','magnetic_wavelengths_nm','material_csv_sha256'):
            if recipe[key]!=base[key]:raise ValueError('Field or material source changed')
        case['profiles']={profile:profile_statistics(case['data'],profile,geometry) for profile in
                          ('nf_mid_E','nf_top_E','nf_mid_H','nf_top_H')}
        vacuum=load_audited(output,'vacuum'+name[-1])
        for profile in ('nf_mid_E','nf_top_E','nf_mid_H','nf_top_H'):
            if not np.array_equal(case['data'][profile+'_wavelength_nm'],vacuum['data'][profile+'_wavelength_nm']):
                raise ValueError('Sample and vacuum calibration frequencies differ')
    reference=double_pair(cases['Au4']['data']['wavelength_nm'],cases['Au4']['data']['reflectance'])['dominant_pair']
    if not reference:raise ValueError('The verified Au4 case does not retain the required double-dip candidate')
    targets=plan['gold_target_wavelengths_nm'];criteria=plan['unchanged_criteria']
    comparisons={name:compare_cases(cases[left],cases[right],reference['dips'],targets,criteria) for name,left,right in
                 (('Au_vs_bare_8nm','Au8','bare8'),('Au_vs_bare_4nm','Au4','bare4'),
                  ('Au_8nm_vs_4nm','Au8','Au4'),('bare_8nm_vs_4nm','bare8','bare4'))}
    comparisons['bare_8nm_vs_4nm']['own_peak_mesh_comparison']=compare_own_peaks(cases['bare8'],cases['bare4'],criteria)
    auxiliary=comparisons['bare_8nm_vs_4nm']['own_peak_mesh_comparison']
    auxiliary['field_at_shared_preliminary_peak_wavelengths']=compare_fields(cases['bare8'],cases['bare4'],plan['bare_peak_wavelengths_nm'],criteria)
    auxiliary['own_peaks_and_sampled_fields_stable']=bool(auxiliary['both_peaks_stable'] and len(plan['bare_peak_wavelengths_nm'])==2
            and all(row['passes'] for row in auxiliary['field_at_shared_preliminary_peak_wavelengths']))
    remove_gold=comparisons['Au_vs_bare_4nm']['small_effect_under_current_criteria']
    stack='bare' if remove_gold else 'Au'
    adopt8=comparisons[stack+'_8nm_vs_4nm']['small_effect_under_current_criteria']
    summaries={}
    for name,case in cases.items():
        data=case['data'];recipe=case['state']['recipe']
        features={signal:spectral_features(data['wavelength_nm'],data[signal]) for signal in
                  ('reflectance','transmittance','absorptance')}
        summaries[name]={'gold_present':bool(data['gold_present']),'maximum_mesh_nm':recipe['simulation']['maximum_mesh_nm'],
                         'elapsed_seconds':float(data['elapsed_seconds']),'status':int(data['status']),
                         'actual_mesh':case['audit']['actual_mesh'],'sample_sha256':case['state']['sample_sha256'],
                         'native_audit_sha256':checksum(case['directory']/'native_audit.json'),
                         'spectral_features':features,'field_profiles':case['profiles'],
                         'region_quadrature':{region:{key:value[key] for key in ('sampled_area_nm2','sampled_nodes')}
                                              for region,value in case['profiles']['nf_mid_E'][0]['regions'].items()},
                         'sampled_field_extrema':sampled_field_extrema(case['profiles']),
                         'fields_near_own_spectral_features':fields_near_spectral_features(case['profiles'],features['reflectance'])}
    report={'complete_matrix':True,'geometry_values':base['geometry_values'],'criteria':criteria,'cases':summaries,
            'vacuum_calibrations':calibrations,
            'material_diagnostics':material_diagnostics,
            'exact_eroded_pillar_area_nm2':float(4*(geometry.l-20)**2),
            'comparisons':comparisons,'decision':{'remove_gold':remove_gold,'selected_stack':stack,
             'adopt_8nm':adopt8,'selected_final_mesh_nm':8 if adopt8 else 4},
            'scope':'Current fixed geometry and sampled 700-1700 nm spectra; two XY planes and selected E/H frequencies.',
            'limitations':['Total A does not isolate As2Se3 absorption.',
                           'Incident-normalized fields include background interference from the stack.',
                           'Mean and p95 use native-grid area quadrature and a fixed 10 nm edge exclusion at the mid-plane.',
                           'Maximum fields concern sampled planes/frequencies and can be sensitive to corners.',
                           'Criteria assess direct substitution of the current double-dip response; another desired spectral shape may lead to another design decision.',
                           'Material accuracy and experimental behavior are not established by these numerical comparisons.']}
    destination.mkdir(parents=True,exist_ok=True);atomic_json(destination/'analysis.json',report)
    with (destination/'field_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        rows=[]
        for comparison,value in comparisons.items():
            for row in value['field_at_common_wavelengths']:
                rows.append({k:row[k] for k in ('profile','region','wavelength_nm','mean_ratio_first_over_second',
                                               'p95_ratio_first_over_second','relative_mean_change','relative_p95_change','passes')}|{'comparison':comparison})
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    p.add_argument('--destination',type=Path,default=PROJECT_ROOT/'research_results/gold_mesh_comparison')
    args=p.parse_args();r=analyze(args.output,args.destination);print(json.dumps(r['decision']))
