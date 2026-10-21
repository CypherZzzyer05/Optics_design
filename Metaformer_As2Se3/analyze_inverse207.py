"""Compare native reference and inverse-geometry spectra with held-out target 207."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from compare_double_mesh import compare_curves
from explore_double_dips import double_pair
from fdtd_pipeline.pipeline import checksum

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/inverse_roundtrip_207'


def analyze(reference_only=False):
    catalog=json.loads((ROOT/'results/feature_targets_1000/analysis.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'examples/au100_metal8_1000/dataset.npz',allow_pickle=False) as archive:
        matches=np.flatnonzero(archive['sample_ids']==207)
        assert len(matches)==1
        i=int(matches[0])
        source={key:archive[key][i].copy() for key in ('spectra','transmittance','absorptance')}
        wavelength=archive['wavelength_nm'].copy()
    cases={}
    for name in (('reference',) if reference_only else ('reference','prediction')):
        directory=ROOT/'runs/inverse_roundtrip_207'/name
        report=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        path=directory/'spectrum.npz'
        assert checksum(path)==report['result_sha256']
        assert report['dataset_sha256']==catalog['dataset_sha256']
        assert report['inverse_checkpoint_sha256']==catalog['inverse_checkpoint_sha256']
        with np.load(path,allow_pickle=False) as archive:
            sample={key:archive[key].copy() for key in archive.files}
        assert int(sample['status'])==2
        assert np.allclose(sample['wavelength_nm'],wavelength,rtol=0,atol=1e-5)
        assert np.array_equal(sample['geometry_values'],report['geometry_values'])
        comparison=compare_curves(wavelength,source['spectra'],sample['reflectance'])
        cases[name]={'data':sample,'report':report,'source_comparison':comparison,'directory':directory}
    original=cases['reference']
    result={'sample_id':207,'dataset_sha256':catalog['dataset_sha256'],
        'inverse_checkpoint_sha256':catalog['inverse_checkpoint_sha256'],
        'reference_only':reference_only,'reference_reproduction':original['source_comparison'],
        'comparison_criteria_note':'Previously used 2nm / 0.03 absolute R / 10% half-prominence width, as preliminary spectral checks',
        'local_material_fits_match_server_at_1e_7':all(row['matches_server_at_1e_7'] for row in
            original['report']['material_fit_differences_from_server'].values()),
        'cases':{name:{'status':int(case['data']['status']),'elapsed_seconds':float(case['data']['elapsed_seconds']),
                      'geometry_values':case['report']['geometry_values'],
                      'local_fdtd_version':case['report']['local_fdtd_version'],
                      'source_fdtd_versions':case['report']['source_fdtd_versions'],
                      'result_sha256':case['report']['result_sha256'],
                      'mesh_audit':json.loads(str(case['data']['mesh_audit_json'].item()))}
                 for name,case in cases.items()},
        'quality_level':'pipeline_validation','model_or_source_data_modified':False,
        'scope':'Generator 1.0 Au100 below finite SiO2; not the 2.0 middle-metal stack'}
    if not reference_only:
        predicted=cases['prediction']
        for name,identity in original['report']['material_identity'].items():
            np.testing.assert_allclose(identity['fitted_index_ri'],
                predicted['report']['material_identity'][name]['fitted_index_ri'],rtol=1e-7,atol=1e-7)
        assert original['report']['local_fdtd_version']==predicted['report']['local_fdtd_version']
        result['prediction_vs_uploaded_target']=predicted['source_comparison']
        result['prediction_vs_local_reference']=compare_curves(wavelength,original['data']['reflectance'],predicted['data']['reflectance'])
        result['prediction_retains_deep_resolved_double_pair']=bool(double_pair(wavelength,predicted['data']['reflectance'])['dominant_pair'])
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,case in cases.items():
        for original_name,new_name in (('spectrum.npz',name+'.npz'),('spectrum.csv',name+'.csv'),('result.json',name+'_native.json')):
            shutil.copy2(case['directory']/original_name,OUTPUT/new_name)
    suffix='_reference' if reference_only else ''
    (OUTPUT/('analysis'+suffix+'.json')).write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    plot(wavelength,source,cases,suffix)
    brief={'reference_reproduction_passes':result['reference_reproduction']['both_dips_stable'],
           'reference_R_max_abs_difference':result['reference_reproduction']['maximum_R_difference'],
           'reference_R_rmse':result['reference_reproduction']['rms_R_difference']}
    if not reference_only:
        brief.update(predicted_pair_preserved=result['prediction_retains_deep_resolved_double_pair'],
                     prediction_matches_uploaded_target=result['prediction_vs_uploaded_target']['both_dips_stable'],
                     prediction_R_rmse=result['prediction_vs_uploaded_target']['rms_R_difference'])
    print(json.dumps(brief,indent=2))
    return result


def plot(wavelength,source,cases,suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors={'reference':'#22a181','prediction':'#c5742e'}
    labels={'reference':'原始结构本机重算','prediction':'网络预测结构回仿真'}
    for ax in axes[0]:
        ax.plot(wavelength,source['spectra'],color='#2457bd',label='服务器原目标',linewidth=1.8)
        for name,case in cases.items():
            ax.plot(wavelength,case['data']['reflectance'],label=labels[name],color=colors[name],linestyle='--' if name=='reference' else '-')
        ax.set(xlabel='波长（nm）',ylabel='R',ylim=(-.02,1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    axes[0,0].set_title('反射光谱：完整波段')
    axes[0,1].set(xlim=(735,920),title='原始两谷所在波段')
    for ax,field,label in ((axes[1,0],'transmittance','透射率 T'),(axes[1,1],'absorptance','总吸收率 A')):
        ax.plot(wavelength,source[field],color='#2457bd',label='服务器原目标')
        for name,case in cases.items():
            ax.plot(wavelength,case['data'][field],color=colors[name],label=labels[name],linestyle='--' if name=='reference' else '-')
        ax.set(xlabel='波长（nm）',ylabel=label,title=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('样本 207：真实 FDTD 对照'+('（仅原始结构复现）' if len(cases)==1 else '与逆向结构回仿真'),
                 fontsize=16,fontweight='bold')
    fig.savefig(OUTPUT/('spectral_comparison'+suffix+'.png'),dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only',action='store_true')
    args=parser.parse_args()
    analyze(args.reference_only)
