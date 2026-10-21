"""Verify the uploaded 1.0 job and stage its merged data for inverse training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def prepare(source, generator_source):
    source = Path(source).resolve()
    sys.path.insert(0, str(Path(generator_source).resolve()/'src'))
    from deta_generator.config import load_config
    from deta_generator.pipeline import checksum, load_sample, plan_checksum

    config = load_config(source/'job.json')
    archive_path = source/'merged_recovered/dataset.npz'
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(data['metadata_json'].item()))
    manifest = json.loads((source/'merged_recovered/manifest.json').read_text(encoding='utf-8'))
    layout = json.loads((source/'parallel_plan.json').read_text(encoding='utf-8-sig'))
    assert config.sample_count == layout['samples'] == 1000
    assert config.metadata() == metadata['config'] == manifest['config']
    fingerprint = hashlib.sha256(json.dumps(config.metadata(),sort_keys=True).encode()).hexdigest()
    assert fingerprint == metadata['config_fingerprint'] == manifest['config_fingerprint']
    assert plan_checksum(manifest) == manifest['sampling_plan_sha256']
    assert metadata['successful_samples'] == metadata['planned_samples'] == 1000
    assert data['spectra'].shape == (1000,1000) and data['geometry_values'].shape == (1000,7)
    assert np.array_equal(data['sample_ids'],np.arange(1000)) and np.all(data['status']==2)
    assert np.array_equal(data['geometry_values'][:,:5],data['geometry_nm'])
    assert np.allclose(data['wavelength_nm'],np.linspace(700,1700,1000),rtol=0,atol=1e-10)
    for name in ('spectra','transmittance','absorptance','geometry_values','geometry_01'):
        assert np.isfinite(data[name]).all(), name
    assert np.allclose(data['geometry_01'],config.normalize(data['geometry_values']),rtol=0,atol=1e-7)
    assert np.all((data['geometry_01']>=0)&(data['geometry_01']<=1))
    pieces = []
    for name,count in (('train',800),('val',100),('test',100)):
        indices=data[name+'_indices']
        assert indices.shape == (count,) and np.issubdtype(indices.dtype,np.integer)
        assert np.all(data['split_names'][indices]==name)
        pieces.append(indices)
    assert np.array_equal(np.sort(np.concatenate(pieces)),np.arange(1000))
    expected=np.array([config.geometry_values(g) for g in config.sample_geometries()])
    np.testing.assert_allclose(data['geometry_values'],expected,rtol=0,atol=1e-10)
    assert len(np.unique(data['geometry_values'],axis=0)) == 1000
    assert len(np.unique(data['spectra'],axis=0)) == 1000
    max_metal_step = 0.
    checked = []
    for shard_id in range(layout['workers']):
        directory=source/f'shard{shard_id}'
        shard=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        assert shard['config_fingerprint']==fingerprint
        assert plan_checksum(shard)==manifest['sampling_plan_sha256']
        assert shard['material_identity']==manifest['material_identity']
        assert shard['fdtd_version'] in metadata['fdtd_versions']
        for sample_id in range(shard_id,1000,layout['workers']):
            entry=shard['samples'][sample_id]
            assert entry['state']=='success' and entry==manifest['samples'][sample_id]
            raw=directory/'samples'/f'sample_{sample_id:06d}.npz'
            result=load_sample(raw,entry,config,fingerprint)
            assert checksum(source/'merged_recovered/samples'/raw.name)==entry['sha256']
            assert int(result['sample_id'])==sample_id
            assert np.array_equal(result['geometry_values'],data['geometry_values'][sample_id])
            assert entry['split']==str(data['split_names'][sample_id])
            for field,destination in (('reflectance','spectra'),('transmittance','transmittance'),('absorptance','absorptance')):
                np.testing.assert_array_equal(result[field].astype(np.float32),data[destination][sample_id])
            mesh=json.loads(str(result['mesh_audit_json'].item()))
            max_metal_step=max(max_metal_step,mesh['metal_region']['maximum_step_nm'])
            checked.append(sample_id)
        print(f'Verified shard {shard_id+1}/{layout["workers"]}; {len(checked)}/1000 samples',flush=True)
    assert sorted(checked)==list(range(1000)) and max_metal_step<10
    prior_overlap={}
    with np.load(ROOT/'examples/au100_metal8_100/dataset.npz',allow_pickle=False) as old:
        assert np.array_equal(old['geometry_values'],data['geometry_values'][:100])
        old_ids=old['sample_ids']
        new_test=data['sample_ids'][data['test_indices']]
        prior_overlap={name:np.intersect1d(new_test,old_ids[old[name+'_indices']]).tolist()
                       for name in ('train','val','test')}
        previously_seen_test_ids=np.intersect1d(new_test,old_ids).tolist()
    destination=ROOT/'examples/au100_metal8_1000'
    destination.mkdir(parents=True,exist_ok=True)
    for origin in (archive_path,source/'job.json'):
        target=destination/origin.name
        if target.exists():
            assert checksum(target)==checksum(origin), 'Existing staged data differs'
        else:
            shutil.copy2(origin,target)
    shutil.copytree(source/'materials',destination/'materials',dirs_exist_ok=True)
    report={'source':str(source),'dataset_sha256':checksum(archive_path),'config_fingerprint':fingerprint,
        'raw_shards_verified':layout['workers'],'raw_samples_verified':len(checked),
        'samples':1000,'spectrum_points':1000,'parameter_names':data['parameter_names'].tolist(),
        'parameter_units':data['parameter_units'].tolist(),'splits':{'train':800,'val':100,'test':100},
        'all_status_2':True,'all_raw_checksums_and_merged_arrays_match':True,
        'all_geometries_match_original_seeded_plan':True,'unique_geometries':1000,'unique_spectra':1000,
        'metal_maximum_step_nm':max_metal_step,'quality_level':config.quality_level,
        'fdtd_versions':metadata['fdtd_versions'],'preprocessing':'No clipping, smoothing or resampling; normalization fits training rows only',
        'previous_100_dataset_overlap_in_current_test':prior_overlap,
        'previously_seen_test_sample_ids':previously_seen_test_ids,
        'previously_unseen_test_count':100-len(previously_seen_test_ids),
        'reflectance_range':[float(data['spectra'].min()),float(data['spectra'].max())],
        'transmittance_range':[float(data['transmittance'].min()),float(data['transmittance'].max())],
        'absorptance_range':[float(data['absorptance'].min()),float(data['absorptance'].max())],
        'source_data_modified':False,'solver_run_performed':False,
        'physical_stack':'Generator 1.0: Au below finite SiO2, then As2Se3 tetramer',
        'limitations':['Material fit accuracy remains pipeline_validation',
                       'Original test split retained; report the 88 previously unseen test structures separately']}
    output=ROOT/'results/inverse_au100_1000'
    output.mkdir(parents=True,exist_ok=True)
    (output/'data_audit.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'audit':'passed','staged_dataset':str(destination/'dataset.npz'),
                      'raw_verified':len(checked),'splits':report['splits'],
                      'previously_unseen_test_count':report['previously_unseen_test_count']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--generator-source',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.source,args.generator_source)
