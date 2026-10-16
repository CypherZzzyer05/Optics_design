"""Prepare and audit the uploaded real 100-sample dataset; never run FDTD."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--source',type=Path,default=Path('D:/')/'\u4eff\u771f'/'au100_metal8_100')
parser.add_argument('--generator-source',type=Path,default=Path(r'D:\projects\Deta_generator_dev'))
args = parser.parse_args()
root,source = args.project.resolve(),args.source.resolve()
destination = root/'data/as2se3_au100_metal8_100'
destination.mkdir(parents=True,exist_ok=True)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('dataset.npz','job.json'):
    target = destination/name
    if target.exists():
        assert sha(target) == sha(source/name), f'Refuse to replace {target}'
    else:
        shutil.copy2(source/name,target)
materials = destination/'materials'
materials.mkdir(exist_ok=True)
generator = args.generator_source.resolve()
for name in ('as2se3_joseph_700nm.csv','Joseph-700nm.yml','README.md'):
    shutil.copy2(generator/'src/deta_generator/assets/materials'/name,materials/name)
with zipfile.ZipFile(destination/'dataset.npz') as archive:
    assert archive.testzip() is None, 'NPZ CRC failure'
with np.load(destination/'dataset.npz',allow_pickle=False) as archive:
    data = {key:archive[key].copy() for key in archive.files}
metadata = json.loads(str(data['metadata_json'].item()))
assert data['spectra'].shape == (100,1000) and data['geometry_values'].shape == (100,7)
assert np.array_equal(data['sample_ids'],np.arange(100))
assert metadata['successful_samples'] == metadata['planned_samples'] == 100
assert np.all(data['status']==2)
for key in ('spectra','geometry_values','geometry_01','wavelength_nm','transmittance','absorptance'):
    assert np.isfinite(data[key]).all(),key
assert len(np.unique(data['geometry_values'],axis=0)) == 100
assert len(np.unique(data['spectra'],axis=0)) == 100
assert np.array_equal(data['geometry_values'][:,:5],data['geometry_nm'])
assert np.allclose(data['wavelength_nm'],np.linspace(700,1700,1000),rtol=0,atol=1e-10)
expected_labels = (data['geometry_values']-data['lower_values'])/(data['upper_values']-data['lower_values'])
assert np.allclose(expected_labels,data['geometry_01'],rtol=0,atol=1e-7)
pieces = []
for name,count in (('train',80),('val',10),('test',10)):
    indices = data[name+'_indices']
    assert indices.shape == (count,) and np.issubdtype(indices.dtype,np.integer)
    assert np.all(data['split_names'][indices] == name)
    pieces.append(indices)
assert np.array_equal(np.sort(np.concatenate(pieces)),np.arange(100))
sys.path.insert(0,str(generator/'src'))
from deta_generator.config import load_config
job = load_config(destination/'job.json')
assert job.simulation.metal_thickness_nm == 100
assert job.simulation.metal_mesh_nm == 8 and job.simulation.mesh_policy == 'metal_only'
assert job.metadata() == metadata['config'], 'Job physics/material/source differs from dataset metadata'
fingerprint = hashlib.sha256(json.dumps(job.metadata(),sort_keys=True).encode()).hexdigest()
assert fingerprint == metadata['config_fingerprint']
expected_geometry = np.array([job.geometry_values(g) for g in job.sample_geometries()])
np.testing.assert_allclose(data['geometry_values'],expected_geometry,rtol=0,atol=1e-10)
assert data['spectra'].min() >= -.03 and data['spectra'].max() <= 1.03
assert data['transmittance'].min() >= -.03 and data['transmittance'].max() <= 1.03
assert data['absorptance'].min() >= -.03 and data['absorptance'].max() <= 1.03
def counts(array):
    values,n = np.unique(array,return_counts=True)
    return {str(x):int(y) for x,y in zip(values,n)}
report = dict(dataset_sha256=sha(destination/'dataset.npz'),job_sha256=sha(destination/'job.json'),
    config_fingerprint=fingerprint,source=str(source),samples=100,spectrum_points=1000,
    parameter_names=data['parameter_names'].tolist(),parameter_units=data['parameter_units'].tolist(),
    splits={name:len(data[name+'_indices']) for name in ('train','val','test')},
    split_sample_ids={name:data['sample_ids'][data[name+'_indices']].tolist() for name in ('train','val','test')},
    all_geometry_matches_seeded_plan=True,all_status_2=True,unique_geometries=100,unique_spectra=100,
    simulation_time_limit_ps=counts(data['simulation_time_ps']),run_attempts=counts(data['run_attempts']),
    quality_level=metadata['config']['quality_level'],fdtd_versions=metadata['fdtd_versions'],
    reflectance_range=[float(data['spectra'].min()),float(data['spectra'].max())],
    transmittance_range=[float(data['transmittance'].min()),float(data['transmittance'].max())],
    absorptance_range=[float(data['absorptance'].min()),float(data['absorptance'].max())],
    negative_transmittance_points=int(np.sum(data['transmittance']<0)),
    negative_transmittance_sample_count=int(np.sum(np.any(data['transmittance']<0,axis=1))),
    rta_closure_max_abs=float(np.max(abs(data['spectra']+data['transmittance']+data['absorptance']-1))),
    closure_note='A is defined as 1-R-T; closure is not an independent accuracy test.',
    clipping_or_smoothing_performed=False,summary_json_received=(source/'summary.json').exists(),
    material_fit_errors={name:{k:v for k,v in row.items() if k!='fitted_index_ri'} for name,row in metadata['material_identity'].items()},
    limitations=['pipeline_validation physics; Au loss fit accuracy not certified',
                 'Only 10 held-out test structures; geometry prediction does not establish reconstructed spectral accuracy',
                 'Per-sample native logs and actual mesh arrays not included in the uploaded merged file'])
report_path = root/'docs/au100_metal8_100_data_audit.json'
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
template = json.loads((generator/'docs/train_metaformer_100.json').read_text(encoding='utf-8'))
config = root/'configs/train_au100_metal8_100.json'
config.write_text(json.dumps(template,indent=2)+'\n',encoding='utf-8')
if Path(__file__).resolve() != root/'audit_train100_dataset.py':
    shutil.copy2(Path(__file__),root/'audit_train100_dataset.py')
sys.path.insert(0,str(root))
from train_metaformer import load_training_config, load_training_data
training_config = load_training_config(config)
datasets = load_training_data(training_config)
print(json.dumps(dict(audit='passed',dataset_sha256=report['dataset_sha256'],
                     splits={name:len(d) for name,d in datasets.items()},config=str(config),
                     transmittance_range=report['transmittance_range']),indent=2))
