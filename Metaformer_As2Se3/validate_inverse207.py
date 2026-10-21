"""Native 1.0-stack reference/prediction check for held-out target 207."""
import argparse
from dataclasses import replace
import ctypes
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parent


def memory_state():
    buffer=ctypes.create_string_buffer(64)
    struct.pack_into('I',buffer,0,64)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):
        raise OSError('Cannot read current Windows memory budget')
    total,available,limit,free_commit=struct.unpack_from('QQQQ',buffer,8)
    return {'available_ram_gib':available/2**30,'available_commit_gib':free_commit/2**30,
            'commit_limit_gib':limit/2**30,'c_free_gib':shutil.disk_usage('C:/').free/2**30}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator-source',type=Path,required=True)
    parser.add_argument('--case',choices=('reference','prediction'),default='reference')
    parser.add_argument('--lumapi')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--run',action='store_true',help='Advance the solver after the native preflight')
    args=parser.parse_args()
    sys.path.insert(0,str(args.generator_source.resolve()/'src'))
    from deta_generator.config import load_config
    from deta_generator.lumerical import FDTDSimulator,locate_lumapi,mesh_audit
    from deta_generator.pipeline import atomic_json,atomic_npz,checksum,directory_lock
    if not 1<=args.threads<=psutil.cpu_count(logical=True):
        raise ValueError('Threads exceed the local CPU budget')
    catalog=json.loads((ROOT/'results/feature_targets_1000/analysis.json').read_text(encoding='utf-8'))
    candidate=next(row for row in catalog['selected_targets'] if row['sample_id']==207)
    key='geometry_values' if args.case=='reference' else 'inverse_predicted_geometry_values'
    values=candidate[key]
    job=load_config(ROOT/'examples/au100_metal8_1000/job.json')
    settings=replace(job.simulation,threads=args.threads,processes=1,save_projects=True)
    geometry=job.make_geometry(values)
    api=str(locate_lumapi(args.lumapi))
    output=ROOT/'runs/inverse_roundtrip_207'/args.case
    output.mkdir(parents=True,exist_ok=True)
    result_path=output/'spectrum.npz'
    if args.run and result_path.exists():
        raise FileExistsError(f'Saved result already exists: {result_path}; inspect it before another solve')
    with np.load(ROOT/'examples/au100_metal8_1000/dataset.npz',allow_pickle=False) as archive:
        source_metadata=json.loads(str(archive['metadata_json'].item()))
    with directory_lock(output),FDTDSimulator(settings,api) as simulator:
        materials=simulator.material_identity()
        differences={}
        for name,material in materials.items():
            local=np.asarray(material['fitted_index_ri'])
            server=np.asarray(source_metadata['material_identity'][name]['fitted_index_ri'])
            differences[name]={'max_n_difference':float(np.max(np.abs(local[:,0]-server[:,0]))),
                'max_k_difference':float(np.max(np.abs(local[:,1]-server[:,1]))),
                'matches_server_at_1e_7':bool(np.allclose(local,server,rtol=1e-7,atol=1e-7))}
        simulator.build(geometry)
        grids={axis:np.asarray(simulator.fdtd.getresult('FDTD',axis)).reshape(-1) for axis in ('x','y','z')}
        audit=mesh_audit(grids,settings)
        resources=simulator.fdtd.runsystemcheck()
        recommendation=resources.get('Memory_Recommended_Bytes',resources.get('Memory_Recommended'))
        if recommendation is None or not np.isfinite(recommendation) or float(recommendation)<=0:
            raise ValueError('Native memory recommendation missing')
        memory=memory_state()
        report={'case':args.case,'sample_id':207,'geometry_values':values,'threads':args.threads,
                'native_layout_verified':True,'mesh_audit':audit,
                'recommended_ram_gib':float(recommendation)/2**30,'memory_while_api_open':memory,
                'local_fdtd_version':simulator.version,'source_fdtd_versions':source_metadata['fdtd_versions'],
                'material_fit_differences_from_server':differences,'material_identity':materials,
                'dataset_sha256':catalog['dataset_sha256'],
                'inverse_checkpoint_sha256':catalog['inverse_checkpoint_sha256'],
                'solver_run_performed':False,'physical_stack':'Au100 / finite SiO2 350nm / As2Se3 tetramer'}
        atomic_json(output/'preflight.json',report)
        print(json.dumps({key:value for key,value in report.items() if key!='material_identity'},indent=2),flush=True)
        if not args.run:
            return
        # Commit availability matters here because the user's C: disk is nearly full.
        # Refuse a solve that would need Windows to expand its paging file.
        required=float(recommendation)/2**30
        if min(memory['available_ram_gib'],memory['available_commit_gib'])<required:
            raise MemoryError('Native recommendation exceeds RAM or current commit headroom; no solver started')
        print('Starting real FDTD '+args.case+'; source dataset and trained weights remain unchanged.',flush=True)
        result=simulator.run(geometry,output/'projects'/'sample_000207.fsp')
        atomic_npz(result_path,**result,geometry_values=np.asarray(values),
                   sample_id=np.int64(207),source_dataset_sha256=np.array(catalog['dataset_sha256']))
        np.savetxt(output/'spectrum.csv',np.column_stack([result[name] for name in
            ('wavelength_nm','reflectance','transmittance','absorptance')]),delimiter=',',
            header='wavelength_nm,reflectance,transmittance,absorptance',comments='')
        report.update(solver_run_performed=True,status=int(result['status']),
                      elapsed_seconds=float(result['elapsed_seconds']),result_sha256=checksum(result_path))
        atomic_json(output/'result.json',report)
        print(json.dumps({'completed':args.case,'status':int(result['status']),
                          'elapsed_seconds':float(result['elapsed_seconds']),'spectrum':str(result_path)},indent=2),flush=True)


if __name__=='__main__':
    main()
