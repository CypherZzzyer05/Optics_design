"""Re-simulate a ranked candidate on finer grids without changing pilot results."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

from fdtd_pipeline.config import PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import atomic_json, atomic_npz, checksum, directory_lock
from explore_spectra import features


def verify(config, study, case_id, output, meshes=(10., 5.)):
    manifest=json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    entry=next(e for e in manifest['cases'] if e['id']==case_id)
    if entry['state']!='success':
        raise ValueError('Only completed study cases can be verified')
    path=study/'samples'/(case_id+'.npz')
    if checksum(path)!=entry['sha256']:
        raise ValueError('Candidate sample checksum mismatch')
    # Reject accidental comparison against a different pilot material / stack / band.
    if config.metadata()!=manifest['config']:
        raise ValueError('Verification config must match the pilot study config')
    with np.load(path,allow_pickle=False) as data:
        grid=data['wavelength_nm'].copy();previous=data['reflectance'].copy()
    previous_mesh=config.simulation.mesh_nm
    if not meshes or any(not np.isfinite(m) or m<=0 for m in meshes) or any(b>=a for a,b in zip((previous_mesh,*meshes),meshes)):
        raise ValueError('Verification meshes must be positive and strictly decreasing')
    records=[]
    with directory_lock(output):
        for mesh in meshes:
            material='Au (Gold) - Palik' if entry['metal']=='Au' else 'Ag (Silver) - Palik (0-2um)'
            settings=replace(config.simulation,mesh_nm=mesh,metal_mesh_nm=min(2.5,mesh/2),
                             mesh_accuracy=max(4,config.simulation.mesh_accuracy),metal_material=material,
                             auto_shutoff_min=1e-6,simulation_time_ps=max(4,config.simulation.simulation_time_ps))
            signature=hashlib.sha256(json.dumps({'config':config.metadata(),'case':entry['geometry_values'],
                                               'mesh':mesh,'settings':settings.__dict__},sort_keys=True).encode()).hexdigest()
            sample_path=output/f'{case_id}_mesh_{mesh:g}.npz'
            if sample_path.exists():
                with np.load(sample_path,allow_pickle=False) as data:
                    if str(data['verification_fingerprint'].item())!=signature:
                        raise ValueError('Verification sample has a different configuration')
                    result={k:data[k].copy() for k in data.files}
            else:
                print(f'Verifying {case_id}, mesh {mesh:g} nm',flush=True)
                with FDTDSimulator(settings,config.lumapi_path) as simulator:
                    result=simulator.run(config.make_geometry(entry['geometry_values']),output/f'{case_id}_mesh_{mesh:g}.fsp',
                                         include_silicon=entry['include_resonators'])
                atomic_npz(sample_path,**result,verification_fingerprint=np.array(signature))
            reflection=result['reflectance']
            current_dips=features(grid,reflection)
            old_dips=features(grid,previous)
            # Match the pilot's dominant feature by proximity, not by rank changes.
            pilot_dips=features(grid,previous) if not records else records[0]['previous_dips']
            anchor=pilot_dips[0]['wavelength_nm'] if pilot_dips else None
            current=min(current_dips,key=lambda d:abs(d['wavelength_nm']-anchor)) if current_dips and anchor is not None else None
            old=min(old_dips,key=lambda d:abs(d['wavelength_nm']-anchor)) if old_dips and anchor is not None else None
            shift=abs(current['wavelength_nm']-old['wavelength_nm']) if current and old else None
            depth_change=abs(current['reflectance_min']-old['reflectance_min']) if current and old else None
            width_change=(abs(current['half_prominence_width_nm']/old['half_prominence_width_nm']-1)
                          if current and old else None)
            passed=bool(current and old and shift<=2 and depth_change<=.03 and width_change<=.1)
            records.append({'mesh_nm':mesh,'previous_mesh_nm':previous_mesh,'dips':current_dips,
                            'previous_dips':old_dips,'matched_dip':current,
                            'peak_shift_nm':shift,'dip_depth_change':depth_change,'relative_width_change':width_change,
                            'max_R_difference':float(np.max(np.abs(reflection-previous))),
                            'rms_R_difference':float(np.sqrt(np.mean((reflection-previous)**2))),
                            'passes_pairwise_thresholds':passed,'status':int(result['status'])})
            previous,previous_mesh=reflection,mesh
            report={'case_id':case_id,'geometry_values':entry['geometry_values'],'steps':records,
                    'mesh_converged':bool(len(records)>=2 and records[-1]['passes_pairwise_thresholds']),
                    'thresholds':{'peak_shift_nm':2,'dip_depth_change':.03,'relative_width_change':.1},
                    'material_fit_validated':False,'note':'Mesh checks do not establish material-fit accuracy or modal origin.'}
            atomic_json(output/'verification.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/as2se3_au_pilot.json')
    parser.add_argument('--study',type=Path,default=PROJECT_ROOT/'data/near_ir_study')
    parser.add_argument('--case',required=True)
    parser.add_argument('--mesh',type=float,nargs='+',default=[10,5])
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/candidate_verification')
    args=parser.parse_args()
    print(json.dumps(verify(load_config(args.config),args.study,args.case,args.output.resolve(),tuple(args.mesh)),indent=2))
