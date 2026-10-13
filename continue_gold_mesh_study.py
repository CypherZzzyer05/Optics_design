"""Serial spectral/field matrix; incomplete or unaudited results stop later steps."""
import argparse
import json
from pathlib import Path
import time
import ctypes
import numpy as np

from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json,directory_lock,checksum
from gold_mesh_research import run_case,validate_saved_case,audit_has_complete_fields
from gold_mesh_metrics import make_plan,profile_statistics
from audit_gold_mesh_case import audit_case


def available_memory():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                 ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended')]
    value=Status();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError('Could not read available system memory')
    return int(value.available_physical)


def check_normalization(output,case_id):
    from gold_mesh_research import geometry_source
    geometry=geometry_source();report={}
    with np.load(output/case_id/'sample.npz') as z:
        for name in ('nf_mid_E','nf_top_E','nf_mid_H','nf_top_H'):
            values=profile_statistics(z,name,geometry)
            deviations=[max(abs(row['regions']['all'][key]-1) for key in
                        ('mean_intensity_enhancement','p95_intensity_enhancement')) for row in values]
            maximum=max(deviations)
            if maximum>.02:raise ValueError(f'Vacuum field normalization error exceeds 2%: {case_id}/{name}')
            report[name]={'maximum_mean_or_p95_deviation_from_unity':maximum,'passed':True}
    atomic_json(output/case_id/'normalization_check.json',report)
    print(f'Vacuum normalization passed: {case_id}',flush=True)


def reuse_verified_native_audit(directory):
    """Reuse only the audit of the same current NPZ and unchanged native FSP."""
    state=validate_saved_case(directory)
    audit=json.loads((directory/'native_audit.json').read_text())
    if (audit['sample_sha256']!=state['sample_sha256'] or audit['project_sha256']!=checksum(directory/'simulation.fsp')
            or audit['status']!=2 or not audit['native_geometry_verified'] or not audit['native_spectra_match_export']):
        raise ValueError('Current files no longer match their native audit: '+directory.name)
    if directory.name!='bare8_preliminary' and not audit_has_complete_fields(audit):
        raise ValueError('Native field audit is incomplete: '+directory.name)
    print('Verified and reused unchanged native audit: '+directory.name,flush=True)
    return audit


def continue_study(output,reuse_audits=False):
    output.mkdir(parents=True,exist_ok=True)
    state={'state':'running','steps':[],'started_at_unix':time.time(),
           'scope':'Au/air replacement and 8/4 nm at the same candidate geometry; E/H normalization and native audits'}
    def step(name,function,*args,**kwargs):
        state.update(current_step=name,updated_at_unix=time.time());atomic_json(output/'workflow.json',state)
        print('Gold/mesh workflow: '+name,flush=True)
        value=function(*args,**kwargs)
        state['steps'].append({'name':name,'state':'completed'});atomic_json(output/'workflow.json',state)
        return value
    # A separate coordinator lock avoids conflicts with the individual-case lock.
    with directory_lock(output/'coordinator'):
        try:
            step('bare8_preliminary',run_case,'bare8_preliminary',output)
            step('audit_bare8_preliminary',reuse_verified_native_audit if reuse_audits else audit_case,output/'bare8_preliminary')
            step('select_shared_real_frequency_samples',make_plan,output)
            cases=('vacuum8','vacuum4','Au8','bare8','Au4','bare4')
            for case in cases:
                existing=output/case/'state.json'
                if existing.exists() and json.loads(existing.read_text())['state']=='success':
                    step('verify_'+case,validate_saved_case,output/case)
                else:
                    step('preflight_'+case,run_case,case,output,preflight_only=True)
            for case in cases:
                completed=json.loads((output/case/'state.json').read_text())['state']=='success'
                resource=json.loads((output/case/'simulation.resources.json').read_text())
                recommended=float(resource.get('Memory_Recommended_Bytes',resource.get('Memory_Recommended')))
                free=available_memory()
                atomic_json(output/case/'memory_at_launch.json',{'recommended_bytes':recommended,'available_bytes':free,'recorded_at_unix':time.time()})
                # Do not launch an oversubscribed long solve without inspecting the setup.
                if not completed and recommended>free:
                    raise MemoryError(f'{case} recommends {recommended/1024**3:.2f} GiB but only {free/1024**3:.2f} GiB is available')
                step('run_'+case,run_case,case,output)
                step('native_audit_'+case,reuse_verified_native_audit if reuse_audits and completed else audit_case,output/case)
                if case.startswith('vacuum'):step('normalization_'+case,check_normalization,output,case)
            state.update(state='cases_completed',current_step='analyze_and_author_report',finished_cases_at_unix=time.time())
            atomic_json(output/'workflow.json',state)
        except BaseException as error:
            state.update(state='needs_attention',error=str(error),updated_at_unix=time.time())
            atomic_json(output/'workflow.json',state)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    p.add_argument('--reuse-audits',action='store_true',help='Reuse existing native audits only after current NPZ/FSP checksum and field checks; audit new cases normally')
    args=p.parse_args();continue_study(args.output.resolve(),args.reuse_audits)
