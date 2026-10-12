"""Archive only complete native-audited comparison cases; mark partial progress."""
import argparse
import json
import shutil
from pathlib import Path
from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import checksum,atomic_json
from gold_mesh_research import validate_saved_case,audit_has_complete_fields


def archive(output,destination):
    destination.mkdir(parents=True,exist_ok=True)
    completed=[]
    for name in ('bare8_preliminary','vacuum8','vacuum4','Au8','bare8','Au4','bare4'):
        source=output/name
        if not (source/'native_audit.json').exists():continue
        state=validate_saved_case(source)
        audit=json.loads((source/'native_audit.json').read_text())
        if (audit['sample_sha256']!=state['sample_sha256'] or audit['project_sha256']!=checksum(source/'simulation.fsp')
                or audit['status']!=2 or not audit['native_geometry_verified'] or not audit['native_spectra_match_export']):
            raise ValueError('Audit no longer verifies '+name)
        if name!='bare8_preliminary' and not audit_has_complete_fields(audit):
            raise ValueError('The field export is incomplete: '+name)
        target=destination/'cases'/name;target.mkdir(parents=True,exist_ok=True)
        for filename in ('sample.npz','state.json','native_audit.json','simulation.resources.json','normalization_check.json'):
            if (source/filename).exists():shutil.copyfile(source/filename,target/filename)
        completed.append(name)
    if (output/'sampling_plan.json').exists():shutil.copyfile(output/'sampling_plan.json',destination/'sampling_plan.json')
    complete=all(name in completed for name in ('Au8','bare8','Au4','bare4','vacuum8','vacuum4'))
    atomic_json(destination/'archive_status.json',{'complete_matrix':complete,'completed_cases':completed,
        'scope':'Completed native-audited exports only; large FSP projects remain local and their checksums are recorded.',
        'no_final_decision_yet':not complete})
    files={str(path.relative_to(destination)).replace('\\','/'):checksum(path)
           for path in destination.rglob('*') if path.is_file() and path.name!='checksums.json'}
    atomic_json(destination/'checksums.json',files)
    return {'complete_matrix':complete,'completed_cases':completed,'files':len(files)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'data/gold_mesh_comparison')
    parser.add_argument('--destination',type=Path,default=PROJECT_ROOT/'research_results/gold_mesh_comparison/raw')
    args=parser.parse_args();print(json.dumps(archive(args.output,args.destination)))
