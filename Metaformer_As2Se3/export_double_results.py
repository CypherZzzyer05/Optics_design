"""Archive completed strict-mesh spectra only after independent native audits."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from compare_double_mesh import read_case
from explore_double_dips import rank
from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json, checksum


def export(directory, output):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    successful = [e for e in manifest['cases'] if e['state'] == 'success']
    if not successful:
        raise ValueError('There are no completed spectra to export')
    verified = []
    # Validate everything before writing an archive snapshot.
    for entry in successful:
        case = read_case(directory, entry['id'])
        audit_path = directory / (entry['id'] + '_native_audit.json')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        project = directory / 'projects' / (entry['id'] + '.fsp')
        if (audit['sample_sha256'] != entry['sha256'] or audit['project_sha256'] != checksum(project)
                or audit['status'] != 2 or not audit['native_geometry_verified']
                or not audit['native_spectra_match_export'] or not audit['all_actual_mesh_cells_below_10_nm']):
            raise ValueError('Native audit does not verify the current saved spectrum and project')
        verified.append((case, audit_path))
    report = rank(directory, manifest)
    (output / 'samples').mkdir(parents=True, exist_ok=True)
    for case, audit_path in verified:
        shutil.copy2(case['path'], output / 'samples' / case['path'].name)
        data = case['data']
        np.savetxt(output / 'samples' / case['path'].with_suffix('.csv').name,
                   np.column_stack([data[key] for key in ('wavelength_nm', 'reflectance', 'transmittance', 'absorptance')]),
                   delimiter=',', header='wavelength_nm,reflectance,transmittance,absorptance', comments='')
        shutil.copy2(audit_path, output / audit_path.name)
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'ranking.json', report)
    shutil.copy2(directory / 'spectra.png', output / 'spectra.png')
    atomic_json(output / 'archive_status.json', {'verified_real_cases': len(verified),
                'planned': len(manifest['cases']), 'complete_study': len(verified) == len(manifest['cases']),
                'all_native_projects_verified': True, 'all_actual_mesh_cells_below_10_nm': True,
                'note': 'Only successful, native-audited cases are archived; a partial snapshot does not complete the planned study.'})
    atomic_json(output / 'checksums.json', {str(p.relative_to(output)): checksum(p)
                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'checksums.json'})
    return {'exported': len(verified), 'planned': len(manifest['cases']), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=PROJECT_ROOT / 'data/double_dip_controls8')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'research_results/double_dip_controls8')
    args = parser.parse_args()
    print(json.dumps(export(args.directory, args.output), indent=2))
