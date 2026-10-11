"""Track a fixed pair of troughs through six native-audited Au/Ag angle controls."""
import argparse
import csv
from dataclasses import asdict
import itertools
import json
from pathlib import Path

import numpy as np

from compare_double_mesh import read_case
from explore_double_dips import double_pair
from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json, checksum


def track_pair(wavelength, reflection, reference_dips, maximum_shift_nm=15):
    found = double_pair(wavelength, reflection)
    if len(found['dips']) < 2:
        return {'tracked_dips': [], 'retained': False, 'dominant_pair': found['dominant_pair']}
    matched = min(itertools.permutations(found['dips'], 2),
                  key=lambda pair: sum(abs(a['wavelength_nm']-b['wavelength_nm'])
                                       for a, b in zip(reference_dips, pair)))
    if any(abs(a['wavelength_nm']-b['wavelength_nm']) > maximum_shift_nm
           for a, b in zip(reference_dips, matched)):
        return {'tracked_dips': [], 'retained': False, 'dominant_pair': found['dominant_pair']}
    left, right = matched
    between = (wavelength > left['wavelength_nm']) & (wavelength < right['wavelength_nm'])
    ridge = float(np.max(reflection[between])) if np.any(between) else 0.
    retained = bool(right['wavelength_nm']-left['wavelength_nm'] >= 30
                    and ridge-max(d['reflectance_min'] for d in matched) >= .2
                    and all(d['resolved'] and d['reflectance_min'] <= .2 and d['prominence'] >= .4
                            and wavelength[0]+30 < d['wavelength_nm'] < wavelength[-1]-30 for d in matched))
    return {'tracked_dips': list(matched), 'retained': retained,
            'intervening_max_reflectance': ridge, 'dominant_pair': found['dominant_pair']}


def analyze(directory, output):
    ids = [f'{metal}_{angle}' for metal in ('Au', 'Ag')
           for angle in ('full_difference', 'half_difference', 'common_angle')]
    # Read and verify all cases before generating any completed-study finding.
    cases = [read_case(directory, case_id) for case_id in ids]
    reference = cases[0]
    reference_pair = double_pair(reference['data']['wavelength_nm'], reference['data']['reflectance'])['dominant_pair']
    if not reference_pair:
        raise ValueError('The reference must contain the resolved pair being tracked')
    reference_settings = asdict(reference['settings'])
    reference_settings.pop('metal_material')
    rows = []
    for case in cases:
        entry, data = case['entry'], case['data']
        settings = asdict(case['settings'])
        settings.pop('metal_material')
        if settings != reference_settings or entry['geometry_values'][:5] != reference['entry']['geometry_values'][:5]:
            raise ValueError('Angle controls must share dimensions, mesh and other physical settings')
        angles = entry['geometry_values'][5:]
        if not np.isclose(np.mean(angles), reference['entry']['mean_angle_deg'], atol=1e-10):
            raise ValueError('Mean angle changed between the angle controls')
        if case['manifest']['source_candidate'] != reference['manifest']['source_candidate']:
            raise ValueError('Controls must share the same source candidate')
        if not np.array_equal(data['wavelength_nm'], reference['data']['wavelength_nm']):
            raise ValueError('Spectral grids differ between angle controls')
        native = directory/(entry['id']+'_native_audit.json')
        audit = json.loads(native.read_text(encoding='utf-8'))
        if (audit['sample_sha256'] != checksum(case['path'])
                or audit['project_sha256'] != checksum(directory/'projects'/(entry['id']+'.fsp'))
                or audit['status'] != 2 or not audit['native_geometry_verified']
                or not audit['native_spectra_match_export'] or not audit['all_actual_mesh_cells_below_10_nm']):
            raise ValueError('Native audit is missing or does not verify the current control')
        rows.append({'id': entry['id'], 'metal': entry['metal'],
                     'angle_difference_fraction': entry['half_difference_deg']/reference['entry']['half_difference_deg'],
                     'odd_angle_deg': angles[0], 'even_angle_deg': angles[1],
                     'sample_sha256': checksum(case['path']), 'native_audit_sha256': checksum(native),
                     'actual_mesh': case['actual_mesh'],
                     **track_pair(data['wavelength_nm'], data['reflectance'], reference_pair['dips'])})
    conclusions = {}
    for metal in ('Au', 'Ag'):
        selected = sorted((row for row in rows if row['metal'] == metal), key=lambda row: row['angle_difference_fraction'])
        retained = [row for row in selected if row['retained']]
        conclusions[metal] = {'equal_angles_retain_reference_double_dip': selected[0]['retained'],
                             'retained_control_ids': [row['id'] for row in retained],
                             'deepest_joint_pair_case': min(retained, key=lambda row: max(d['reflectance_min']
                                                       for d in row['tracked_dips']))['id'] if retained else None,
                             'tracked_centers_range_nm': [[min(row['tracked_dips'][i]['wavelength_nm'] for row in selected),
                                                          max(row['tracked_dips'][i]['wavelength_nm'] for row in selected)]
                                                         for i in range(2)] if all(row['tracked_dips'] for row in selected) else None}
    report = {'verified_controls': len(rows), 'reference_case': ids[0], 'reference_pair': reference_pair,
              'fixed_dimensions_nm': reference['entry']['geometry_values'][:5],
              'mean_angle_deg': reference['entry']['mean_angle_deg'], 'rows': rows, 'findings': conclusions,
              'matching_maximum_shift_nm': 15, 'pair_criteria_source': 'explore_double_dips.double_pair',
              'note': 'Wavelength matching tracks spectral features, not proven modal identity. Findings concern three tested angle settings at fixed dimensions and mean angle; they are not universal rotation laws or material-accuracy claims.'}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'angle_controls.json', report)
    table = []
    for row in rows:
        flat = {key: row[key] for key in ('id', 'metal', 'angle_difference_fraction', 'odd_angle_deg', 'even_angle_deg', 'retained')}
        for i in range(2):
            for source, label in (('wavelength_nm', 'center_nm'), ('reflectance_min', 'R_min'), ('half_prominence_width_nm', 'half_prominence_width_nm')):
                flat[f'dip_{i+1}_{label}'] = row['tracked_dips'][i][source] if row['tracked_dips'] else None
        table.append(flat)
    with (output/'angle_controls.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, metal in enumerate(('Au', 'Ag')):
        selected = sorted((row for row in rows if row['metal'] == metal and row['tracked_dips']),
                          key=lambda row: row['angle_difference_fraction'])
        x = [row['angle_difference_fraction'] for row in selected]
        for j in range(2):
            axes[i, j].plot(x, [row['tracked_dips'][j]['wavelength_nm'] for row in selected], 'o-', color='#28558b')
            axes[i, j].set(title=f'{metal}: tracked trough {j+1}', ylabel='Center wavelength (nm)')
        for j, color in enumerate(('#28558b', '#c64232')):
            axes[i, 2].plot(x, [row['tracked_dips'][j]['reflectance_min'] for row in selected], 'o-',
                           color=color, label=f'Trough {j+1}')
        axes[i, 2].set(title=f'{metal}: trough depth', ylabel='Minimum reflectance')
        axes[i, 2].legend()
        for ax in axes[i]:
            ax.set(xlabel='Fraction of initial odd/even angle difference', xticks=[0, .5, 1])
            ax.grid(alpha=.2)
    fig.suptitle('Six real FDTD controls | fixed dimensions and mean angle | guides join three tested settings')
    fig.tight_layout()
    fig.savefig(output/'angle_controls.png', dpi=180)
    plt.close(fig)
    atomic_json(output/'checksums.json', {p.name: checksum(p) for p in output.iterdir()
                                        if p.is_file() and p.name != 'checksums.json'})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=PROJECT_ROOT/'data/double_dip_controls8')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'research_results/angle_controls8')
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, args.output), indent=2))
