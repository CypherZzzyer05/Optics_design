"""Real, resumable double-dip controls; dense research spectra never enter training."""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from explore_spectra import features
from fdtd_pipeline.config import PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import atomic_json, atomic_npz, checksum, directory_lock


def double_pair(wavelength, reflection):
    """Rank deep, resolved interior pairs by their weaker prominence.

    This detects two dominant troughs, not an exclusive two-mode spectrum.
    Widths are half-prominence widths, not modal linewidths or Q factors.
    """
    dips = features(wavelength, reflection)
    eligible = [d for d in dips if d['resolved'] and d['reflectance_min'] <= .2
                and d['prominence'] >= .4
                and wavelength[0] + 30 < d['wavelength_nm'] < wavelength[-1] - 30]
    pairs = []
    for a, b in itertools.combinations(eligible, 2):
        left, right = sorted((a, b), key=lambda d: d['wavelength_nm'])
        separation = right['wavelength_nm'] - left['wavelength_nm']
        if separation < 30:
            continue
        between = (wavelength > left['wavelength_nm']) & (wavelength < right['wavelength_nm'])
        ridge = float(np.max(reflection[between]))
        if ridge - max(left['reflectance_min'], right['reflectance_min']) < .2:
            continue
        pairs.append({'dips': [left, right], 'separation_nm': separation,
                      'intervening_max_reflectance': ridge,
                      'score': min(left['prominence'], right['prominence'])})
    return {'dips': dips, 'dominant_pair': max(pairs, key=lambda p: p['score']) if pairs else None}


def control_plan(values):
    mean = (values[5] + values[6]) / 2
    delta = (values[5] - values[6]) / 2
    plan = []
    # Full-difference case first to test whether the candidate survives the
    # improved As2Se3 loss fit before spending time on angular controls.
    for metal in ('Au', 'Ag'):
        for name, fraction in (('full_difference', 1), ('half_difference', .5), ('common_angle', 0)):
            angles = [mean + fraction * delta, mean - fraction * delta]
            plan.append({'id': f'{metal}_{name}', 'metal': metal,
                         'mean_angle_deg': mean, 'half_difference_deg': fraction * delta,
                         'geometry_values': list(values[:5]) + angles})
    return plan


def source_candidate(config, sample_id):
    manifest = json.loads((config.output_dir / 'manifest.json').read_text(encoding='utf-8'))
    entry = next(e for e in manifest['samples'] if e['id'] == sample_id)
    path = config.output_dir / 'samples' / f'sample_{sample_id:06d}.npz'
    if entry['state'] != 'success' or checksum(path) != entry['sha256']:
        raise ValueError('Candidate must be a verified successful real FDTD sample')
    with np.load(path, allow_pickle=False) as data:
        values = data['geometry_values'].tolist()
        original = double_pair(data['wavelength_nm'], data['reflectance'])
    if len(values) != 7:
        raise ValueError('Expected seven independent geometry parameters')
    return values, {'sample_id': sample_id, 'sha256': checksum(path),
                    'source_config_fingerprint': manifest['config_fingerprint'], **original}


def rank(output, manifest):
    rows = []
    curves = []
    for entry in manifest['cases']:
        if entry['state'] != 'success':
            continue
        path = output / 'samples' / (entry['id'] + '.npz')
        if checksum(path) != entry['sha256']:
            raise ValueError('Double-dip spectrum checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            if str(data['config_fingerprint'].item()) != manifest['config_fingerprint']:
                raise ValueError('Double-dip spectrum configuration mismatch')
            w, r = data['wavelength_nm'].copy(), data['reflectance'].copy()
        row = {**entry, **double_pair(w, r)}
        rows.append(row)
        curves.append((entry['id'], w, r))
    rows.sort(key=lambda row: -(row['dominant_pair']['score'] if row['dominant_pair'] else 0))
    report = {'successful': len(rows), 'planned': len(manifest['cases']),
              'quality_level': 'exploratory_not_mesh_converged',
              'pair_criteria': {'max_R': .2, 'minimum_prominence': .4, 'minimum_width_samples': 5,
                                'minimum_separation_nm': 30, 'band_edge_exclusion_nm': 30,
                                'minimum_intervening_contrast': .2},
              'ranking': rows, 'source_candidate': manifest['source_candidate'],
              'simulation': manifest['simulation'],
              'note': 'Two dominant dips may coexist with other spectral features. One mesh is not a mesh-convergence study.'}
    atomic_json(output / 'ranking.json', report)
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for name, w, r in curves:
            row = 0 if name.startswith('Au') else 1
            for ax in axes[row]:
                ax.plot(w, r, lw=1.1, label=name)
        for row, metal in enumerate(('Au', 'Ag')):
            for col, limits in enumerate(((700, 1700), (800, 980))):
                axes[row, col].set(xlim=limits, ylim=(-.02, 1.03), ylabel='Reflectance',
                                   xlabel='Wavelength (nm)', title=f'{metal}: angle-difference controls')
                axes[row, col].grid(alpha=.2)
                if axes[row, col].lines:
                    axes[row, col].legend(fontsize=8)
        fig.suptitle('As2Se3 / 350 nm SiO2 / 150 nm metal | exploratory real FDTD')
        fig.tight_layout()
        fig.savefig(output / 'spectra.png', dpi=180)
        plt.close(fig)
    return report


def run(config, output, sample_id=3, mesh_nm=20, points=3000, case_id=None, limit=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    values, source = source_candidate(config, sample_id)
    plan = control_plan(values)
    if case_id:
        plan = [entry for entry in plan if entry['id'] == case_id]
        if not plan:
            raise ValueError('Unknown control case')
    base = replace(config.simulation, research_grid=True, spectrum_points=points, mesh_nm=mesh_nm,
                   mesh_accuracy=4 if mesh_nm < 20 else 3,
                   metal_mesh_nm=2.5 if mesh_nm < 20 else 5,
                   simulation_time_ps=4 if mesh_nm < 20 else 2,
                   material_fit_tolerance=0, material_imaginary_weight=2)
    metadata = {'source_candidate': source, 'simulation': asdict(base), 'cases': plan,
                'material_csv_sha256': checksum(Path(base.resonator_nk_csv))}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    with directory_lock(output):
        path = output / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text(encoding='utf-8'))
            if manifest['config_fingerprint'] != fingerprint:
                raise ValueError('Configuration changed; choose a separate output directory')
        else:
            manifest = {**metadata, 'config_fingerprint': fingerprint}
            for entry in manifest['cases']:
                entry['state'] = 'pending'
            atomic_json(path, manifest)
        completed = 0
        for metal in ('Au', 'Ag'):
            pending = [entry for entry in manifest['cases'] if entry['metal'] == metal and entry['state'] != 'success']
            if not pending or (limit is not None and completed >= limit):
                continue
            settings = replace(base, metal_material='Au (Gold) - Palik' if metal == 'Au' else 'Ag (Silver) - Palik (0-2um)')
            with FDTDSimulator(settings, config.lumapi_path) as simulator:
                for entry in pending:
                    if limit is not None and completed >= limit:
                        break
                    print(f"Solving {entry['id']}; mesh={mesh_nm} nm; spectral points={points}", flush=True)
                    entry.update(state='running', metal_material=settings.metal_material)
                    atomic_json(path, manifest)
                    try:
                        result = simulator.run(config.make_geometry(entry['geometry_values']),
                                               output / 'projects' / (entry['id'] + '.fsp'))
                        sample = output / 'samples' / (entry['id'] + '.npz')
                        sample.parent.mkdir(exist_ok=True)
                        atomic_npz(sample, **result, geometry_values=np.array(entry['geometry_values']),
                                   config_fingerprint=np.array(fingerprint))
                        entry.update(state='success', sha256=checksum(sample), status=int(result['status']))
                        entry.pop('error', None)
                    except Exception as error:
                        entry.update(state='failed', error=str(error))
                        raise
                    finally:
                        atomic_json(path, manifest)
                        rank(output, manifest)
                    completed += 1
        return rank(output, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'rank'))
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/as2se3_au_pilot.json')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/double_dip_controls')
    parser.add_argument('--sample-id', type=int, default=3)
    parser.add_argument('--mesh-nm', type=float, default=20)
    parser.add_argument('--points', type=int, default=3000)
    parser.add_argument('--case-id')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'plan':
        values, source = source_candidate(config, args.sample_id)
        print(json.dumps({'source': source, 'cases': control_plan(values)}, indent=2))
    elif args.command == 'rank':
        print(json.dumps(rank(args.output, json.loads((args.output / 'manifest.json').read_text(encoding='utf-8'))), indent=2))
    else:
        result = run(config, args.output.resolve(), args.sample_id, args.mesh_nm, args.points, args.case_id, args.limit)
        print(f"Solved {result['successful']}/{result['planned']}")
