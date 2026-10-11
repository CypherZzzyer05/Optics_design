"""Paired Au/Ag quadrant sweeps using real FDTD, with resumable spectral ranking."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from fdtd_pipeline.config import PROJECT_ROOT, Geometry, load_config
from fdtd_pipeline.lumerical import FDTDSimulator, validate_spectrum
from fdtd_pipeline.pipeline import atomic_json, atomic_npz, checksum, directory_lock


def features(wavelength, reflectance, minimum_prominence=.10):
    wavelength, reflectance = np.asarray(wavelength), np.asarray(reflectance)
    if (wavelength.ndim != 1 or len(wavelength) < 3 or wavelength.shape != reflectance.shape
            or not np.isfinite(wavelength).all() or not np.isfinite(reflectance).all()
            or np.any(np.diff(wavelength) <= 0)):
        raise ValueError("Features require finite increasing wavelength and matching reflection")
    indices, properties = find_peaks(-reflectance, prominence=minimum_prominence)
    widths, _, left, right = peak_widths(-reflectance, indices, rel_height=.5)
    axis = np.arange(len(wavelength))
    result = []
    for i, index in enumerate(indices):
        width_nm = float(np.interp(right[i], axis, wavelength)-np.interp(left[i], axis, wavelength))
        result.append({"wavelength_nm": float(wavelength[index]), "reflectance_min": float(reflectance[index]),
                       "prominence": float(properties['prominences'][i]),
                       "half_prominence_width_nm": width_nm,
                       "width_samples": float(widths[i]), "resolved": bool(widths[i] >= 5),
                       "lambda_over_width": float(wavelength[index]/width_nm)})
    return sorted(result, key=lambda item: item['prominence'], reverse=True)


def study_plan():
    # Keep centers fixed while changing common angle and angle difference.
    cases = [("bare_stack", 0, 0, False), ("aligned", 0, 0, True),
             ("common_15", 15, 15, True), ("opposite_10", 10, -10, True),
             ("opposite_20", 20, -20, True), ("odd_only_20", 20, 0, True),
             ("mirror_opposite_20", -20, 20, True)]
    result = []
    for metal in ('Au', 'Ag'):
        for name, odd, even, include in cases:
            g = Geometry(660, 190, 300, 120, 120, theta_odd_deg=odd, theta_even_deg=even)
            result.append({'id': f'{metal}_{name}', 'metal': metal, 'case': name,
                           'include_resonators': include, 'geometry_values': g.array(include_quadrant=True).tolist()})
    return result


def rank_results(output, manifest, target_csv=None):
    rows, plotted = [], []
    target = np.loadtxt(target_csv, delimiter=',', skiprows=1) if target_csv else None
    if target is not None and (target.ndim != 2 or target.shape[1] != 2 or not np.isfinite(target).all()
                               or np.any(np.diff(target[:, 0]) <= 0) or np.any((target[:, 1] < 0) | (target[:, 1] > 1))):
        raise ValueError('Target CSV needs increasing wavelength_nm,reflectance columns in [0,1]')
    for entry in manifest['cases']:
        if entry['state'] != 'success':
            continue
        path = output / 'samples' / (entry['id']+'.npz')
        if checksum(path) != entry['sha256']:
            raise ValueError(f"Study sample checksum mismatch: {entry['id']}")
        with np.load(path, allow_pickle=False) as data:
            if str(data['config_fingerprint'].item()) != manifest['config_fingerprint']:
                raise ValueError('Study sample provenance mismatch')
            wavelength, reflection = data['wavelength_nm'].copy(), data['reflectance'].copy()
            absorption = data['absorptance'].copy()
        dips = features(wavelength, reflection)
        resolved = [d for d in dips if d['resolved']]
        row = {**entry, 'dips': dips, 'resolved_dip_count': len(resolved),
               'score': resolved[0]['prominence'] if resolved else 0.0,
               'max_absorptance': float(absorption.max())}
        if target is not None:
            if target[0, 0] > wavelength[0] or target[-1, 0] < wavelength[-1]:
                raise ValueError('Target must cover the complete simulation wavelength band')
            row['target_rmse'] = float(np.sqrt(np.mean((reflection-np.interp(wavelength, target[:, 0], target[:, 1]))**2)))
        rows.append(row)
        plotted.append((entry['id'], wavelength, reflection))
    rows.sort(key=(lambda row: row['target_rmse']) if target is not None else (lambda row: -row['score']))
    comparisons = []
    curves = {name: reflection for name, _, reflection in plotted}
    for metal in ('Au', 'Ag'):
        baseline = curves.get(metal+'_bare_stack')
        aligned = curves.get(metal+'_aligned')
        if baseline is not None and aligned is not None:
            comparisons.append({'metal': metal, 'comparison': 'tetramer_vs_bare_stack',
                                'rms_reflection_change': float(np.sqrt(np.mean((aligned-baseline)**2)))})
        plus, minus = curves.get(metal+'_opposite_20'), curves.get(metal+'_mirror_opposite_20')
        if plus is not None and minus is not None:
            # For equal x/y gaps, swapping the two opposite angles is a global
            # 90-degree rotation (square orientations are modulo 90 degrees).
            # It compares x/y-polarization response, rather than a mirror pair.
            comparisons.append({'metal': metal, 'comparison': 'quadrant_swap_90deg_rotation',
                                'max_reflection_difference': float(np.max(np.abs(plus-minus)))})
    settings = manifest['config']['simulation']
    report = {'band_nm': [settings['wavelength_start_nm'], settings['wavelength_stop_nm']], 'quality_level': 'exploratory_not_mesh_converged',
              'ranking_metric': 'target_rmse' if target is not None else 'resolved reflection-dip prominence',
              'material_extension_nm': 12, 'comparisons': comparisons, 'ranking': rows,
              'successful': len(rows), 'planned': len(manifest['cases'])}
    atomic_json(output / 'ranking.json', report)
    if plotted:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for name, wavelength, reflection in plotted:
            display_name = name.replace('mirror_opposite_20', 'swapped_opposite_20')
            axes[0 if name.startswith('Au') else 1].plot(wavelength, reflection, label=display_name, lw=1.1)
        for ax, metal in zip(axes, ('Au', 'Ag')):
            ax.set(ylabel='Reflectance', title=f'As2Se3 tetramers / SiO2 / {metal} | exploratory')
            ax.axvspan(1688, 1700, color='orange', alpha=.25, label='nk endpoint extension')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, ncol=2)
        axes[-1].set_xlabel('Wavelength (nm)')
        fig.tight_layout()
        fig.savefig(output / 'spectra.png', dpi=160)
        plt.close(fig)
    return report


def run(config, output, limit=None, target_csv=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    with directory_lock(output):
        plan = study_plan()
        metadata = {'config': config.metadata(), 'cases': plan}
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['config_fingerprint'] != fingerprint:
                raise ValueError('Study configuration changed; choose a new output directory')
        else:
            manifest = {'config_fingerprint': fingerprint, **metadata}
            for entry in manifest['cases']:
                entry['state'] = 'pending'
            atomic_json(manifest_path, manifest)
        completed = 0
        for metal in ('Au', 'Ag'):
            pending = []
            for entry in manifest['cases']:
                if entry['metal'] != metal:
                    continue
                sample_path = output / 'samples' / (entry['id']+'.npz')
                if entry['state'] == 'success' and sample_path.is_file() and checksum(sample_path) == entry.get('sha256'):
                    continue
                pending.append(entry)
            if not pending or (limit is not None and completed >= limit):
                continue
            settings = replace(config.simulation, metal_material='Au (Gold) - Palik' if metal=='Au' else 'Ag (Silver) - Palik (0-2um)')
            with FDTDSimulator(settings, config.lumapi_path) as simulator:
                for entry in pending:
                    if limit is not None and completed >= limit:
                        break
                    print(f"Solving {entry['id']}: {entry['geometry_values']}", flush=True)
                    entry['metal_material'] = settings.metal_material
                    entry['state'] = 'running'
                    atomic_json(manifest_path, manifest)
                    geometry = config.make_geometry(entry['geometry_values'])
                    try:
                        result = simulator.run(geometry, output/'projects'/(entry['id']+'.fsp'),
                                               include_silicon=entry['include_resonators'])
                        validate_spectrum(result['wavelength_nm'], result['reflectance'], result['transmittance'], settings, result['status'])
                        path = output/'samples'/(entry['id']+'.npz')
                        path.parent.mkdir(exist_ok=True)
                        atomic_npz(path, **result, geometry_values=geometry.array(include_quadrant=True),
                                   config_fingerprint=np.array(fingerprint))
                        entry.update(state='success', sha256=checksum(path), status=int(result['status']))
                        entry.pop('error', None)
                    except Exception as error:
                        entry.update(state='failed', error=str(error))
                        raise
                    finally:
                        atomic_json(manifest_path, manifest)
                        rank_results(output, manifest, target_csv)
                    completed += 1
        return rank_results(output, manifest, target_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'rank'))
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/as2se3_au_pilot.json')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'data/near_ir_study')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--target-csv', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'plan':
        print(json.dumps(study_plan(), indent=2))
    elif args.command == 'rank':
        print(json.dumps(rank_results(args.output, json.loads((args.output/'manifest.json').read_text(encoding='utf-8')), args.target_csv), indent=2))
    else:
        report = run(config, args.output.resolve(), args.limit, args.target_csv)
        print(f"Solved {report['successful']}/{report['planned']}; report: {args.output/'ranking.json'}")
