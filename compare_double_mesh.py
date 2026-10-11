"""Compare both tracked troughs on two verified, strictly sub-10 nm meshes."""
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from explore_double_dips import double_pair
from fdtd_pipeline.config import PROJECT_ROOT, SimulationSettings
from fdtd_pipeline.lumerical import mesh_grid_summary, validate_spectrum, validate_power_integration
from fdtd_pipeline.pipeline import atomic_json, checksum
from focused_spectra import common_spectra


def compare_curves(wavelength, reference, fine):
    reference_features = double_pair(wavelength, reference)
    fine_features = double_pair(wavelength, fine)
    pair = reference_features['dominant_pair']
    matched = []
    if pair and len(fine_features['dips']) >= 2:
        # Track the reference pair by wavelength with two distinct matches;
        # prominence ranking can change when the mesh changes.
        assignment = min(itertools.permutations(fine_features['dips'], 2),
                         key=lambda dips: sum(abs(a['wavelength_nm']-b['wavelength_nm'])
                                              for a, b in zip(pair['dips'], dips)))
        for old, new in zip(pair['dips'], assignment):
            shift = abs(new['wavelength_nm']-old['wavelength_nm'])
            depth = abs(new['reflectance_min']-old['reflectance_min'])
            width = abs(new['half_prominence_width_nm']/old['half_prominence_width_nm']-1)
            preserved = bool(new['resolved'] and new['reflectance_min'] <= .2 and new['prominence'] >= .4)
            matched.append({'reference_dip': old, 'fine_dip': new, 'shift_nm': shift,
                            'absolute_R_min_change': depth, 'relative_width_change': width,
                            'feature_preserved': preserved,
                            'passes_thresholds': bool(preserved and shift <= 2 and depth <= .03 and width <= .1)})
    return {'reference_pair': pair, 'fine_pair': fine_features['dominant_pair'],
            'matched_dips': matched,
            'both_dips_stable': bool(len(matched) == 2 and all(d['passes_thresholds'] for d in matched)
                                     and fine_features['dominant_pair'] is not None),
            'thresholds': {'peak_shift_nm': 2, 'absolute_R_min_change': .03, 'relative_width_change': .1},
            'maximum_R_difference': float(np.max(np.abs(np.asarray(reference)-fine))),
            'rms_R_difference': float(np.sqrt(np.mean((np.asarray(reference)-fine)**2)))}


def read_case(directory, case_id):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    entry = next(e for e in manifest['cases'] if e['id'] == case_id)
    path = directory / 'samples' / (case_id + '.npz')
    if entry['state'] != 'success' or checksum(path) != entry['sha256']:
        raise ValueError('Comparison requires successful samples with verified checksums')
    settings = replace(SimulationSettings(**manifest['simulation']), metal_material=entry['metal_material'])
    if settings.maximum_mesh_nm is None:
        raise ValueError('Formal mesh comparison requires whole-domain maximum mesh and actual grid evidence')
    with np.load(path, allow_pickle=False) as sample:
        data = {key: sample[key].copy() for key in sample.files}
    if str(data['config_fingerprint'].item()) != manifest['config_fingerprint']:
        raise ValueError('Sample belongs to another study configuration')
    if not np.array_equal(data['geometry_values'], entry['geometry_values']):
        raise ValueError('Sample geometry does not match the study case')
    validate_spectrum(data['wavelength_nm'], data['reflectance'], data['transmittance'], settings, int(data['status']))
    mesh = mesh_grid_summary({axis: data[f'mesh_{axis}_nm']*1e-9 for axis in ('x', 'y', 'z')},
                             settings.maximum_mesh_nm)
    recording_bound = max(mesh[axis]['maximum_step_nm'] for axis in ('x', 'y'))*settings.monitor_downsample_xy
    if recording_bound >= 10:
        raise ValueError('Actual monitor recording spacing reaches the strict 10 nm limit')
    integration = None
    if settings.monitor_downsample_xy > 1:
        integration, check_w, _, _ = validate_power_integration(data['wavelength_nm'], data['reflectance'], data['transmittance'],
            data['power_check_wavelength_nm'], data['power_check_reflectance'], data['power_check_transmittance'])
        if not np.allclose(check_w, settings.power_validation_wavelengths_nm, rtol=0, atol=1e-5):
            raise ValueError('Full-spatial checks differ from the specified validation wavelengths')
    return {'directory': directory, 'path': path, 'manifest': manifest, 'entry': entry,
            'settings': settings, 'data': data, 'actual_mesh': mesh,
            'power_integration_validation': integration, 'monitor_recording_spacing_bound_nm': recording_bound}


def compare(reference_dir, fine_dir, case_id, output, allow_focused=False):
    reference, fine = read_case(reference_dir, case_id), read_case(fine_dir, case_id)
    if not 0 < fine['settings'].maximum_mesh_nm < reference['settings'].maximum_mesh_nm < 10:
        raise ValueError('Require two decreasing meshes, both strictly below 10 nm')
    if reference['entry']['geometry_values'] != fine['entry']['geometry_values']:
        raise ValueError('Cannot attribute different geometries to mesh convergence')
    if reference['manifest']['source_candidate']['sha256'] != fine['manifest']['source_candidate']['sha256']:
        raise ValueError('Mesh studies must start from the same source candidate')
    differences = [key for key, value in vars(reference['settings']).items()
                   if key not in ('mesh_nm', 'maximum_mesh_nm', 'metal_mesh_nm', 'mesh_accuracy',
                                  'processes', 'threads', 'hide_fdtd', 'save_projects')
                   and key not in ('monitor_downsample_xy', 'power_validation_wavelengths_nm')
                   and not (allow_focused and key in ('spectrum_points', 'research_wavelengths_nm'))
                   and value != getattr(fine['settings'], key)]
    if differences:
        raise ValueError(f'Physical or spectral settings changed: {differences}')
    shared_w, shared_r, fine_r, sampling = common_spectra(reference['data']['wavelength_nm'], reference['data']['reflectance'],
        fine['data']['wavelength_nm'], fine['data']['reflectance'], allow_focused)
    report = compare_curves(shared_w, shared_r, fine_r)
    report.update(case_id=case_id, geometry_values=reference['entry']['geometry_values'],
                  reference_mesh_nm=reference['settings'].maximum_mesh_nm,
                  fine_mesh_nm=fine['settings'].maximum_mesh_nm,
                  reference_sha256=checksum(reference['path']), fine_sha256=checksum(fine['path']),
                  reference_actual_mesh=reference['actual_mesh'], fine_actual_mesh=fine['actual_mesh'],
                  reference_power_integration_check=reference['power_integration_validation'],
                  fine_power_integration_check=fine['power_integration_validation'],
                  monitor_recording_spacing_bounds_nm=[reference['monitor_recording_spacing_bound_nm'], fine['monitor_recording_spacing_bound_nm']],
                  statuses=[int(reference['data']['status']), int(fine['data']['status'])],
                  comparison_kind='Two-mesh stability check for both dominant troughs',
                  spectral_sampling_comparison=sampling,
                  material_fit_validated=False,
                  note='Agreement at these thresholds does not establish material accuracy, modal identity, or infinite-resolution convergence.')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'comparison.json', report)
    for name, case in (('reference', reference), ('fine', fine)):
        shutil.copy2(case['path'], output / (name + '.npz'))
        atomic_json(output / (name + '_manifest.json'), case['manifest'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, limits in zip(axes, ((700, 1700), (800, 980))):
        for label, case in (('Reference', reference), ('Fine', fine)):
            suffix = f"; focused {case['settings'].spectrum_points} points" if case['settings'].research_wavelengths_nm else ''
            ax.plot(case['data']['wavelength_nm'], case['data']['reflectance'], lw=1.2,
                    label=f"{label}: maximum {case['settings'].maximum_mesh_nm:g} nm mesh{suffix}")
        ax.set(xlim=limits, ylim=(-.02, 1.03), xlabel='Wavelength (nm)', ylabel='Reflectance')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f'{case_id}: both troughs | real FDTD mesh comparison')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=PROJECT_ROOT / 'data/double_dip_controls8')
    parser.add_argument('--fine', type=Path, default=PROJECT_ROOT / 'data/double_dip_fine5_checked')
    parser.add_argument('--case-id', default='Au_full_difference')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'research_results/double_dip_mesh_comparison')
    parser.add_argument('--allow-focused', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.fine, args.case_id, args.output, args.allow_focused), indent=2))
