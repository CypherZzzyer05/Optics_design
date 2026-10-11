"""Audit real seven-parameter samples and a completed pilot training run."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fdtd_pipeline.config import PROJECT_ROOT, QUADRANT_PARAMETER_NAMES, load_config
from fdtd_pipeline.lumerical import FDTDSimulator, mesh_grid_summary
from fdtd_pipeline.pipeline import atomic_json, checksum, load_sample, prepare_manifest
from metaformer import InverseMetaformer
from train_metaformer import load_training_config, load_training_data


def inspect_native(config, sample_id=0):
    manifest = json.loads((config.output_dir / 'manifest.json').read_text(encoding='utf-8'))
    entry = next(e for e in manifest['samples'] if e['id'] == sample_id)
    if entry['state'] != 'success':
        raise ValueError('Native inspection requires a successful real sample')
    geometry = config.make_geometry(entry['geometry_values'])
    project = config.output_dir / 'projects' / f'sample_{sample_id:06d}.fsp'
    objects = []
    with FDTDSimulator(config.simulation, config.lumapi_path) as simulator:
        f = simulator.fdtd
        f.load(str(project))
        for i, (center, angle) in enumerate(zip(geometry.centers(), geometry.angles())):
            name = f'si_{i}'
            xy = [float(f.getnamed(name, axis)) * 1e9 for axis in ('x', 'y')]
            rotation = float(f.getnamed(name, 'rotation 1'))
            material = str(f.getnamed(name, 'material'))
            axis = str(f.getnamed(name, 'first axis'))
            if not np.allclose(xy, center, atol=1e-6) or not np.isclose(rotation, -angle):
                raise ValueError('Native geometry center / angle differs from requested values')
            if material != config.simulation.resonator_material or axis != 'z':
                raise ValueError('Native pillar material / rotation axis differs')
            objects.append({'name': name, 'center_nm': xy, 'project_angle_deg': angle,
                            'native_rotation_deg': rotation, 'material': material, 'axis': axis})
        substrate = {key: float(f.getnamed('substrate', key)) * 1e9 for key in ('z min', 'z max')}
        metal = {key: float(f.getnamed('back_metal', key)) * 1e9 for key in ('z min', 'z max')}
        metal['material'] = str(f.getnamed('back_metal', 'material'))
        if not np.isclose(metal['z max'], substrate['z min']) or metal['z min'] >= metal['z max']:
            raise ValueError('Metal is not directly below the finite substrate')
        if metal['material'] != config.simulation.metal_material:
            raise ValueError('Native metal material differs')
    report = {'sample_id': sample_id, 'project_sha256': checksum(project), 'objects': objects,
              'substrate': substrate, 'metal': metal, 'native_geometry_verified': True}
    atomic_json(config.output_dir / 'native_objects.json', report)
    return report


def audit(sampling, training, run_dir):
    manifest = prepare_manifest(sampling)
    successful = [e for e in manifest['samples'] if e['state'] == 'success']
    if len(successful) != sampling.sample_count:
        raise ValueError('Pilot sample plan is incomplete')
    datasets = load_training_data(training)
    actual_meshes = {}
    if datasets['train'].parameter_names != QUADRANT_PARAMETER_NAMES:
        raise ValueError('Expected seven-parameter data')
    with np.load(training.dataset, allow_pickle=False) as data:
        for entry in successful:
            sample = load_sample(sampling.output_dir / 'samples' / f"sample_{entry['id']:06d}.npz",
                                 entry, sampling, manifest['config_fingerprint'])
            if sampling.simulation.maximum_mesh_nm is not None:
                path = sampling.output_dir/'samples'/f"sample_{entry['id']:06d}.npz"
                with np.load(path, allow_pickle=False) as original:
                    actual_meshes[str(entry['id'])] = mesh_grid_summary(
                        {axis: original[f'mesh_{axis}_nm']*1e-9 for axis in ('x', 'y', 'z')},
                        sampling.simulation.maximum_mesh_nm)
            row = int(np.flatnonzero(data['sample_ids'] == entry['id'])[0])
            if not np.array_equal(data['geometry_values'][row], sample['geometry_values']):
                raise ValueError('Packed geometry differs from its source sample')
            if not np.allclose(data['spectra'][row], sample['reflectance'], atol=1e-7, rtol=0):
                raise ValueError('Packed spectrum differs from its source sample')
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    status = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'completed' or metrics['optimizer_steps'] < 1:
        raise ValueError('Training has not completed actual optimizer updates')
    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['metadata']['dataset_sha256'] != checksum(training.dataset):
        raise ValueError('Selected checkpoint belongs to another dataset')
    torch.manual_seed(training.seed)
    model = InverseMetaformer(training.model).eval()
    initial_rows = model.regressor[2].weight.detach().clone()
    model.load_state_dict(checkpoint['model_state'])
    changes = torch.linalg.vector_norm(model.regressor[2].weight.detach() - initial_rows, dim=1).numpy()
    if len(changes) != 7 or np.any(changes <= 0):
        raise ValueError('Not all seven output rows changed from initialization')
    torch.set_num_threads(4)
    differences = {}
    for split, dataset in datasets.items():
        with torch.inference_mode():
            predictions = model(dataset.spectra).numpy()
        csv = np.atleast_1d(np.genfromtxt(run_dir / f'{split}_predictions.csv', delimiter=',', names=True))
        expected = np.column_stack([csv[f'pred_01_{name}'] for name in QUADRANT_PARAMETER_NAMES])
        if not np.array_equal(csv['sample_id'].astype(int), dataset.sample_ids):
            raise ValueError('CSV sample IDs differ from the stored split')
        error = float(np.max(np.abs(predictions - expected)))
        if error > 2e-5:
            raise ValueError('Reloaded checkpoint predictions differ from saved CSV')
        differences[split] = error
    report = {'successful_real_samples': len(successful), 'statuses': [e['status'] for e in successful],
              'spectra_shape': [len(successful), training.model.spectrum_points], 'geometry_shape': [len(successful), 7],
              'parameter_names': list(QUADRANT_PARAMETER_NAMES), 'parameter_units': ['nm'] * 5 + ['deg'] * 2,
              'split_sizes': {s: len(d) for s, d in datasets.items()}, 'dataset_sha256': checksum(training.dataset),
              'best_checkpoint_sha256': checksum(run_dir / 'best.pt'), 'epochs_completed': metrics['epochs_completed'],
              'optimizer_steps': metrics['optimizer_steps'], 'seven_output_row_weight_changes': changes.tolist(),
              'reloaded_prediction_max_abs_errors': differences, 'metrics': metrics['metrics'],
              'train_mean_baseline': metrics['train_mean_baseline'],
              'actual_mesh_by_sample': actual_meshes,
              'all_real_sample_meshes_below_10_nm': bool(actual_meshes and len(actual_meshes)==len(successful)),
              'note': 'Real-data pipeline check only; small pilot does not establish inverse-design accuracy or mesh convergence.'}
    atomic_json(run_dir / 'validation_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampling-config', type=Path, default=PROJECT_ROOT / 'configs/as2se3_au_pilot.json')
    parser.add_argument('--training-config', type=Path, default=PROJECT_ROOT / 'configs/train_as2se3_pilot.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    sampling = load_config(args.sampling_config)
    if args.native_only:
        report = inspect_native(sampling)
    else:
        training = load_training_config(args.training_config)
        run_dir = args.run_dir
        if run_dir is None:
            run_dir = Path(json.loads((training.output_dir / 'latest_run.json').read_text(encoding='utf-8'))['run_dir'])
        report = audit(sampling, training, run_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
