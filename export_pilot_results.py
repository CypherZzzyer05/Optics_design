"""Export compact real pilot data, training evidence, and a candidate figure."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from explore_double_dips import double_pair
from fdtd_pipeline.config import PROJECT_ROOT, load_config
from fdtd_pipeline.pipeline import atomic_json, checksum
from train_metaformer import load_training_config
from audit_quadrant_pilot import audit


def export(sampling_config=PROJECT_ROOT/'configs/as2se3_au_pilot.json',
           training_config=PROJECT_ROOT/'configs/train_as2se3_pilot.json',
           output=PROJECT_ROOT/'research_results/seven_parameter_pilot'):
    config = load_config(sampling_config)
    training_settings = load_training_config(training_config)
    source = config.output_dir
    run_dir = Path(json.loads((training_settings.output_dir/'latest_run.json').read_text(encoding='utf-8'))['run_dir'])
    # Revalidate source meshes, packed rows, checkpoint and predictions before copying.
    audit(config, training_settings, run_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for entry in manifest['samples']:
        path = source / 'samples' / f"sample_{entry['id']:06d}.npz"
        if entry['state'] != 'success' or checksum(path) != entry['sha256']:
            raise ValueError('Export requires all real samples with verified checksums')
        with np.load(path, allow_pickle=False) as data:
            rows.append({'sample_id': entry['id'], 'geometry_values': data['geometry_values'].tolist(),
                         'sha256': checksum(path), 'status': int(data['status']),
                         **double_pair(data['wavelength_nm'], data['reflectance'])})
        (output / 'samples').mkdir(exist_ok=True)
        for extension in ('.npz', '.csv'):
            shutil.copy2(path.with_suffix(extension), output / 'samples' / path.with_suffix(extension).name)
    for name in ('manifest.json', 'dataset.npz', 'native_objects.json'):
        shutil.copy2(source / name, output / name)
    training = output / 'training'
    training.mkdir(exist_ok=True)
    for name in ('history.json', 'metrics.json', 'run.json', 'status.json', 'validation_report.json',
                 'train_predictions.csv', 'val_predictions.csv', 'test_predictions.csv', 'training_preview.png'):
        shutil.copy2(run_dir / name, training / name)
    rows.sort(key=lambda row: -(row['dominant_pair']['score'] if row['dominant_pair'] else 0))
    atomic_json(output / 'double_dip_ranking.json', {'criteria_source': 'explore_double_dips.double_pair',
                'successful': len(rows), 'ranking': rows,
                'simulation': config.metadata()['simulation'],
                'note': 'Real pilot spectra; two main dips do not imply only two features or inverse-model accuracy.'})
    candidate = rows[0]
    geometry = config.make_geometry(candidate['geometry_values'])
    with np.load(output / 'samples' / f"sample_{candidate['sample_id']:06d}.npz", allow_pickle=False) as data:
        w, r = data['wavelength_nm'], data['reflectance']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        fig = plt.figure(figsize=(12, 7))
        grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.65))
        ax = fig.add_subplot(grid[:, 0])
        for i, vertices in enumerate(geometry.vertices()):
            ax.add_patch(Polygon(vertices, color=('#e29641' if i in (0, 3) else '#417bc4'), alpha=.9))
        ax.set(xlim=(-geometry.p/2, geometry.p/2), ylim=(-geometry.p/2, geometry.p/2),
               aspect='equal', xlabel='x (nm)', ylabel='y (nm)', title='As2Se3 tetramer, top view')
        ax.grid(alpha=.2)
        for row, limits in enumerate(((700, 1700), (800, 980))):
            a = fig.add_subplot(grid[row, 1])
            a.plot(w, r, lw=1.4, color='#28558b')
            for index, dip in enumerate(candidate['dominant_pair']['dips'] if candidate['dominant_pair'] else []):
                a.plot(dip['wavelength_nm'], dip['reflectance_min'], 'o', color='#c64232')
                a.annotate(f"{dip['wavelength_nm']:.1f} nm\nR={dip['reflectance_min']:.3f}",
                           (dip['wavelength_nm'], dip['reflectance_min']),
                           xytext=((-8 if row == 0 and index == 0 else 8), 12),
                           ha=('right' if row == 0 and index == 0 else 'left'),
                           textcoords='offset points', fontsize=9)
            a.set(xlim=limits, ylim=(-.02, 1.03), xlabel='Wavelength (nm)', ylabel='Reflectance')
            a.grid(alpha=.2)
        mesh = config.simulation.maximum_mesh_nm or config.simulation.mesh_nm
        fig.suptitle(f'Real FDTD pilot: {config.simulation.substrate_depth_nm:g} nm SiO2 / '
                     f'{config.simulation.metal_thickness_nm:g} nm metal | {mesh:g} nm mesh')
        fig.tight_layout()
        fig.savefig(output / 'double_dip_candidate.png', dpi=180)
        plt.close(fig)
    files = {str(path.relative_to(output)): checksum(path) for path in sorted(output.rglob('*'))
             if path.is_file() and path.name != 'checksums.json'}
    atomic_json(output / 'checksums.json', files)
    print(f'Exported {len(rows)} real samples and training evidence to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampling-config', type=Path, default=PROJECT_ROOT/'configs/as2se3_au_pilot.json')
    parser.add_argument('--training-config', type=Path, default=PROJECT_ROOT/'configs/train_as2se3_pilot.json')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'research_results/seven_parameter_pilot')
    args = parser.parse_args()
    export(args.sampling_config, args.training_config, args.output)
