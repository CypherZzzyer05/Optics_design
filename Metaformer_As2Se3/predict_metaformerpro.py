"""Predict seven physical parameters with the selected MetaformerPro checkpoint."""
import argparse
import json
from pathlib import Path
import webbrowser

import numpy as np
import torch

from fdtd_pipeline.config import Geometry
from train_metaformerpro import load_pro

ROOT = Path(__file__).resolve().parent


def predict(checkpoint_path, reflectance, wavelength_nm):
    model, saved = load_pro(checkpoint_path)
    expected = np.asarray(saved['wavelength_nm'])
    values, wavelength = np.asarray(reflectance, dtype=np.float32), np.asarray(wavelength_nm)
    if values.shape != expected.shape or wavelength.shape != expected.shape:
        raise ValueError('Expected the saved 1000 wavelength points')
    if not np.isfinite(values).all() or not np.allclose(wavelength, expected, rtol=0, atol=1e-5):
        raise ValueError('Nonfinite reflectance or different wavelength grid')
    with torch.inference_mode():
        normalized = model(torch.from_numpy(values[None])).numpy()[0]
    lower, upper = np.asarray(saved['lower_values']), np.asarray(saved['upper_values'])
    physical = lower + normalized * (upper - lower)
    try:
        Geometry(*physical[:5], theta_odd_deg=physical[5], theta_even_deg=physical[6])
        valid, reason = True, None
    except ValueError as error:
        valid, reason = False, str(error)
    return {'model': 'MetaformerPro', 'version': saved['version'], 'seed': saved['seed'],
            'model_config': saved['model_config'],
            'parameters': dict(zip(saved['parameter_names'], physical.tolist())),
            'units': saved['parameter_units'], 'geometry_valid': valid, 'geometry_error': reason,
            'status': 'candidate_requires_FDTD_validation' if valid else 'invalid_geometry',
            'quality_scope': saved['quality_scope'],
            'simulation_settings': saved.get('simulation_settings')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/metaformerpro.pt')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--spectrum', type=Path, help='CSV columns: wavelength_nm,reflectance')
    source.add_argument('--dataset', type=Path, default=ROOT / 'examples/au100_metal8_1000/dataset.npz')
    parser.add_argument('--sample-id', type=int, default=207)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/metaformerpro_prediction.json')
    parser.add_argument('--show', action='store_true', help='Open the saved structure image in the default browser')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.spectrum:
        table = np.genfromtxt(args.spectrum, delimiter=',', names=True, encoding='utf-8-sig')
        spectrum, wavelength = table['reflectance'], table['wavelength_nm']
        source = {'type': 'csv', 'path': str(args.spectrum.resolve())}
        source_label = args.spectrum.name
    else:
        with np.load(args.dataset, allow_pickle=False) as data:
            matches = np.flatnonzero(data['sample_ids'] == args.sample_id)
            if len(matches) != 1:
                raise ValueError('sample-id must match one saved sample')
            spectrum, wavelength = data['spectra'][matches[0]], data['wavelength_nm']
        source = {'type': 'dataset', 'path': str(args.dataset.resolve()), 'sample_id': args.sample_id}
        source_label = f'固定样本 {args.sample_id}'
    result = predict(args.checkpoint, spectrum, wavelength)
    result['input'] = source
    result['checkpoint'] = str(args.checkpoint.resolve())
    result['figure'] = None
    if result['geometry_valid']:
        from render_metaformerpro_structure import render_structure
        figure = render_structure(result, wavelength, spectrum, args.output.with_suffix('.png'), source_label)
        result['figure'] = str(figure)
    else:
        args.output.with_suffix('.png').unlink(missing_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.show and result['figure']:
        webbrowser.open(Path(result['figure']).as_uri())


if __name__ == '__main__':
    main()
