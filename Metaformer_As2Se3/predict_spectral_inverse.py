"""Predict a seven-parameter candidate with the validation-selected model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fdtd_pipeline.config import Geometry
from optimize_spectral_inverse import load_model

ROOT = Path(__file__).resolve().parent


def predict(checkpoint_path, reflectance, wavelength_nm):
    model, checkpoint = load_model(checkpoint_path, torch.device('cpu'))
    metadata = checkpoint.get('metadata', checkpoint)
    expected = np.asarray(metadata['wavelength_nm'])
    values, wavelength = np.asarray(reflectance, dtype=np.float32), np.asarray(wavelength_nm)
    if values.shape != expected.shape or wavelength.shape != expected.shape:
        raise ValueError('Input must contain the saved 1000 wavelength samples')
    if not np.isfinite(values).all() or not np.allclose(wavelength, expected, rtol=0, atol=1e-5):
        raise ValueError('Nonfinite reflectance or different wavelength grid')
    with torch.inference_mode():
        normalized = model(torch.from_numpy(values[None])).numpy()[0]
    lower, upper = np.asarray(metadata['lower_values']), np.asarray(metadata['upper_values'])
    physical = lower + normalized * (upper - lower)
    try:
        Geometry(*physical[:5], theta_odd_deg=physical[5], theta_even_deg=physical[6])
        valid, reason = True, None
    except ValueError as error:
        valid, reason = False, str(error)
    return {'parameters': dict(zip(metadata['parameter_names'], physical.tolist())),
            'units': list(metadata['parameter_units']), 'normalized': normalized.tolist(),
            'geometry_valid': valid, 'geometry_error': reason,
            'status': 'candidate_requires_FDTD_validation',
            'quality_scope': 'Historical 1.0 Au-bottom numerical dataset; known Au loss-fit limitation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/inverse_features1000.pt')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--spectrum', type=Path, help='CSV: wavelength_nm,reflectance')
    source.add_argument('--dataset', type=Path, default=ROOT / 'examples/au100_metal8_1000/dataset.npz')
    parser.add_argument('--sample-id', type=int, default=207)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/feature_prediction.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.spectrum:
        table = np.genfromtxt(args.spectrum, delimiter=',', names=True, encoding='utf-8-sig')
        spectrum, wavelength = table['reflectance'], table['wavelength_nm']
    else:
        with np.load(args.dataset, allow_pickle=False) as data:
            matches = np.flatnonzero(data['sample_ids'] == args.sample_id)
            if len(matches) != 1:
                raise ValueError('sample-id must match one row')
            spectrum, wavelength = data['spectra'][matches[0]], data['wavelength_nm']
    result = predict(args.checkpoint, spectrum, wavelength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
