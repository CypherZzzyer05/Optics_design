"""Load the selected inverse checkpoint and predict seven physical parameters."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metaformer import InverseMetaformer, MetaformerConfig

ROOT = Path(__file__).resolve().parent


def load_inverse(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    model = InverseMetaformer(MetaformerConfig.from_dict(metadata['config']['model']))
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    return model, metadata


def predict(model, metadata, reflectance, wavelength_nm):
    values = np.asarray(reflectance, dtype=np.float32)
    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    expected = np.asarray(metadata['wavelength_nm'])
    if values.shape != expected.shape or wavelength.shape != expected.shape:
        raise ValueError('Input must have the same 1000 wavelength points as the checkpoint')
    if not np.isfinite(values).all() or not np.isfinite(wavelength).all():
        raise ValueError('Input contains nonfinite numbers')
    if not np.allclose(wavelength, expected, rtol=0, atol=1e-5):
        raise ValueError('Wavelength grid differs; explicitly resample before calling inference')
    device = next(model.parameters()).device
    with torch.inference_mode():
        normalized = model(torch.from_numpy(values[None]).to(device)).cpu().numpy()[0]
    lower, upper = np.asarray(metadata['lower_values']), np.asarray(metadata['upper_values'])
    physical = lower + normalized * (upper - lower)
    return {'parameters': {name: {'value': float(value), 'unit': unit}
                          for name, value, unit in zip(metadata['parameter_names'], physical,
                                                       metadata['parameter_units'])},
            'normalized': normalized.tolist(), 'checkpoint_seed': metadata['config']['seed'],
            'quality_level': metadata['quality_level'],
            'status': 'candidate_geometry_requires_geometry_and_FDTD_validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/inverse_au100_selected.pt')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--spectrum', type=Path, help='CSV columns: wavelength_nm,reflectance')
    source.add_argument('--dataset', type=Path, default=ROOT / 'examples/au100_metal8_100/dataset.npz')
    parser.add_argument('--sample-id', type=int, default=1, help='Used only for the dataset demo')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, metadata = load_inverse(args.checkpoint)
    if args.spectrum:
        table = np.genfromtxt(args.spectrum, delimiter=',', names=True, encoding='utf-8-sig')
        if not {'wavelength_nm', 'reflectance'} <= set(table.dtype.names or ()):
            raise ValueError('CSV requires wavelength_nm and reflectance columns')
        values, wavelength = table['reflectance'], table['wavelength_nm']
    else:
        with np.load(args.dataset, allow_pickle=False) as data:
            matches = np.flatnonzero(data['sample_ids'] == args.sample_id)
            if len(matches) != 1:
                raise ValueError('sample-id must identify exactly one row')
            values, wavelength = data['spectra'][matches[0]], data['wavelength_nm']
    result = predict(model, metadata, values, wavelength)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(text.encode('utf-8'))
    print(text)


if __name__ == '__main__':
    main()
