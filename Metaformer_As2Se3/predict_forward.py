"""Use a saved forward model on seven physical parameters; no FDTD required."""
import argparse
from pathlib import Path

import numpy as np
import torch

from forward_metaformer import ForwardMetaformer, ForwardMetaformerConfig

ROOT = Path(__file__).resolve().parent


def load_forward_checkpoint(path, device='cpu'):
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if payload.get('format') != 'forward_metaformer_v1':
        raise ValueError('This file is not a forward Metaformer checkpoint')
    model = ForwardMetaformer(ForwardMetaformerConfig(**payload['model_config']))
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device).eval()
    return model, payload['metadata']


def normalize_physical_geometry(values, metadata):
    geometry = np.asarray(values, dtype=float)
    if geometry.ndim == 1:
        geometry = geometry[None, :]
    lower, upper = np.array(metadata['lower_values']), np.array(metadata['upper_values'])
    if geometry.ndim != 2 or geometry.shape[1] != 7 or not np.isfinite(geometry).all():
        raise ValueError('Expected finite physical geometry [N,7]')
    if np.any(geometry < lower) or np.any(geometry > upper):
        raise ValueError('Geometry lies outside the checkpoint training bounds; no silent clipping')
    return ((geometry-lower)/(upper-lower)).astype(np.float32)


@torch.inference_mode()
def predict_physical_geometry(model, metadata, values, device='cpu'):
    normalized = normalize_physical_geometry(values, metadata)
    prediction = model(torch.from_numpy(normalized).to(device)).cpu().numpy()
    return np.asarray(metadata['wavelength_nm']), prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'models/forward_au100_seed42.pt')
    parser.add_argument('--geometry', type=float, nargs=7, required=True,
                        metavar=('P','L','H','dX','dY','odd_deg','even_deg'))
    parser.add_argument('--output', type=Path, default=ROOT/'output/forward_prediction.csv')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    from train_forward_metaformer import configure_torch
    configure_torch()
    model, metadata = load_forward_checkpoint(args.checkpoint, args.device)
    wave, prediction = predict_physical_geometry(model, metadata, args.geometry, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.column_stack((wave, prediction[0])), delimiter=',',
               header='wavelength_nm,predicted_reflectance', comments='')
    print(f'Saved {len(wave)} predicted reflectance values: {args.output}')
    print('Scope: pilot numerical surrogate; no FDTD solver was called.')


if __name__ == '__main__':
    main()
