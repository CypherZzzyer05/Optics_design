"""Reuse the audited inverse dataset schema and the exact saved data splits."""
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.utils.data import TensorDataset

from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import validate_training_splits


def load_forward_data(path, allow_validation_data=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_training_splits(arrays)
    source = SpectrumGeometryDataset(path, split='all',
                                    allow_validation_data=allow_validation_data)
    if len(source.parameter_names) != 7:
        raise ValueError('Forward pilot requires the seven-parameter dataset')
    if torch.any(source.spectra < 0) or torch.any(source.spectra > 1):
        raise ValueError('Reflectance targets must lie within [0,1]; do not silently clip data')
    splits = {}
    ids = {}
    for name in ('train', 'val', 'test'):
        indices = arrays[name + '_indices']
        splits[name] = TensorDataset(source.labels[indices], source.spectra[indices])
        ids[name] = source.sample_ids[indices]
    # This constant predictor is fitted on training labels only.
    train_mean = splits['train'].tensors[1].numpy().mean(axis=0, dtype=np.float64).astype(np.float32)
    metadata = {'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'dataset_config_fingerprint': source.metadata['config_fingerprint'],
                'quality_level': source.metadata['config']['quality_level'],
                'source_dataset_metadata': source.metadata,
                'parameter_names': list(source.parameter_names),
                'parameter_units': list(source.parameter_units),
                'lower_values': source.lower_values.tolist(),
                'upper_values': source.upper_values.tolist(),
                'wavelength_nm': source.wavelength_nm.tolist(),
                'split_sample_ids': {key: value.tolist() for key, value in ids.items()},
                'target': 'reflectance',
                'scope': 'Historical v1 Au100/finite-SiO2/As2Se3 numerical outputs; pipeline validation, not certified physical material accuracy.'}
    return splits, ids, train_mean, metadata
