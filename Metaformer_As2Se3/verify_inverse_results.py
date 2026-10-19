"""Independently check published inverse artifacts against data and checkpoints."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fdtd_pipeline.pipeline import checksum
from predict_inverse import load_inverse

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT, check_cuda=True):
    root = Path(root)
    directory = root / 'results/inverse_au100_ablation'
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    checkpoint_info = json.loads((directory / 'checkpoint.json').read_text(encoding='utf-8'))
    data_path = root / 'examples/au100_metal8_100/dataset.npz'
    assert checksum(data_path) == report['plan']['dataset_sha256']
    for name, expected in report['plan']['source_sha256'].items():
        assert checksum(root / name) == expected, name
    model_path = root / checkpoint_info['path']
    assert checksum(model_path) == checkpoint_info['sha256']
    model, metadata = load_inverse(model_path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = report['selection']['selected_variant']
    means = {name: np.mean([t['best_val_mse'] for t in report['trials'] if t['variant'] == name])
             for name in report['plan']['variants']}
    assert selected == min(means, key=means.get)
    ranked = sorted((t for t in report['trials'] if t['variant'] == selected), key=lambda t: t['best_val_mse'])
    assert checkpoint_info['seed'] == ranked[len(ranked) // 2]['seed'] == metadata['config']['seed']
    maximum_metric_difference = 0.
    for row in report['test_seed_metrics']:
        table = np.genfromtxt(directory / f"test_{row['variant']}_seed{row['seed']}.csv", delimiter=',', names=True)
        truth = np.column_stack([table[f'true_01_{p}'] for p in metadata['parameter_names']])
        prediction = np.column_stack([table[f'pred_01_{p}'] for p in metadata['parameter_names']])
        difference = prediction - truth
        difference[:, 5:] = (difference[:, 5:] + .5) % 1 - .5
        mse = float(np.mean(difference ** 2))
        maximum_metric_difference = max(maximum_metric_difference, abs(mse - row['metrics']['normalized_mse']))
        assert abs(mse - row['metrics']['normalized_mse']) < 1e-9
        assert sorted(table['sample_id'].astype(int)) == sorted(report['split_sample_ids']['test'])
    with np.load(data_path, allow_pickle=False) as data:
        train_spectra = torch.from_numpy(data['spectra'][data['train_indices']].astype(np.float32))
        test_spectra = torch.from_numpy(data['spectra'][data['test_indices']].astype(np.float32))
        expected_ids = data['sample_ids'][data['train_indices']].tolist()
        assert metadata['input_normalization_fit_sample_ids'] == expected_ids
        if model.config.input_normalization == 'train_standard':
            torch.testing.assert_close(model.spectrum_mean, train_spectra.mean(0), atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(model.spectrum_scale,
                train_spectra.std(0, unbiased=False).clamp_min(model.config.normalization_floor), atol=1e-7, rtol=1e-6)
    with torch.inference_mode():
        cpu = model(test_spectra).numpy()
        table = np.genfromtxt(directory / f"test_{selected}_seed{checkpoint_info['seed']}.csv", delimiter=',', names=True)
        archived = np.column_stack([table[f'pred_01_{p}'] for p in metadata['parameter_names']])
        max_reload_difference = float(np.max(np.abs(cpu - archived)))
        assert max_reload_difference < 1e-6
        cuda_difference = None
        if check_cuda and torch.cuda.is_available():
            model.to('cuda')
            cuda = model(test_spectra.to('cuda')).cpu().numpy()
            cuda_difference = float(np.max(np.abs(cuda - cpu)))
            assert cuda_difference < 1e-5
            model.cpu()
        for _ in range(10):
            model(test_spectra[:1])
        timings = []
        for _ in range(100):
            start = time.perf_counter()
            model(test_spectra[:1])
            timings.append((time.perf_counter() - start) * 1000)
    result = {'passed': True, 'source_and_dataset_hashes_verified': True,
              'selection_recomputed_from_validation_only': True,
              'normalization_training_ids_verified': True,
              'max_csv_metric_recalculation_difference': maximum_metric_difference,
              'max_checkpoint_reload_prediction_difference': max_reload_difference,
              'max_cpu_cuda_prediction_difference': cuda_difference,
              'cpu_batch1_median_ms': float(np.median(timings)), 'cpu_threads': 4,
              'timing_scope': 'FP32 warmed model forward only; excludes loading and CSV I/O',
              'trainable_parameters': sum(p.numel() for p in model.parameters())}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    result = verify(check_cuda=not args.no_cuda)
    if args.save:
        path = ROOT / 'results/inverse_au100_ablation/verification.json'
        path.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
