"""Train the first forward-only Metaformer pilot; inverse training is unchanged."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fdtd_pipeline.pipeline import atomic_json, directory_lock
from forward_data import load_forward_data
from forward_metaformer import ForwardMetaformer, ForwardMetaformerConfig
from forward_metrics import spectral_metrics
from train_metaformer import save_checkpoint

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/train_forward_au100_pilot.json'


def configure_torch(threads=4):
    torch.set_num_threads(threads)
    # Match full-FP32 CPU/GPU evaluation; AMP remains available for training.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    required = {'dataset', 'output_dir', 'allow_validation_data', 'seed', 'device',
                'epochs', 'batch_size', 'learning_rate', 'weight_decay',
                'early_stopping_patience', 'min_delta', 'mixed_precision', 'model'}
    if set(config) != required:
        raise ValueError(f'Unexpected or missing configuration keys: {set(config) ^ required}')
    for key in ('dataset', 'output_dir'):
        config[key] = str((path.parent / config[key]).resolve())
    ForwardMetaformerConfig(**config['model'])
    for key in ('epochs', 'batch_size', 'early_stopping_patience'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('learning_rate', 'weight_decay', 'min_delta'):
        if not math.isfinite(config[key]) or config[key] < (1e-12 if key == 'learning_rate' else 0):
            raise ValueError(f'Invalid {key}')
    if config['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Unknown device')
    for key in ('mixed_precision', 'allow_validation_data'):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be boolean')
    return config


@torch.inference_mode()
def predict_batches(model, dataset, device, batch_size):
    model.eval()
    predictions = []
    for geometry, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        predictions.append(model(geometry.to(device)).float().cpu().numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite forward prediction')
    return result


def train(config):
    configure_torch()
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cuda' if config['device'] == 'auto' and torch.cuda.is_available()
                          else 'cpu' if config['device'] == 'auto' else config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    datasets, ids, baseline, metadata = load_forward_data(
        config['dataset'], config['allow_validation_data'])
    model = ForwardMetaformer(ForwardMetaformerConfig(**config['model'])).to(device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=8, min_lr=1e-6)
    use_amp = config['mixed_precision'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(datasets['train'], batch_size=config['batch_size'],
                              shuffle=True, num_workers=0, generator=generator)
    metadata.update({'config': config, 'task': 'forward_geometry_to_reflectance',
                     'device': str(device), 'pytorch_version': str(torch.__version__),
                     'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                     'model_parameters': sum(p.numel() for p in model.parameters()),
                     'precision': {'training_amp': use_amp, 'evaluation': 'FP32; TF32 disabled'},
                     'selection': 'Lowest validation full-spectrum MSE; test evaluated after selection only.',
                     'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                         for name in ['forward_metaformer.py', 'forward_metrics.py',
                                      'forward_data.py', 'train_forward_metaformer.py',
                                      'metaformer.py', 'spectrum_dataset.py', 'train_metaformer.py']}})
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    # UTC names avoid depending on the host's timezone configuration.
    run = output / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    run.mkdir(exist_ok=False)
    with directory_lock(output), directory_lock(run):
        handler = logging.FileHandler(run/'training.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        LOG.addHandler(handler)
        try:
            atomic_json(run/'run.json', metadata)
            atomic_json(run/'status.json', {'state': 'training'})
            best, best_epoch, stale, total_steps = float('inf'), 0, 0, 0
            history = []
            start = time.perf_counter()
            for epoch in range(1, config['epochs']+1):
                model.train()
                total, count, epoch_steps = 0., 0, 0
                for geometry, spectrum in train_loader:
                    geometry, spectrum = geometry.to(device), spectrum.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        prediction = model(geometry)
                        loss = F.mse_loss(prediction.float(), spectrum)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if torch.isfinite(norm):
                        scaler.step(optimizer)
                        epoch_steps += 1
                    elif not use_amp:
                        raise ValueError('Nonfinite gradients')
                    scaler.update()
                    total += float(loss.detach()) * len(geometry)
                    count += len(geometry)
                if not epoch_steps:
                    raise ValueError('No optimizer update')
                total_steps += epoch_steps
                val = predict_batches(model, datasets['val'], device, config['batch_size'])
                val_mse = float(np.mean((val-datasets['val'].tensors[1].numpy())**2))
                history.append({'epoch': epoch, 'train_mse': total/count, 'val_mse': val_mse,
                                'learning_rate': optimizer.param_groups[0]['lr'],
                                'optimizer_steps': total_steps})
                if val_mse < best - config['min_delta']:
                    best, best_epoch, stale = val_mse, epoch, 0
                    save_checkpoint(run/'best.pt', {'format': 'forward_metaformer_v1',
                        'model_config': asdict(model.config),
                        'model_state': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'metadata': metadata, 'best_epoch': best_epoch, 'best_val_mse': best})
                else:
                    stale += 1
                scheduler.step(val_mse)
                atomic_json(run/'history.json', history)
                if epoch == 1 or epoch % 10 == 0:
                    LOG.info('Epoch %d: train MSE %.6f; validation MSE %.6f; best epoch %d',
                             epoch, total/count, val_mse, best_epoch)
                if stale >= config['early_stopping_patience']:
                    break
            if device.type == 'cuda':
                torch.cuda.synchronize()
            training_seconds = time.perf_counter()-start
            checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            change = max(float((value.detach().cpu()-initial_state[key]).abs().max())
                         for key, value in model.state_dict().items())
            if not change > 0:
                raise ValueError('Parameters did not change during training')
            scores, baselines = {}, {}
            arrays = {'wavelength_nm': np.array(metadata['wavelength_nm']), 'train_mean_spectrum': baseline}
            # No test inference or test metric calculation occurs before this point.
            for split in ('train', 'val', 'test'):
                prediction = predict_batches(model, datasets[split], device, config['batch_size'])
                truth = datasets[split].tensors[1].numpy()
                scores[split] = spectral_metrics(prediction, truth, arrays['wavelength_nm'], ids[split])
                constant = np.broadcast_to(baseline, truth.shape)
                baselines[split] = spectral_metrics(constant, truth, arrays['wavelength_nm'], ids[split])
                arrays.update({split+'_prediction': prediction, split+'_target': truth,
                               split+'_sample_ids': ids[split],
                               split+'_geometry_01': datasets[split].tensors[0].numpy()})
            # Verify the exact inference checkpoint on CPU, including batch-size invariance.
            cpu_model = ForwardMetaformer(ForwardMetaformerConfig(**checkpoint['model_config']))
            cpu_model.load_state_dict(checkpoint['model_state'])
            cpu_prediction = predict_batches(cpu_model, datasets['test'], torch.device('cpu'), 1)
            cpu_diff = float(np.max(np.abs(cpu_prediction-arrays['test_prediction'])))
            if cpu_diff > 2e-5:
                raise ValueError(f'CPU/checkpoint prediction mismatch: {cpu_diff}')
            np.savez_compressed(run/'predictions.npz', **arrays)
            metrics = {'best_epoch': best_epoch, 'epochs_completed': len(history),
                       'stop_reason': 'early_stopping' if stale >= config['early_stopping_patience'] else 'epoch_limit',
                       'optimizer_steps': total_steps, 'training_seconds': training_seconds,
                       'max_parameter_change_from_initial': change,
                       'cpu_checkpoint_max_abs_difference': cpu_diff,
                       'metrics': scores, 'train_mean_baseline': baselines,
                       'best_checkpoint_sha256': hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()}
            atomic_json(run/'metrics.json', metrics)
            atomic_json(run/'status.json', {'state':'completed', 'best_epoch': best_epoch})
            atomic_json(output/'latest_run.json', {'run_dir':str(run),'state':'completed'})
            LOG.info('Completed: test MSE %.6f; train-mean MSE %.6f; %s',
                     scores['test']['mse'], baselines['test']['mse'], run)
            return run, metrics
        except BaseException as error:
            atomic_json(run/'status.json', {'state':'failed','error':str(error)})
            raise
        finally:
            LOG.removeHandler(handler)
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run, metrics = train(load_config(args.config))
    print(json.dumps({'run':str(run), 'best_epoch':metrics['best_epoch'],
                      'test_mse':metrics['metrics']['test']['mse']}, indent=2))


if __name__ == '__main__':
    main()
