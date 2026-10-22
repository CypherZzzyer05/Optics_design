"""Train validation-only ablations, freeze selection, then evaluate the winner.

No native solver is used. Historical data and checkpoints remain unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from metaformer import InverseMetaformer, MetaformerConfig
from spectral_inverse import SpectralConfig, SpectralInverse, inverse_loss, parameter_errors
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import validate_training_splits

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(prediction, target, widths):
    p, t = torch.as_tensor(prediction).double(), torch.as_tensor(target).double()
    result = {}
    for name, mirror in [('periodic', False), ('mirror_equivalent', True)]:
        error = parameter_errors(p, t, mirror).numpy()
        physical = error * widths
        result[name] = {
            'mse': float(np.mean(error ** 2)),
            'length_mse': float(np.mean(error[:, :5] ** 2)),
            'angle_mse': float(np.mean(error[:, 5:] ** 2)),
            'mae_nm': np.abs(physical[:, :5]).mean(0).tolist(),
            'mae_deg': np.abs(physical[:, 5:]).mean(0).tolist(),
        }
    return result


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=32):
    model.eval()
    predictions = [model(x.to(device)).cpu().numpy() for x, _ in
                   DataLoader(dataset, batch_size=batch_size, shuffle=False)]
    prediction = np.concatenate(predictions)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite model prediction')
    return prediction, metrics(prediction, dataset.labels.numpy(), dataset.upper_values - dataset.lower_values)


def make_model(variant, step_nm):
    if variant['architecture'] == 'baseline':
        settings = json.loads((ROOT / 'configs/train_inverse_1000.json').read_text(encoding='utf-8'))['model']
        return InverseMetaformer(MetaformerConfig.from_dict(settings))
    return SpectralInverse(SpectralConfig(view=variant['view'], mirror_loss=variant['mirror_loss'],
                                         wavelength_step_nm=step_nm))


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'metadata' in checkpoint:
        model = InverseMetaformer(MetaformerConfig.from_dict(checkpoint['metadata']['config']['model']))
    elif checkpoint['architecture'] == 'baseline':
        model = InverseMetaformer(MetaformerConfig.from_dict(checkpoint['model_config']))
    else:
        model = SpectralInverse(SpectralConfig(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(), checkpoint


def train_one(name, variant, seed, plan, datasets, output, device):
    run = output / name / f'seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    completed = run / 'validation.json'
    if completed.exists():
        return json.loads(completed.read_text(encoding='utf-8'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    step = float(np.diff(datasets['train'].wavelength_nm).mean())
    widths = datasets['train'].upper_values - datasets['train'].lower_values
    history = []
    start = time.perf_counter()
    if name == 'baseline' and seed == 44 and Path(plan['historical_checkpoint']).is_file():
        model, checkpoint = load_model(plan['historical_checkpoint'], device)
        if checkpoint['metadata']['dataset_sha256'] != plan['dataset_sha256']:
            raise ValueError('Historical checkpoint belongs to different data')
        history = json.loads((ROOT / 'results/inverse_au100_1000/history.json').read_text(encoding='utf-8'))
        best_epoch = checkpoint['epoch']
        shutil.copyfile(plan['historical_checkpoint'], run / 'best.pt')
        reused = True
        original = json.loads((ROOT / 'results/inverse_au100_1000/summary.json').read_text(encoding='utf-8'))
        training_seconds = original['training_seconds']
    else:
        model = make_model(variant, step).to(device)
        model.fit_input_normalization(datasets['train'].spectra.to(device))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(datasets['train'], batch_size=plan['batch_size'], shuffle=True,
                            generator=generator, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, min_lr=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        best, best_epoch, stale = float('inf'), 0, 0
        for epoch in range(1, plan['epochs'] + 1):
            model.train()
            total = 0.
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                    prediction = model(x)
                    loss = inverse_loss(prediction, y, variant['mirror_loss'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if torch.isfinite(norm):
                    scaler.step(optimizer)
                scaler.update()
                total += loss.detach().item() * len(x)
            _, score = evaluate(model, datasets['val'], device)
            # Within-run early stopping uses that run's training objective.
            # Across variants selection ALWAYS uses the same mirror-equivalent metric.
            objective = score['mirror_equivalent' if variant['mirror_loss'] else 'periodic']['mse']
            scheduler.step(objective)
            significant = objective < best - 1e-5
            if objective < best:
                best, best_epoch = objective, epoch
                torch.save({'model_state': model.state_dict(), 'model_config': asdict(model.config),
                            'architecture': variant['architecture'], 'variant': variant,
                            'seed': seed, 'epoch': epoch, 'dataset_sha256': plan['dataset_sha256'],
                            'parameter_names': list(datasets['train'].parameter_names),
                            'parameter_units': list(datasets['train'].parameter_units),
                            'lower_values': datasets['train'].lower_values.tolist(),
                            'upper_values': datasets['train'].upper_values.tolist(),
                            'wavelength_nm': datasets['train'].wavelength_nm.tolist(),
                            'quality_scope': plan['quality_scope']}, run / 'best.pt')
            stale = 0 if significant else stale + 1
            history.append({'epoch': epoch, 'train_mse': total / len(datasets['train']),
                            'validation': score, 'learning_rate': optimizer.param_groups[0]['lr']})
            if epoch % 10 == 0 or epoch == 1:
                LOG.info('%s seed%d epoch%d train=%.5f val=%.5f equivalent=%.5f best_epoch=%d',
                         name, seed, epoch, history[-1]['train_mse'], score['periodic']['mse'],
                         score['mirror_equivalent']['mse'], best_epoch)
            write_json(output / 'status.json', {'variant': name, 'seed': seed, 'epoch': epoch,
                       'best_epoch': best_epoch, 'validation': score, 'state': 'training'})
            if stale >= plan['patience']:
                break
        training_seconds = time.perf_counter() - start
        model, _ = load_model(run / 'best.pt', device)
        reused = False
    val_prediction, val_metrics = evaluate(model, datasets['val'], device)
    train_prediction, train_metrics = evaluate(model, datasets['train'], device)
    np.savez_compressed(run / 'validation_predictions.npz', prediction=val_prediction,
                        target=datasets['val'].labels.numpy(), sample_ids=datasets['val'].sample_ids)
    record = {'variant': name, 'seed': seed, 'best_epoch': best_epoch, 'epochs': len(history),
              'parameters': sum(p.numel() for p in model.parameters()),
              'training_seconds': training_seconds, 'historical_checkpoint_reused': reused,
              'validation': val_metrics, 'train': train_metrics,
              'checkpoint': str(run / 'best.pt'), 'checkpoint_sha256': sha(run / 'best.pt'),
              'test_evaluated': False}
    write_json(run / 'history.json', history)
    write_json(completed, record)
    LOG.info('Finished %s seed%d: equivalent val %.6f, %.1fs', name, seed,
             val_metrics['mirror_equivalent']['mse'], training_seconds)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return record


def run_study(plan, datasets, output, device):
    records = []
    for name, variant in plan['variants'].items():
        for seed in plan['seeds']:
            records.append(train_one(name, variant, seed, plan, datasets, output, device))
            write_json(output / 'validation_results.json', records)
    ranking = []
    for name in plan['variants']:
        trials = [r for r in records if r['variant'] == name]
        scores = [r['validation']['mirror_equivalent']['mse'] for r in trials]
        ranking.append({'variant': name, 'mean_equivalent_validation_mse': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'mean_periodic_validation_mse': float(np.mean([r['validation']['periodic']['mse'] for r in trials])),
                        'parameters': trials[0]['parameters']})
    ranking.sort(key=lambda r: r['mean_equivalent_validation_mse'])
    selected = ranking[0]['variant']
    selected_trials = sorted([r for r in records if r['variant'] == selected],
                             key=lambda r: r['validation']['mirror_equivalent']['mse'])
    selection = {'selected': selected, 'representative_seed': selected_trials[len(selected_trials) // 2]['seed'],
                 'ranking': ranking, 'selection_metric': plan['selection'], 'test_evaluated': False,
                 'plan_sha256': sha(output / 'plan.json')}
    # This file is written BEFORE any access to test predictions.
    write_json(output / 'selection.json', selection)
    write_json(output / 'status.json', {'state': 'validation_complete', **selection})
    return selection


def finalize(plan, output, results, device):
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    trials = json.loads((output / 'validation_results.json').read_text(encoding='utf-8'))
    test = SpectrumGeometryDataset(plan['dataset'], 'test', allow_validation_data=True)
    evaluated = []
    results.mkdir(parents=True, exist_ok=True)
    for name in dict.fromkeys(('baseline', selection['selected'])):
        predictions = []
        for trial in [r for r in trials if r['variant'] == name]:
            model, _ = load_model(trial['checkpoint'], device)
            prediction, score = evaluate(model, test, device)
            predictions.append(prediction)
            evaluated.append({'variant': name, 'seed': trial['seed'], 'test': score})
            np.savez_compressed(results / f'{name}_seed{trial["seed"]}_test.npz', prediction=prediction,
                                target=test.labels.numpy(), sample_ids=test.sample_ids)
            del model
        # Physical lengths average linearly; square angles average on the circle.
        # Mirror representatives are first aligned jointly to the first member.
        aligned = [predictions[0]]
        reference = torch.as_tensor(predictions[0])
        from spectral_inverse import mirror_labels
        for p in predictions[1:]:
            t = torch.as_tensor(p)
            alternate = mirror_labels(t)
            direct = parameter_errors(reference, t).square()[:, 5:].sum(-1)
            other = parameter_errors(reference, alternate).square()[:, 5:].sum(-1)
            aligned.append(torch.where((other < direct)[:, None], alternate, t).numpy())
        stack = np.stack(aligned)
        ensemble = stack.mean(0)
        phase = np.exp(2j * np.pi * stack[:, :, 5:]).mean(0)
        ensemble[:, 5:] = (np.angle(phase) / (2 * np.pi)) % 1
        np.savez_compressed(results / f'{name}_ensemble_test.npz', prediction=ensemble,
                            target=test.labels.numpy(), sample_ids=test.sample_ids)
        evaluated.append({'variant': name, 'seed': 'ensemble', 'test': metrics(ensemble, test.labels.numpy(),
                                                 test.upper_values - test.lower_values)})
    for filename in ('plan.json', 'selection.json', 'validation_results.json'):
        shutil.copyfile(output / filename, results / filename)
    summary = {'selection': selection, 'evaluated': evaluated, 'test_samples': len(test),
               'test_policy': plan['test_policy'], 'dataset_sha256': plan['dataset_sha256'],
               'quality_scope': plan['quality_scope'], 'fdtd_run_performed': False}
    write_json(results / 'summary.json', summary)
    selected_trial = next(r for r in trials if r['variant'] == selection['selected']
                          and r['seed'] == selection['representative_seed'])
    target = ROOT / 'models/inverse_features1000.pt'
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(selected_trial['checkpoint'], target)
    write_json(output / 'status.json', {'state': 'completed', 'selected': selection['selected'],
                                      'checkpoint': str(target), 'summary': str(results / 'summary.json')})
    LOG.info('Final selection: %s, representative seed %s', selection['selected'], selection['representative_seed'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'finalize', 'all'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/inverse_feature_study.json')
    args = parser.parse_args()
    config = args.config.resolve()
    plan = json.loads(config.read_text(encoding='utf-8'))
    for key in ('dataset', 'output_dir', 'results_dir', 'historical_checkpoint'):
        plan[key] = str((config.parent / plan[key]).resolve())
    plan['dataset_sha256'] = sha(plan['dataset'])
    with np.load(plan['dataset'], allow_pickle=False) as archive:
        validate_training_splits(archive)
    output, results = Path(plan['output_dir']), Path(plan['results_dir'])
    output.mkdir(parents=True, exist_ok=True)
    plan['source_sha256'] = {name: sha(ROOT / name) for name in
                            ('spectral_inverse.py', 'optimize_spectral_inverse.py', 'metaformer.py')}
    saved_plan = output / 'plan.json'
    if saved_plan.exists() and json.loads(saved_plan.read_text(encoding='utf-8')) != plan:
        raise ValueError('Study plan changed; use a separate output directory')
    write_json(saved_plan, plan)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'study.log', encoding='utf-8')])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info('Device=%s; dataset=%s; no FDTD run', device, plan['dataset_sha256'])
    if args.command in ('train', 'all'):
        datasets = {s: SpectrumGeometryDataset(plan['dataset'], s, allow_validation_data=True) for s in ('train', 'val')}
        run_study(plan, datasets, output, device)
    if args.command in ('finalize', 'all'):
        finalize(plan, output, results, device)


if __name__ == '__main__':
    main()
