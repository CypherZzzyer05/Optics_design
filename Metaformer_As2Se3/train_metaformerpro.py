"""MetaformerPro validation-first ablation, checkpoint export and inference metadata."""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from metaformerpro import MetaformerPro, MetaformerProConfig, import_v01_state
from optimize_spectral_inverse import evaluate, metrics, sha, write_json, load_model as load_legacy
from spectral_inverse import inverse_loss
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import validate_training_splits

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


def load_pro(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = MetaformerPro(MetaformerProConfig(**saved['model_config']))
    model.load_state_dict(saved['model_state'], strict=True)
    return model.to(device).eval(), saved


def payload(model, plan, dataset, seed, epoch, version='0.2.0'):
    return {'family': 'MetaformerPro', 'version': version, 'model_config': asdict(model.config),
            'model_state': model.state_dict(), 'seed': seed, 'epoch': epoch,
            'dataset_sha256': plan['dataset_sha256'], 'source_sha256': plan['source_sha256'],
            'parameter_names': list(dataset.parameter_names), 'parameter_units': list(dataset.parameter_units),
            'lower_values': dataset.lower_values.tolist(), 'upper_values': dataset.upper_values.tolist(),
            'wavelength_nm': dataset.wavelength_nm.tolist(), 'quality_scope': plan['quality_scope'],
            'normalization_fit_sample_ids': dataset.sample_ids.tolist()}


def reference(seed, plan, datasets, run, device):
    records = json.loads(Path(plan['reference_results']).read_text(encoding='utf-8'))
    old = next(r for r in records if r['variant'] == 'compact_raw_mirror' and r['seed'] == seed)
    bundled = ROOT / f'models/metaformerpro_v01_seed{seed}.pt'
    archive = ROOT / f'results/inverse_features1000/validation_runs/compact_raw_mirror_seed{seed}'
    if bundled.is_file():
        model, checkpoint = load_pro(bundled, device)
        with np.load(archive / 'validation_predictions.npz', allow_pickle=False) as data:
            before = data['prediction'].copy()
    else:
        legacy, checkpoint = load_legacy(old['checkpoint'], device)
        model = MetaformerPro(MetaformerProConfig()).to(device)
        model.load_state_dict(import_v01_state(checkpoint['model_state']), strict=True)
        before, _ = evaluate(legacy, datasets['val'], device)
    if checkpoint['dataset_sha256'] != plan['dataset_sha256']:
        raise ValueError('Reference checkpoint uses different data')
    after, _ = evaluate(model, datasets['val'], device)
    difference = float(np.max(np.abs(before - after)))
    if difference > 1e-5:
        raise ValueError('MetaformerPro naming conversion changed reference predictions')
    torch.save(payload(model, plan, datasets['train'], seed, old['best_epoch'], '0.1.0'), run / 'best.pt')
    if not bundled.is_file():
        bundled.parent.mkdir(exist_ok=True)
        shutil.copyfile(run / 'best.pt', bundled)
    history = json.loads((archive / 'history.json').read_text(encoding='utf-8'))
    return model, history, old['best_epoch'], old['training_seconds'], difference


def train_one(name, settings, seed, plan, datasets, output, device):
    run = output / name / f'seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'validation.json').exists():
        return json.loads((run / 'validation.json').read_text(encoding='utf-8'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    conversion = None
    if name == 'pro_v01':
        model, history, best_epoch, seconds, conversion = reference(seed, plan, datasets, run, device)
    else:
        model = MetaformerPro(MetaformerProConfig(**settings)).to(device)
        model.fit_input_normalization(datasets['train'].spectra.to(device))
        loader = DataLoader(datasets['train'], batch_size=plan['batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=plan['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, min_lr=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        history, best, best_epoch, stale = [], float('inf'), 0, 0
        start = time.perf_counter()
        for epoch in range(1, plan['epochs'] + 1):
            model.train()
            total, updates = 0., 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                    loss = inverse_loss(model(x), y, True)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if torch.isfinite(norm):
                    scaler.step(optimizer)
                    updates += 1
                scaler.update()
                total += loss.detach().item() * len(x)
            if not updates:
                raise ValueError('Epoch had no finite optimizer update')
            _, score = evaluate(model, datasets['val'], device)
            value = score['mirror_equivalent']['mse']
            significant = value < best - 1e-5
            if value < best:
                best, best_epoch = value, epoch
                torch.save(payload(model, plan, datasets['train'], seed, epoch), run / 'best.pt')
            stale = 0 if significant else stale + 1
            scheduler.step(value)
            history.append({'epoch': epoch, 'train_mse': total / len(datasets['train']),
                            'validation': score, 'learning_rate': optimizer.param_groups[0]['lr']})
            write_json(output / 'status.json', {'state': 'training', 'variant': name, 'seed': seed,
                       'epoch': epoch, 'best_epoch': best_epoch, 'validation': score})
            if epoch == 1 or epoch % 10 == 0:
                LOG.info('%s seed%d epoch%d train=%.5f val=%.5f lengths=%.5f dX=%.3fnm best=%d',
                         name, seed, epoch, history[-1]['train_mse'], value,
                         score['mirror_equivalent']['length_mse'], score['mirror_equivalent']['mae_nm'][3], best_epoch)
            if stale >= plan['patience']:
                break
        seconds = time.perf_counter() - start
        model, _ = load_pro(run / 'best.pt', device)
    prediction, score = evaluate(model, datasets['val'], device)
    _, training_score = evaluate(model, datasets['train'], device)
    record = {'variant': name, 'seed': seed, 'validation': score, 'train': training_score,
              'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
              'epochs': len(history), 'training_seconds': seconds,
              'reference_reused': name == 'pro_v01', 'reference_conversion_max_difference': conversion,
              'checkpoint': str(run / 'best.pt'), 'checkpoint_sha256': sha(run / 'best.pt'),
              'test_evaluated': False}
    np.savez_compressed(run / 'validation_predictions.npz', prediction=prediction,
                        target=datasets['val'].labels.numpy(), sample_ids=datasets['val'].sample_ids)
    write_json(run / 'history.json', history)
    write_json(run / 'validation.json', record)
    LOG.info('Finished %s seed%d: val %.6f, %.1fs', name, seed, score['mirror_equivalent']['mse'], seconds)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return record


def selection_from_validation(records, plan):
    ranking = []
    for name in plan['variants']:
        runs = [r for r in records if r['variant'] == name]
        values = [r['validation']['mirror_equivalent'] for r in runs]
        ranking.append({'variant': name, 'mean_validation_mse': float(np.mean([r['mse'] for r in values])),
                        'std_validation_mse': float(np.std([r['mse'] for r in values])),
                        'length_mse': float(np.mean([r['length_mse'] for r in values])),
                        'dx_mae_nm': float(np.mean([r['mae_nm'][3] for r in values])),
                        'parameters': runs[0]['parameters']})
    ref = next(r for r in ranking if r['variant'] == 'pro_v01')
    for row in ranking:
        row['eligible'] = row['length_mse'] <= ref['length_mse'] * 1.02 and row['dx_mae_nm'] <= ref['dx_mae_nm'] * 1.05
    ranking.sort(key=lambda r: r['mean_validation_mse'])
    selected = next(r['variant'] for r in ranking if r['eligible'])
    runs = sorted([r for r in records if r['variant'] == selected], key=lambda r: r['validation']['mirror_equivalent']['mse'])
    return {'selected': selected, 'representative_seed': runs[len(runs) // 2]['seed'], 'ranking': ranking,
            'rule': plan['selection'], 'test_evaluated_at_selection': False}


def finalize(plan, output, results, device):
    selection = json.loads((output / 'selection.json').read_text(encoding='utf-8'))
    records = json.loads((output / 'validation_results.json').read_text(encoding='utf-8'))
    test = SpectrumGeometryDataset(plan['dataset'], 'test', allow_validation_data=True)
    results.mkdir(parents=True, exist_ok=True)
    scores = []
    for record in records:
        if record['variant'] not in ('pro_v01', selection['selected']):
            continue
        model, _ = load_pro(record['checkpoint'], device)
        prediction, score = evaluate(model, test, device)
        scores.append({'variant': record['variant'], 'seed': record['seed'], 'test': score})
        np.savez_compressed(results / f'{record["variant"]}_seed{record["seed"]}_test.npz',
                             prediction=prediction, target=test.labels.numpy(), sample_ids=test.sample_ids)
        del model
    chosen = next(r for r in records if r['variant'] == selection['selected'] and r['seed'] == selection['representative_seed'])
    model_path = ROOT / 'models/metaformerpro.pt'
    model_path.parent.mkdir(exist_ok=True)
    shutil.copyfile(chosen['checkpoint'], model_path)
    for name in ('plan.json', 'selection.json', 'validation_results.json'):
        shutil.copyfile(output / name, results / name)
    summary = {'selection': selection, 'evaluated': scores, 'test_policy': plan['test_policy'],
               'model_path': str(model_path), 'model_sha256': sha(model_path),
               'dataset_sha256': plan['dataset_sha256'], 'quality_scope': plan['quality_scope'],
               'fdtd_run_performed': False}
    write_json(results / 'summary.json', summary)
    write_json(output / 'status.json', {'state': 'completed', 'selected': selection['selected'],
                                      'model': str(model_path), 'results': str(results)})
    LOG.info('Selected MetaformerPro: %s seed%d', selection['selected'], selection['representative_seed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'finalize', 'all'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/metaformerpro_v02.json')
    args = parser.parse_args()
    config = args.config.resolve()
    plan = json.loads(config.read_text(encoding='utf-8'))
    for name in ('dataset', 'output_dir', 'results_dir', 'reference_results'):
        plan[name] = str((config.parent / plan[name]).resolve())
    plan['dataset_sha256'] = sha(plan['dataset'])
    plan['source_sha256'] = {f: sha(ROOT / f) for f in ('metaformerpro.py', 'train_metaformerpro.py', 'spectral_inverse.py')}
    plan['runtime'] = {'torch': str(torch.__version__), 'numpy': str(np.__version__)}
    output, results = Path(plan['output_dir']), Path(plan['results_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists() and json.loads((output / 'plan.json').read_text(encoding='utf-8')) != plan:
        raise ValueError('Use a new output_dir when changing the experiment')
    write_json(output / 'plan.json', plan)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'study.log', encoding='utf-8')])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info('MetaformerPro device=%s; dataset=%s', device, plan['dataset_sha256'])
    if args.command in ('train', 'all'):
        with np.load(plan['dataset'], allow_pickle=False) as data:
            validate_training_splits(data)
        datasets = {s: SpectrumGeometryDataset(plan['dataset'], s, allow_validation_data=True) for s in ('train', 'val')}
        records = []
        for name, settings in plan['variants'].items():
            for seed in plan['seeds']:
                records.append(train_one(name, settings, seed, plan, datasets, output, device))
                write_json(output / 'validation_results.json', records)
        selection = selection_from_validation(records, plan)
        write_json(output / 'selection.json', selection)
        write_json(output / 'status.json', {'state': 'validation_complete', **selection})
    if args.command in ('finalize', 'all'):
        finalize(plan, output, results, device)


if __name__ == '__main__':
    main()
