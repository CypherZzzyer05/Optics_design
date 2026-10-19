"""Controlled inverse-only ablation; choose settings on validation before testing."""
import argparse
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fdtd_pipeline.pipeline import checksum
from metaformer import InverseMetaformer, MetaformerConfig
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import (TrainingConfig, evaluate, export_predictions, normalized_errors,
                             regression_metrics, train)

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def variants():
    base = MetaformerConfig(geometry_dim=7, angle_symmetry='square_periodic')
    paper = replace(base, activation='relu', norm_order='post')
    return {
        'baseline': (base, 'adamw', .001),
        'baseline_standard': (replace(base, input_normalization='train_standard'), 'adamw', .001),
        'paper_style': (paper, 'adam', 0.),
        'paper_standard': (replace(paper, input_normalization='train_standard'), 'adam', 0.),
        'paper_standard_fine': (replace(paper, input_normalization='train_standard', patch_count=200), 'adam', 0.),
    }


def experiment_plan(dataset):
    return {
        'dataset_sha256': checksum(dataset), 'seeds': list(SEEDS),
        'selection': 'Lowest mean best-validation periodic normalized geometry MSE across seeds',
        'representative_checkpoint': 'Selected variant seed with median validation MSE; never chosen by test',
        'epochs': 100, 'early_stopping_patience': 20, 'learning_rate': .0003, 'batch_size': 8,
        'normalization': 'Training-only wavelength-wise mean/std; fixed std floor=0.02 reflectance',
        'test_policy': 'No trial evaluates test; after configuration selection evaluate baseline and winner only',
        'variants': {name: {'model': asdict(model), 'optimizer': optimizer, 'weight_decay': decay}
                     for name, (model, optimizer, decay) in variants().items()},
        'source_sha256': {name: checksum(ROOT / name) for name in
                          ('metaformer.py', 'train_metaformer.py', 'spectrum_dataset.py', 'optimize_inverse.py')},
    }


def run_experiment(dataset, work, output, device='auto'):
    dataset, work, output = dataset.resolve(), work.resolve(), output.resolve()
    torch.set_num_threads(4)
    work.mkdir(parents=True, exist_ok=True)
    plan = experiment_plan(dataset)
    plan_path = work / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Existing plan differs: use a new work directory')
    else:
        write_json(plan_path, plan)  # Written before the first optimizer update.
    all_trials = []
    logging.getLogger('train_metaformer').setLevel(logging.INFO)
    logging.getLogger('train_metaformer').propagate = False
    for name, (model, optimizer, decay) in variants().items():
        for seed in SEEDS:
            trial_root = work / name / f'seed{seed}'
            pointer = trial_root / 'latest_run.json'
            resume = None
            if pointer.exists():
                previous = json.loads(pointer.read_text(encoding='utf-8'))
                run_dir = Path(previous['run_dir'])
                if previous['state'] == 'completed':
                    report = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
                else:
                    resume = run_dir
            if not pointer.exists() or resume is not None:
                print(f'Training {name}, seed {seed}', flush=True)
                config = TrainingConfig(dataset, trial_root, allow_validation_data=True, seed=seed,
                    device=device, epochs=100, early_stopping_patience=20, batch_size=8,
                    learning_rate=.0003, weight_decay=decay, optimizer=optimizer,
                    validation_only=True, model=model)
                run_dir, report = train(config, resume=resume)
            if 'test' in report['metrics'] or (run_dir / 'test_predictions.csv').exists():
                raise ValueError('A trial unexpectedly evaluated test data')
            trial = {'variant': name, 'seed': seed, 'run_dir': str(run_dir),
                     'model_parameters': report['model_parameters'], 'best_epoch': report['best_epoch'],
                     'epochs_completed': report['epochs_completed'], 'best_val_mse': report['best_val_mse'],
                     'training_seconds': report['training_seconds'], 'optimizer_steps': report['optimizer_steps'],
                     'validation_metrics': report['metrics']['val'],
                     'best_checkpoint_sha256': report['best_checkpoint_sha256']}
            all_trials.append(trial)
            write_json(work / 'progress.json', all_trials)
            print(f"Finished {name}/{seed}: val={trial['best_val_mse']:.6f}, "
                  f"best epoch={trial['best_epoch']}, {trial['training_seconds']:.1f}s", flush=True)

    ranking = sorted(({'variant': name,
        'mean_validation_mse': float(np.mean([t['best_val_mse'] for t in all_trials if t['variant'] == name])),
        'std_validation_mse': float(np.std([t['best_val_mse'] for t in all_trials if t['variant'] == name], ddof=1))}
        for name in variants()), key=lambda row: row['mean_validation_mse'])
    winner = ranking[0]['variant']
    selected_trials = sorted((t for t in all_trials if t['variant'] == winner), key=lambda t: t['best_val_mse'])
    representative = selected_trials[len(selected_trials) // 2]
    selection = {'selected_variant': winner, 'ranking': ranking,
                 'representative_seed': representative['seed'], 'test_evaluated': False}
    write_json(work / 'selection_before_test.json', selection)
    print(f'Validation selected {winner}; now evaluating held-out baseline and winner.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    test_dataset = SpectrumGeometryDataset(dataset, split='test', allow_validation_data=True)
    evaluations, errors, history = [], {}, {}
    for trial in all_trials:
        if trial['variant'] not in {'baseline', winner}:
            continue
        checkpoint = torch.load(Path(trial['run_dir']) / 'best.pt', map_location='cpu', weights_only=True)
        config = MetaformerConfig.from_dict(checkpoint['metadata']['config']['model'])
        model = InverseMetaformer(config)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        _, prediction, target = evaluate(model, DataLoader(test_dataset, batch_size=8), torch.device('cpu'))
        metrics = regression_metrics(prediction, target, test_dataset.lower_values, test_dataset.upper_values,
                                     test_dataset.parameter_names, config.angle_symmetry)
        evaluations.append({'variant': trial['variant'], 'seed': trial['seed'], 'metrics': metrics})
        errors.setdefault(trial['variant'], []).append(normalized_errors(prediction, target, config.angle_symmetry) ** 2)
        export_predictions(output / f"test_{trial['variant']}_seed{trial['seed']}.csv", test_dataset, prediction, target)
        history[f"{trial['variant']}_seed{trial['seed']}"] = json.loads(
            (Path(trial['run_dir']) / 'history.json').read_text(encoding='utf-8'))
    test_summary = {}
    for name in {'baseline', winner}:
        rows = [e['metrics'] for e in evaluations if e['variant'] == name]
        test_summary[name] = {
            'mean_mse': float(np.mean([r['normalized_mse'] for r in rows])),
            'std_mse': float(np.std([r['normalized_mse'] for r in rows], ddof=1)),
            'mae_nm': {p: float(np.mean([r['mae_nm'][p] for r in rows])) for p in rows[0]['mae_nm']},
            'mae_deg': {p: float(np.mean([r['mae_deg'][p] for r in rows])) for p in rows[0]['mae_deg']},
        }
    baseline_error = np.mean(errors['baseline'], axis=0).mean(axis=1)
    selected_error = np.mean(errors[winner], axis=0).mean(axis=1)
    differences = selected_error - baseline_error
    rng = np.random.default_rng(20261019)
    bootstrap = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
    summary = {
        'plan': plan, 'selection': selection, 'trials': all_trials,
        'test_summary': test_summary, 'test_seed_metrics': evaluations,
        'paired_test_mse_difference_selected_minus_baseline': float(differences.mean()),
        'exploratory_paired_bootstrap_95_ci': np.quantile(bootstrap, [.025, .975]).tolist(),
        'bootstrap_unit': '10 test sample IDs; average three seeds per ID before resampling',
        'dataset_quality': 'pipeline_validation; historical metal material-fit limitations remain',
        'test_limit': 'Same previously inspected 10 test samples, not an independent fresh blind test',
        'inference_limit': 'Geometry error only; no forward spectral or FDTD validation performed',
        'split_sample_ids': torch.load(Path(representative['run_dir']) / 'best.pt', weights_only=True,
                                      map_location='cpu')['metadata']['split_sample_ids'],
    }
    write_json(output / 'summary.json', summary)
    write_json(output / 'histories.json', history)
    model_path = ROOT / 'models' / 'inverse_au100_selected.pt'
    model_path.parent.mkdir(exist_ok=True)
    shutil.copy2(Path(representative['run_dir']) / 'best.pt', model_path)
    write_json(output / 'checkpoint.json', {'path': str(model_path.relative_to(ROOT)).replace('\\', '/'),
        'sha256': checksum(model_path), 'variant': winner, 'seed': representative['seed'],
        'rule': plan['representative_checkpoint']})
    model_config, optimizer, decay = variants()[winner]
    portable_config = TrainingConfig(Path('../examples/au100_metal8_100/dataset.npz'),
        Path('../runs/inverse_selected'), allow_validation_data=True, epochs=100,
        early_stopping_patience=20, model=model_config, optimizer=optimizer, weight_decay=decay,
        seed=representative['seed']).metadata()
    write_json(ROOT / 'configs' / 'train_inverse_selected.json', portable_config)
    print(json.dumps({'selected': winner, 'test_summary': test_summary,
                      'result_dir': str(output)}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'examples/au100_metal8_100/dataset.npz')
    parser.add_argument('--work', type=Path, default=ROOT / 'runs/inverse_ablation')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/inverse_au100_ablation')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    run_experiment(args.dataset, args.work, args.output, args.device)


if __name__ == '__main__':
    main()
