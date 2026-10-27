"""A bounded, validation-selected refinement of the new top-metal inverse model."""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import torch

from fdtd_pipeline.config import Geometry
from metaformerpro import MetaformerProConfig
from optimize_spectral_inverse import evaluate, sha, write_json
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import validate_training_splits
from train_metaformerpro import load_pro, train_one

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


def baseline_runs(plan, datasets, device):
    original = json.loads(Path(plan['baseline_results']).read_text(encoding='utf-8'))
    records = []
    for seed in plan['seeds']:
        row = next(r for r in original if r['variant'] == 'circular' and r['seed'] == seed)
        checkpoint = plan['baseline_checkpoints'][str(seed)]
        model, saved = load_pro(checkpoint, device)
        if saved['dataset_sha256'] != plan['dataset_sha256']:
            raise ValueError('Baseline uses a different dataset')
        if saved['normalization_fit_sample_ids'] != datasets['train'].sample_ids.tolist():
            raise ValueError('Baseline has different normalization training IDs')
        prediction, score = evaluate(model, datasets['val'], device)
        archive = Path(plan['baseline_archive']) / f'circular_seed{seed}'
        with np.load(archive/'validation_predictions.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['sample_ids'], datasets['val'].sample_ids)
            np.testing.assert_allclose(prediction, data['prediction'], atol=1e-5, rtol=0)
        record = dict(row, variant='baseline', validation=score, checkpoint=checkpoint,
                      checkpoint_sha256=sha(checkpoint), reused_existing_run=True,
                      history_directory=str(archive))
        records.append(record)
        del model
    return records


def select_validation(records, rule):
    names = list(dict.fromkeys(r['variant'] for r in records))
    ranking = []
    for name in names:
        runs = [r for r in records if r['variant'] == name]
        values = [r['validation']['mirror_equivalent'] for r in runs]
        ranking.append({'variant': name,
            'mean_validation_mse': float(np.mean([v['mse'] for v in values])),
            'std_validation_mse': float(np.std([v['mse'] for v in values])),
            'length_mse': float(np.mean([v['length_mse'] for v in values])),
            'mae_nm': np.mean([v['mae_nm'] for v in values], axis=0).tolist(),
            'mae_deg': np.mean([v['mae_deg'] for v in values], axis=0).tolist(),
            'parameters': runs[0]['parameters']})
    baseline = next(r for r in ranking if r['variant'] == 'baseline')
    for row in ranking:
        row['eligible'] = (row['length_mse'] <= baseline['length_mse']*1.02
                           and row['mae_nm'][3] <= baseline['mae_nm'][3]*1.05
                           and row['mae_nm'][4] <= baseline['mae_nm'][4]*1.02)
        row['replacement_threshold_passed'] = row['mean_validation_mse'] <= baseline['mean_validation_mse']*.99
    candidates = [r for r in ranking if r['eligible'] and r['replacement_threshold_passed']]
    chosen = min(candidates, key=lambda r:r['mean_validation_mse']) if candidates else baseline
    runs = sorted([r for r in records if r['variant'] == chosen['variant']],
                  key=lambda r:r['validation']['mirror_equivalent']['mse'])
    return {'selected': chosen['variant'], 'representative_seed': runs[len(runs)//2]['seed'],
            'ranking': sorted(ranking, key=lambda r:r['mean_validation_mse']), 'rule': rule,
            'test_evaluated_at_selection': False}


def finalize(plan, records, selection, device):
    results = Path(plan['results_dir'])
    test = SpectrumGeometryDataset(plan['dataset'], 'test', allow_validation_data=True)
    train = SpectrumGeometryDataset(plan['dataset'], 'train', allow_validation_data=True)
    selected = selection['selected']
    scores, geometry = [], {}
    reload_difference = None
    for row in records:
        name = f'{row["variant"]}_seed{row["seed"]}'
        history_dir = Path(row.get('history_directory', Path(row['checkpoint']).parent))
        archive = results/'validation_runs'/name
        archive.mkdir(parents=True, exist_ok=True)
        for filename in ('history.json', 'validation_predictions.npz'):
            shutil.copyfile(history_dir/filename, archive/filename)
        if row['variant'] not in ('baseline', selected):
            continue
        model, saved = load_pro(row['checkpoint'], device)
        if saved['dataset_sha256'] != plan['dataset_sha256']:
            raise ValueError('Selected model uses a different dataset')
        torch.testing.assert_close(model.normalizer.mean.cpu(), train.spectra[:, None].mean(0, keepdim=True),
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(model.normalizer.scale.cpu(),
            train.spectra[:, None].std(0, unbiased=False, keepdim=True).clamp_min(model.config.normalization_floor),
            rtol=1e-6, atol=1e-6)
        prediction, score = evaluate(model, test, device)
        np.savez_compressed(results/f'{name}_test.npz', prediction=prediction,
                            target=test.labels.numpy(), sample_ids=test.sample_ids)
        scores.append({'variant': row['variant'], 'seed': row['seed'], 'test': score})
        failures = []
        for sample_id, values in zip(test.sample_ids,
                          test.lower_values + prediction*(test.upper_values-test.lower_values)):
            try:
                Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
            except ValueError as error:
                failures.append({'sample_id': int(sample_id), 'error': str(error)})
        geometry[name] = {'valid': len(test)-len(failures), 'total': len(test), 'failures': failures}
        if row['variant'] == selected:
            prefix = plan.get('seed_model_prefix', 'metaformerpro_refine_seed')
            destination = ROOT/f'models/{prefix}{row["seed"]}.pt'
            shutil.copyfile(row['checkpoint'], destination)
        if row['variant'] == selected and row['seed'] == selection['representative_seed']:
            cpu, _ = evaluate(model.to('cpu'), test, torch.device('cpu'))
            reload_difference = float(np.max(np.abs(cpu-prediction)))
            if reload_difference >= 1e-5:
                raise ValueError('CPU inference differs from the stored test prediction')
            shutil.copyfile(row['checkpoint'], plan['model_output'])
        del model
    means = {}
    for name in dict.fromkeys(r['variant'] for r in scores):
        runs = [r['test'] for r in scores if r['variant'] == name]
        means[name] = {mode: {key: np.mean([s[mode][key] for s in runs], axis=0).tolist()
                   for key in ('mse', 'length_mse', 'angle_mse', 'mae_nm', 'mae_deg')}
                   for mode in ('periodic', 'mirror_equivalent')}
    write_json(results/'test_metrics.json', {'runs': scores, 'mean': means})
    write_json(results/'verification.json', {'normalization_training_only': True,
        'cpu_prediction_max_difference': reload_difference, 'geometry': geometry,
        'existing_test_set_reused': True, 'new_fdtd_run_performed': False})
    write_json(results/'summary.json', {'selection': selection, 'test_mean': means,
        'dataset_sha256': plan['dataset_sha256'], 'model_path': plan['model_output'],
        'model_sha256': sha(plan['model_output']), 'quality_scope': plan['quality_scope'],
        'test_policy': plan['test_policy'], 'new_fdtd_run_performed': False})
    shutil.copyfile(Path(plan['output_dir'])/'study.log', results/'training.log')
    write_json(Path(plan['output_dir'])/'status.json', {'state': 'completed', 'selected': selected,
               'model': plan['model_output'], 'results': str(results)})
    LOG.info('Frozen selection: %s seed%s; results=%s', selected, selection['representative_seed'], results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'finalize', 'all'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/metaformerpro_topmetal_refine.json')
    args = parser.parse_args()
    config = args.config.resolve()
    plan = json.loads(config.read_text(encoding='utf-8'))
    for key in ('dataset', 'output_dir', 'results_dir', 'model_output', 'baseline_results', 'baseline_archive'):
        plan[key] = str((config.parent/plan[key]).resolve())
    plan['baseline_checkpoints'] = {k:str((config.parent/v).resolve()) for k,v in plan['baseline_checkpoints'].items()}
    for settings in plan['variants'].values():
        MetaformerProConfig(**settings)
    plan['dataset_sha256'] = sha(plan['dataset'])
    plan['source_sha256'] = {name:sha(ROOT/name) for name in
                           ('metaformerpro.py', 'train_metaformerpro.py', 'train_metaformerpro_refine.py', 'spectral_inverse.py')}
    if any(s.get('sampling', 'uniform') != 'uniform' for s in plan['variants'].values()):
        plan['source_sha256']['resonance_sampling.py'] = sha(ROOT/'resonance_sampling.py')
    plan['runtime'] = {'torch': str(torch.__version__), 'numpy': str(np.__version__)}
    output, results = Path(plan['output_dir']), Path(plan['results_dir'])
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (ROOT/'models').mkdir(exist_ok=True)
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text(encoding='utf-8')) != plan:
        raise ValueError('Use a new output_dir when changing the experiment')
    write_json(output/'plan.json', plan)
    write_json(results/'plan.json', plan)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(output/'study.log', encoding='utf-8')])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info('Top-metal refinement device=%s; data=%s', device, plan['dataset_sha256'])
    if args.command in ('train', 'all'):
        with np.load(plan['dataset'], allow_pickle=False) as data:
            validate_training_splits(data)
        datasets = {s:SpectrumGeometryDataset(plan['dataset'], s, allow_validation_data=True) for s in ('train','val')}
        records = baseline_runs(plan, datasets, device)
        for name, settings in plan['variants'].items():
            for seed in plan['seeds']:
                records.append(train_one(name, settings, seed, plan, datasets, output, device))
                write_json(results/'validation_results.json', records)
        selection = select_validation(records, plan['selection'])
        if set(plan['variants']) == {'uniform_extra', 'resonance'}:
            comparison = {r['variant']: r for r in selection['ranking']}
            uniform, local = comparison['uniform_extra'], comparison['resonance']
            selection['resonance_vs_uniform'] = {
                'relative_validation_mse_change': local['mean_validation_mse']/uniform['mean_validation_mse']-1,
                'mechanism_threshold_passed': bool(
                    local['mean_validation_mse'] <= uniform['mean_validation_mse']*.99
                    and local['length_mse'] <= uniform['length_mse']*1.02
                    and local['mae_nm'][3] <= uniform['mae_nm'][3]*1.05
                    and local['mae_nm'][4] <= uniform['mae_nm'][4]*1.02)}
        write_json(results/'selection.json', selection)
        write_json(output/'status.json', {'state': 'validation_complete', **selection})
        LOG.info('Validation selection frozen: %s seed%d', selection['selected'], selection['representative_seed'])
    else:
        records = json.loads((results/'validation_results.json').read_text(encoding='utf-8'))
        selection = json.loads((results/'selection.json').read_text(encoding='utf-8'))
    if args.command in ('all', 'finalize'):
        finalize(plan, records, selection, device)


if __name__ == '__main__':
    main()
