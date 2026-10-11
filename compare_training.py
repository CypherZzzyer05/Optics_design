"""Compare saved MetaFormer and ResNet-front-end runs on the same fixed data splits."""
import argparse
import json
from pathlib import Path

import numpy as np

from fdtd_pipeline.config import PARAMETER_NAMES, PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json, utc_now


def read_latest(root):
    return Path(json.loads((Path(root) / 'latest_run.json').read_text(encoding='utf-8'))['run_dir'])


def compare_runs(baseline_dir, candidate_dir, show=True):
    baseline_dir, candidate_dir = Path(baseline_dir).resolve(), Path(candidate_dir).resolve()
    reports = [json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
               for directory in (baseline_dir, candidate_dir)]
    baseline, candidate = reports
    if baseline['dataset_sha256'] != candidate['dataset_sha256'] or baseline['split_sample_ids'] != candidate['split_sample_ids']:
        raise ValueError('Comparison requires identical dataset and sample splits')
    if baseline.get('parameter_names') != candidate.get('parameter_names'):
        raise ValueError('Comparison requires identical parameter order and units')
    settings = []
    model_settings = []
    for report in reports:
        config = dict(report['config'])
        config.pop('output_dir')
        model = dict(config.pop('model'))
        model.setdefault('angle_symmetry', 'none')
        for key in ('spectral_frontend', 'resnet_channels', 'resnet_blocks', 'resnet_kernel_size'):
            model.pop(key, None)
        settings.append(config)
        model_settings.append(model)
    if settings[0] != settings[1] or model_settings[0] != model_settings[1]:
        raise ValueError('Training settings and Transformer dimensions differ; comparison is not controlled')
    summary = {'compared_at': utc_now(), 'baseline_run': str(baseline_dir), 'candidate_run': str(candidate_dir),
               'dataset_sha256': baseline['dataset_sha256'], 'same_splits_and_training_settings': True,
               'split_sizes': {s: len(ids) for s, ids in baseline['split_sample_ids'].items()},
               'epochs': {'baseline': baseline['epochs_completed'], 'resnet': candidate['epochs_completed']},
               'best_epochs': {'baseline': baseline['best_epoch'], 'resnet': candidate['best_epoch']},
               'metrics': {}, 'test_mae_nm': {},
               'evaluation_note': 'Same previously evaluated test split; comparison does not establish new-data generalization'}
    for split in ('train', 'val', 'test'):
        before, after = [r['metrics'][split]['normalized_mse'] for r in reports]
        summary['metrics'][split] = {'baseline_mse': before, 'resnet_mse': after,
                                    'relative_mse_reduction': 1 - after / before if before > 0 else None}
    for name in PARAMETER_NAMES:
        before, after = [r['metrics']['test']['mae_nm'][name] for r in reports]
        summary['test_mae_nm'][name] = {'baseline': before, 'resnet': after, 'change_nm': after - before}
    if len(baseline.get('parameter_names', PARAMETER_NAMES)) == 6:
        before, after = [r['metrics']['test']['mae_deg']['rotation_deg'] for r in reports]
        summary['test_mae_deg'] = {'rotation_deg': {'baseline': before, 'resnet': after, 'change_deg': after - before}}
    atomic_json(candidate_dir / 'comparison.json', summary)

    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='#f5f7fb')
    fig.canvas.manager.set_window_title('MetaFormer | 残差网络对比')
    fig.subplots_adjust(left=.08, right=.95, top=.85, bottom=.09, hspace=.4, wspace=.3)
    sizes = summary['split_sizes']
    fig.suptitle(f"一维 ResNet 前端 · 同一批 {sum(sizes.values())} 个样本的训练对比", x=.08, ha='left', y=.965,
                 fontsize=18, fontweight='bold', color='#152849')
    fig.text(.08, .91, f"数据 SHA256、{sizes['train']}/{sizes['val']}/{sizes['test']} 划分、Transformer 维度和超参数相同；各自按验证集选模型。", color='#50617c')
    names = ['原 MetaFormer', 'ResNet + MetaFormer']
    for ax, split, title in zip(axes[0], ('val', 'test'), ('验证集 MSE', '原留出测试集 MSE')):
        values = [r['metrics'][split]['normalized_mse'] for r in reports]
        bars = ax.bar(names, values, color=['#9aaecb', '#2667d8'], width=.5)
        ax.set(title=title, ylabel='归一化参数 MSE', ylim=(0, max(values) * 1.3))
        ax.grid(axis='y', alpha=.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value * 1.03, f'{value:.6f}', ha='center')
    ax = axes[1, 0]
    x = np.arange(5)
    ax.bar(x - .18, [baseline['metrics']['test']['mae_nm'][p] for p in PARAMETER_NAMES], .35,
           color='#9aaecb', label='原模型')
    ax.bar(x + .18, [candidate['metrics']['test']['mae_nm'][p] for p in PARAMETER_NAMES], .35,
           color='#2667d8', label='ResNet 前端')
    ax.set(title='各参数测试绝对误差', ylabel='MAE (nm)', xticks=x, xticklabels=PARAMETER_NAMES)
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    axes[1, 1].axis('off')
    v = summary['metrics']['val']['relative_mse_reduction']
    t = summary['metrics']['test']['relative_mse_reduction']
    text = (f"实测对比\n\n验证 MSE 相对变化：降低 {v*100:.1f}%\n测试 MSE 相对变化：降低 {t*100:.1f}%\n"
            f"最佳轮次：原模型 {baseline['best_epoch']} / ResNet {candidate['best_epoch']}\n"
            f"训练轮次：原模型 {baseline['epochs_completed']} / ResNet {candidate['epochs_completed']}\n\n"
            f"本次使用同一批已评估过的 {sizes['test']} 个测试样本。\n"
            '数据为粗网格流程验证数据。\n'
            '新增残差网络的效果以逐参数误差为准。\n\n'
            '原模型和新模型权重均已保留。')
    if 'test_mae_deg' in summary:
        angle = summary['test_mae_deg']['rotation_deg']
        text += f"\n\n等效角 MAE（deg）：{angle['baseline']:.2f} → {angle['resnet']:.2f}"
    axes[1, 1].text(0, 1, text, va='top', color='#334969', linespacing=1.5)
    output = candidate_dir / 'comparison.png'
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Comparison preview: {output}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    baseline = args.baseline or read_latest(PROJECT_ROOT / 'runs/metaformer_smoke_100')
    candidate = args.candidate or read_latest(PROJECT_ROOT / 'runs/metaformer_resnet_100')
    compare_runs(baseline, candidate, show=not args.no_show)


if __name__ == '__main__':
    main()
