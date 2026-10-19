"""Show the inverse-only comparison selected using validation data."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def plot_results(directory, show=False, save=True):
    directory = Path(directory)
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    winner = report['selection']['selected_variant']
    names = list(report['plan']['variants'])
    labels = {'baseline': '当前基线', 'baseline_standard': '基线 + 输入归一化',
              'paper_style': '原文对齐设置', 'paper_standard': '原文设置 + 归一化',
              'paper_standard_fine': '原文设置 + 归一化 + 200块'}
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    ax = axes[0, 0]
    for i, name in enumerate(names):
        values = [t['best_val_mse'] for t in report['trials'] if t['variant'] == name]
        ax.bar(i, np.mean(values), color='#176b91' if name == winner else '#b0bdc5', alpha=.8)
        ax.scatter(i + np.linspace(-.12, .12, len(values)), values, s=25, color='#293c45', zorder=3)
    ax.set_xticks(range(len(names)), [labels[n] for n in names], rotation=16, ha='right', fontsize=8)
    ax.set_title('按三个种子的平均验证误差选配置')
    ax.set_ylabel('周期归一化结构 MSE（越低越好）')
    names_test = list(dict.fromkeys(['baseline', winner]))
    ax = axes[0, 1]
    for i, name in enumerate(names_test):
        values = [t['metrics']['normalized_mse'] for t in report['test_seed_metrics'] if t['variant'] == name]
        ax.bar(i, np.mean(values), color='#849aa6' if name == 'baseline' else '#176b91', alpha=.8)
        ax.scatter(i + np.linspace(-.12, .12, len(values)), values, color='#293c45', s=30, zorder=3)
    ax.set_xticks(range(len(names_test)), [labels[n] for n in names_test])
    ax.set_title('选定后评估同一测试集（10条，三个种子）')
    ax.set_ylabel('周期归一化结构 MSE')
    for ax, key, title, ylabel in [(axes[1, 0], 'mae_nm', '五个长度的测试误差', '平均绝对误差 / nm'),
                                    (axes[1, 1], 'mae_deg', '两个旋转角的测试误差（90°周期）', '平均绝对误差 / °')]:
        parameters = list(report['test_summary'][names_test[0]][key])
        x = np.arange(len(parameters))
        for i, name in enumerate(names_test):
            ax.bar(x + (i - (len(names_test) - 1) / 2) * .35,
                   [report['test_summary'][name][key][p] for p in parameters],
                   width=.34, label=labels[name], color='#849aa6' if name == 'baseline' else '#176b91')
        ax.set_xticks(x, parameters, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('逆向 Metaformer 配置优化｜100条数据，固定80/10/10划分', fontsize=15)
    fig.text(.5, -.015, '探索性对照；测试集曾用于此前实验。结构回归误差不代表重新仿真的光谱一致性。',
             ha='center', fontsize=9, color='#556570')
    if save:
        fig.savefig(directory / 'comparison.png', dpi=170, bbox_inches='tight')
        fig.savefig(directory / 'comparison.svg', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/inverse_au100_ablation')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    plot_results(args.results, args.show, not args.no_save)


if __name__ == '__main__':
    main()
