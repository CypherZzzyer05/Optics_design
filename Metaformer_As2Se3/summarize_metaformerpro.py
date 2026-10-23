"""Publish MetaformerPro ablations, parameter errors and checkpoint checks."""
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from fdtd_pipeline.config import Geometry
from optimize_spectral_inverse import metrics, sha, write_json
from train_metaformerpro import load_pro

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    directory = ROOT / 'results/metaformerpro_v02'
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    trials = json.loads((directory / 'validation_results.json').read_text(encoding='utf-8'))
    selection = summary['selection']
    selected, seed = selection['selected'], selection['representative_seed']
    selected_run = next(r for r in trials if r['variant'] == selected and r['seed'] == seed)
    baseline_runs = sorted([r for r in trials if r['variant'] == 'pro_v01'],
                            key=lambda r: r['validation']['mirror_equivalent']['mse'])
    baseline_run = baseline_runs[1]
    assert sha(plan['dataset']) == summary['dataset_sha256']
    with np.load(plan['dataset'], allow_pickle=False) as data:
        x = torch.from_numpy(data['spectra'][data['test_indices']].astype(np.float32))
        training = torch.from_numpy(data['spectra'][data['train_indices']].astype(np.float32))
        lower, upper = data['lower_values'].copy(), data['upper_values'].copy()
        training_ids = data['sample_ids'][data['train_indices']].tolist()
    model, checkpoint = load_pro(ROOT / 'models/metaformerpro.pt')
    assert checkpoint['normalization_fit_sample_ids'] == training_ids
    torch.testing.assert_close(model.normalizer.mean, training[:, None].mean(0, keepdim=True), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.normalizer.scale,
                training[:, None].std(0, unbiased=False, keepdim=True).clamp_min(.02), atol=1e-6, rtol=1e-6)
    with torch.inference_mode():
        cpu = model(x).numpy()
    with np.load(directory / f'{selected}_seed{seed}_test.npz', allow_pickle=False) as data:
        reload_difference = float(np.max(np.abs(cpu - data['prediction'])))
    assert reload_difference < 1e-5
    legality, metric_difference = {}, 0.
    for result in summary['evaluated']:
        name = f'{result["variant"]}_seed{result["seed"]}'
        with np.load(directory / f'{name}_test.npz', allow_pickle=False) as data:
            p, t = data['prediction'], data['target']
        checked = metrics(p, t, upper - lower)
        metric_difference = max(metric_difference, *[abs(checked[m]['mse'] - result['test'][m]['mse'])
                                         for m in ('periodic', 'mirror_equivalent')])
        failures = []
        for index, values in enumerate(lower + p * (upper - lower)):
            try:
                Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
            except ValueError as error:
                failures.append({'test_row': index, 'error': str(error)})
        legality[name] = {'valid': len(p) - len(failures), 'total': len(p), 'failures': failures}
    assert metric_difference < 1e-12
    times = {}
    for name, path in [('selected', ROOT / 'models/metaformerpro.pt'), ('pro_v01', Path(baseline_run['checkpoint']))]:
        timed, _ = load_pro(path)
        with torch.inference_mode():
            for _ in range(10):
                timed(x[:1])
            durations = []
            for _ in range(50):
                start = time.perf_counter()
                timed(x[:1])
                durations.append((time.perf_counter() - start) * 1000)
        times[name] = float(np.median(durations))
    verification = {'reload_max_difference': reload_difference, 'metric_recalculation_max_difference': metric_difference,
                    'training_normalization_verified': True, 'cpu_batch1_median_ms': times,
                    'cpu_timing_scope': 'Warm FP32 forward including preprocessing, 4 CPU threads, excludes file I/O',
                    'geometries': legality, 'model_sha256': sha(ROOT / 'models/metaformerpro.pt')}
    write_json(directory / 'verification.json', verification)
    # Preserve actual training/validation traces; large experimental checkpoints remain local.
    for trial in trials:
        source = Path(trial['checkpoint']).parent
        destination = directory / 'validation_runs' / f'{trial["variant"]}_seed{trial["seed"]}'
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ('history.json', 'validation_predictions.npz'):
            shutil.copyfile(source / filename, destination / filename)
    shutil.copyfile(Path(plan['output_dir']) / 'study.log', directory / 'training.log')
    means = {}
    for name in dict.fromkeys(('pro_v01', selected)):
        scores = [r['test'] for r in summary['evaluated'] if r['variant'] == name]
        means[name] = {mode: {'mse': float(np.mean([s[mode]['mse'] for s in scores])),
                              'length_mse': float(np.mean([s[mode]['length_mse'] for s in scores])),
                              'angle_mse': float(np.mean([s[mode]['angle_mse'] for s in scores])),
                              'mae_nm': np.mean([s[mode]['mae_nm'] for s in scores], axis=0).tolist(),
                              'mae_deg': np.mean([s[mode]['mae_deg'] for s in scores], axis=0).tolist()}
                       for mode in ('periodic', 'mirror_equivalent')}
    write_json(directory / 'test_mean.json', means)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    names = list(plan['variants'])
    ranking = {r['variant']: r for r in selection['ranking']}
    colors = ['#64778c', '#327eae', '#248b75', '#ac687f', '#ce8d32']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(range(len(names)), [ranking[n]['mean_validation_mse'] for n in names],
                yerr=[ranking[n]['std_validation_mse'] for n in names], color=colors, capsize=4)
    axes[0].set(ylabel='镜像等价验证 MSE，三种子均值 ± 标准差', title='统一指标选择；尺寸门槛同时生效')
    axes[1].bar(range(len(names)), [ranking[n]['dx_mae_nm'] for n in names], color=colors)
    threshold = ranking['pro_v01']['dx_mae_nm'] * 1.05
    axes[1].axhline(threshold, color='#9d3434', ls='--', label='dX 允许上限：昨日基线 ×1.05')
    axes[1].set_ylim(0, max(threshold, max(ranking[n]['dx_mae_nm'] for n in names)) * 1.22)
    axes[1].set(ylabel='验证集 dX 平均绝对误差 / nm', title='避免角度收益掩盖尺寸退步')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(range(len(names)), names, rotation=20, ha='right')
    fig.suptitle('MetaformerPro：固定划分，测试前冻结方案')
    fig.savefig(directory / 'validation_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for record, color, label in [(baseline_run, colors[0], 'MetaformerPro 0.1'), (selected_run, colors[1], selected)]:
        history = json.loads((Path(record['checkpoint']).parent / 'history.json').read_text(encoding='utf-8'))
        epochs = [h['epoch'] for h in history]
        axes[0].plot(epochs, [h['train_mse'] for h in history], color=color, label=label)
        axes[1].plot(epochs, [h['validation']['mirror_equivalent']['mse'] for h in history], color=color, label=label)
    for ax in axes:
        ax.set(xlabel='轮次', ylabel='联合镜像周期 MSE')
        ax.legend()
    axes[0].set_title('训练损失：相同定义')
    axes[1].set_title('验证损失')
    fig.savefig(directory / 'training_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    old, new = means['pro_v01']['mirror_equivalent'], means[selected]['mirror_equivalent']
    for ax, labels, key in [(axes[0], ['P', 'L', 'H', 'dX', 'dY'], 'mae_nm'),
                            (axes[1], ['奇数象限角', '偶数象限角'], 'mae_deg')]:
        positions = np.arange(len(labels))
        ax.bar(positions - .18, old[key], .36, color=colors[0], label='MetaformerPro 0.1')
        ax.bar(positions + .18, new[key], .36, color=colors[1], label='本轮选中方案')
        ax.set_xticks(positions, labels)
        ax.legend(fontsize=9)
    axes[0].set(ylabel='MAE / nm', title='尺寸：三个种子的平均测试误差')
    axes[1].set(ylabel='联合镜像匹配后的 MAE / °', title='角度：同一评价口径')
    axes[1].set_ylim(0, max(old['mae_deg'] + new['mae_deg']) * 1.25)
    fig.savefig(directory / 'parameter_errors.png', dpi=150)
    plt.close(fig)
    validation_rows = '\n'.join(f"| {r['variant']} | {r['parameters']:,} | {r['mean_validation_mse']:.6f} ± "
                        f"{r['std_validation_mse']:.6f} | {r['length_mse']:.6f} | {r['dx_mae_nm']:.3f} | "
                        f"{'通过' if r['eligible'] else '未通过'} |" for r in selection['ranking'])
    test_rows = '\n'.join(f"| {name} | {value['periodic']['mse']:.6f} | {value['mirror_equivalent']['mse']:.6f} | "
                           f"{value['mirror_equivalent']['length_mse']:.6f} |" for name, value in means.items())
    parameter_rows = '\n'.join(f'| {label} | {a:.3f} | {b:.3f} | {unit} |' for label, a, b, unit in
                          zip(['P', 'L', 'H', 'dX', 'dY', '奇数角', '偶数角'], old['mae_nm'] + old['mae_deg'],
                              new['mae_nm'] + new['mae_deg'], ['nm'] * 5 + ['°'] * 2))
    ratio = 1 - new['mse'] / old['mse']
    change = f'{"降低" if ratio >= 0 else "升高"} {abs(ratio)*100:.1f}%'
    original_ratio = 1 - means[selected]['periodic']['mse'] / means['pro_v01']['periodic']['mse']
    original_change = f'{"降低" if original_ratio >= 0 else "升高"} {abs(original_ratio)*100:.1f}%'
    size_ratio = 1 - new['length_mse'] / old['length_mse']
    size_change = f'{"降低" if size_ratio >= 0 else "升高"} {abs(size_ratio)*100:.1f}%'
    deployed = legality[f'{selected}_seed{seed}']
    settings = plan['variants'][selected]
    latent_label = '5个长度logit及2组圆周向量' if settings['angle_head'] == 'circular' else '7个Sigmoid输出'
    pool_label = '7个参数查询' if settings['readout'] == 'queries' else '学习式全局池化'
    report = f'''# MetaformerPro：圆周角度、参数查询与重叠分块对照

日期：2026-10-23。昨天选中的 compact_raw_mirror 命名为 **MetaformerPro 0.1**，本轮在独立分支 `codex/metaformerpro` 继续优化。原项目、原始数据和历史权重保留。

**验证规则选中 `{selected}`，代表 seed `{seed}`。三个种子平均测试的五尺寸 MSE 相对昨日基线{size_change}，原始标签周期 MSE {original_change}，镜像等价 MSE {change}。** 所有口径同时保留，不把评价口径变化当成性能提升，也不把尺寸改进称为全面改善。

## 数据和选择规则

沿用 1000 条历史反射光谱，700–1700 nm、1000 点，固定 800/100/100 划分；数据 SHA256 `{summary['dataset_sha256']}`。只训练逆向网络，没有新增正向网络或 FDTD 求解。已知这批 1.0 数据的 Au 损耗拟合存在严重偏差，本轮评价数值数据上的学习能力。

每个新方案 seed42/43/44，batch8、AdamW、lr=3e-4、weight_decay=1e-3，最多100轮，20轮停滞早停。标准化仅拟合训练集。三个昨日基线直接复用，转换后预测差记录于 `validation_results.json`，避免重训波动混入改进。

预先确定：先要求五尺寸验证 MSE 不超过昨日均值的1.02倍、dX验证MAE不超过昨日的1.05倍，再按平均联合镜像验证 MSE 选择方案；昨日基线可作为回退。`selection.json` 在本轮测试前保存。测试集此前已经检查过，因此本轮不是新的盲测。

| 方案 | 参数量 | 等价验证 MSE 均值 ± 种子标准差 | 五尺寸验证 MSE | dX验证MAE / nm | 尺寸门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
{validation_rows}

![验证对照](validation_comparison.png)

## 本轮结构改动

- pro_v01：昨天的小型网络，4层、96维、4头、每头24维，100个10点patch、学习式全局池化、7个Sigmoid输出。更名和移除未使用的FFT计算不改变其已保存预测。
- circular：仍输出7个物理参数；网络内部产生5个长度logit及两组cos/sin，共9个潜变量。令 `u=(θ+45°)/90°`，相位 `φ=2πu`，用atan2解码。最终仍按原定义的联合镜像周期MSE训练，不把损失换成数值范围更小的弦长误差。圆周向量的长度不作为置信度。
- queries：7个学习查询分别从编码token中池化特征，读出P/L/H/dX/dY和两个角，使各参数能够关注不同谱段；这类权重不是物理因果或能量分布。
- queries_circular：结合参数查询和圆周角度输出。
- overlap_queries_circular：20点窗口、10点步长、左右复制填充5点，仍产生100个token；相邻窗口重叠，保留跨分块窄谷信息，不对原始曲线做平滑或删峰。

选中配置：`{json.dumps(settings, ensure_ascii=False)}`。镜像标签变换仍为 `(u奇,u偶)→(1−u偶,1−u奇)`，两角必须共同选择同一等价分支。

```mermaid
flowchart LR
    A[1000点反射光谱] --> B[训练集统计标准化]
    B --> C[100个96维token]
    C --> D[4层Transformer编码器]
    D --> E[{pool_label}]
    E --> F[{latent_label}]
    F --> G[7个物理结构参数]
```

![训练和验证曲线](training_curves.png)

## 最终测试

| 方案，均为三种子单模型均值 | 原始标签周期 MSE | 镜像等价 MSE | 五尺寸 MSE |
| --- | ---: | ---: | ---: |
{test_rows}

| 参数 | 昨日基线 MAE | 本轮选中方案 MAE | 单位 |
| --- | ---: | ---: | --- |
{parameter_rows}

角度MAE按同一联合镜像匹配后统计；尺寸标签不变。完整逐种子记录在 `summary.json`，不通过测试结果改选种子，也不把参数误差直接解释成光谱谷位或Q值误差。角度存在一谱多解，几何误差改善仍需后续回仿真检验。

![参数误差](parameter_errors.png)

## 保存、使用和核对

部署权重 `models/metaformerpro.pt`，版本 `{checkpoint['version']}`，采用验证成绩居中的 seed `{seed}`。本轮100个测试结构中，该权重有 **{deployed['valid']}/{deployed['total']}** 个预测通过几何检查；所有种子的计数和失败原因在 `verification.json`。推理对每条新输入返回 `geometry_valid`，不会对失败结构静默修补。

CPU四线程、batch1暖机前向中位耗时：昨日基线 {times['pro_v01']:.3f} ms，本轮 {times['selected']:.3f} ms；包含预处理，不含文件/权重读取，不承诺其他机器时延。CPU重载相对导出GPU预测最大差 `{reload_difference:.3g}`，指标复算通过，训练集标准化及样本ID核对通过。

```powershell
python train_metaformerpro.py all
python summarize_metaformerpro.py
python predict_metaformerpro.py --sample-id 207
```

解释器沿用 `D:/Anaconda3/envs/transformer_env/python.exe`。PyCharm使用 `13 MetaformerPro train v02` 和 `14 MetaformerPro predict`；项目目录 `D:/projects/MetaformerPro/Metaformer_As2Se3`。三个 `models/metaformerpro_v01_seed*.pt` 为复现实验所需基线快照。修改训练配置或代码时使用新的output_dir，避免覆盖冻结记录。

源码、权重、配置及记录位于Optics_design的 `codex/metaformerpro` 分支。新版生成器数据尚未加入本轮训练；这批数值数据上的成绩不认证真实金器件性能。
'''
    (directory / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'selected': selected, 'means': means, 'verification': verification}, indent=2))


if __name__ == '__main__':
    main()
