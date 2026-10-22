"""Plot the frozen ablation results and check exported model predictions."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from fdtd_pipeline.config import Geometry
from optimize_spectral_inverse import load_model, metrics, sha, write_json

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    directory = ROOT / 'results/inverse_features1000'
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    selection = report['selection']
    trials = json.loads((directory / 'validation_results.json').read_text(encoding='utf-8'))
    assert sha(plan['dataset']) == report['dataset_sha256']
    selected = selection['selected']
    chosen = next(r for r in trials if r['variant'] == selected and r['seed'] == selection['representative_seed'])
    baseline = [r for r in trials if r['variant'] == 'baseline']
    chosen_baseline = sorted(baseline, key=lambda r: r['validation']['mirror_equivalent']['mse'])[1]
    model, checkpoint = load_model(ROOT / 'models/inverse_features1000.pt', torch.device('cpu'))
    with np.load(plan['dataset'], allow_pickle=False) as data:
        indices = data['test_indices']
        x = torch.from_numpy(data['spectra'][indices].astype(np.float32))
        target = data['geometry_01'][indices]
        lower, upper = data['lower_values'], data['upper_values']
        widths = upper - lower
        wavelength = data['wavelength_nm'].copy()
        example = data['spectra'][data['train_indices'][0]].copy()
        train_x = torch.from_numpy(data['spectra'][data['train_indices']].astype(np.float32))
        train_y = data['geometry_01'][data['train_indices']].copy()
    # A training-only constant tests whether angle improvements exceed merely
    # choosing a convenient representative in the mirror quotient space.
    from scipy.optimize import minimize
    angles = train_y[:, 5:].astype(np.float64)
    alternate = 1 - angles[:, ::-1]
    def constant_objective(candidate):
        direct = (candidate - angles + .5) % 1 - .5
        other = (candidate - alternate + .5) % 1 - .5
        return np.minimum((direct ** 2).sum(-1), (other ** 2).sum(-1)).mean()
    starts = [(a, b) for a in (1/6, .5, 5/6) for b in (1/6, .5, 5/6)]
    fits = [minimize(constant_objective, start, method='Nelder-Mead',
                     options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12}) for start in starts]
    constant = train_y.mean(0).astype(np.float64)
    constant[5:] = min(fits, key=lambda fit: fit.fun).x % 1
    constant_metrics = metrics(np.broadcast_to(constant, target.shape).copy(), target, widths)
    write_json(directory / 'constant_baseline.json', {'normalized_parameters': constant.tolist(),
                'fit_split': 'train only', 'angle_fit': '9-start joint periodic-mirror Nelder-Mead',
                'test': constant_metrics})
    with torch.inference_mode():
        cpu = model(x).numpy()
        for _ in range(10):
            model(x[:1])
        timings = []
        for _ in range(50):
            start = time.perf_counter()
            model(x[:1])
            timings.append((time.perf_counter() - start) * 1000)
    archive = np.load(directory / f'{selected}_seed{selection["representative_seed"]}_test.npz')
    reload_difference = float(np.abs(cpu - archive['prediction']).max())
    assert reload_difference < 1e-5
    baseline_model, _ = load_model(chosen_baseline['checkpoint'], torch.device('cpu'))
    with torch.inference_mode():
        for _ in range(10):
            baseline_model(x[:1])
        baseline_timings = []
        for _ in range(50):
            start = time.perf_counter()
            baseline_model(x[:1])
            baseline_timings.append((time.perf_counter() - start) * 1000)
    del baseline_model
    if hasattr(model, 'views'):
        with torch.no_grad():
            expected = model.views.raw_views(train_x).mean(0, keepdim=True)
        torch.testing.assert_close(model.views.mean, expected, atol=1e-6, rtol=1e-5)
    maximum_metric_difference = 0.
    legal = {}
    for item in report['evaluated']:
        name, seed = item['variant'], item['seed']
        path = directory / f'{name}_{"ensemble" if seed == "ensemble" else "seed" + str(seed)}_test.npz'
        with np.load(path, allow_pickle=False) as data:
            p, t = data['prediction'], data['target']
            recomputed = metrics(p, t, widths)
            for mode in ('periodic', 'mirror_equivalent'):
                maximum_metric_difference = max(maximum_metric_difference,
                      abs(recomputed[mode]['mse'] - item['test'][mode]['mse']))
            count = 0
            for values in lower + p * widths:
                try:
                    Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
                    count += 1
                except ValueError:
                    pass
            legal[f'{name}_seed{seed}'] = count
    assert maximum_metric_difference < 1e-12
    verification = {'checkpoint_reload_max_difference': reload_difference,
                    'metric_recalculation_max_difference': maximum_metric_difference,
                    'cpu_batch1_median_ms': float(np.median(timings)), 'cpu_threads': 4,
                    'baseline_cpu_batch1_median_ms': float(np.median(baseline_timings)),
                    'timing_scope': 'Warm FP32 forward; includes preprocessing, excludes checkpoint/CSV I/O',
                    'valid_geometry_counts_out_of_100': legal,
                    'selected_checkpoint_sha256': sha(ROOT / 'models/inverse_features1000.pt')}
    write_json(directory / 'verification.json', verification)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    colors = ['#61758b', '#237cb5', '#188875', '#d58a29', '#ac4778', '#7665ac', '#bf5b44']
    names = list(plan['variants'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for j, (mode, title) in enumerate([('periodic', '原始标签周期误差'), ('mirror_equivalent', '镜像等价结构误差')]):
        means = [np.mean([r['validation'][mode]['mse'] for r in trials if r['variant'] == n]) for n in names]
        deviations = [np.std([r['validation'][mode]['mse'] for r in trials if r['variant'] == n]) for n in names]
        axes[j].bar(np.arange(len(names)), means, yerr=deviations, color=colors, capsize=4)
        axes[j].set_xticks(range(len(names)), names, rotation=22, ha='right')
        axes[j].set(ylabel='验证集归一化 MSE，均值 ± 种子标准差', title=title)
    fig.suptitle('固定 800/100/100 划分；三个随机种子；用验证集选择方案')
    fig.savefig(directory / 'validation_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for record, color, label in [(chosen_baseline, colors[0], '原模型'), (chosen, colors[-1], selected)]:
        history = json.loads((Path(record['checkpoint']).parent / 'history.json').read_text(encoding='utf-8'))
        epochs = [h['epoch'] for h in history]
        train = [h['train_mse'] for h in history]
        if 'validation' in history[0]:
            val = [h['validation']['mirror_equivalent']['mse'] for h in history]
            axes[1].plot(epochs, val, color=color, label=f'{label} 验证（镜像等价）')
        axes[0].plot(epochs, train, color=color, label=f'{label} 训练（各自损失）')
    for ax in axes:
        ax.set(xlabel='轮次', ylabel='归一化 MSE')
        ax.legend()
    axes[0].set_title('训练损失；不同对称目标的数值不直接比较')
    axes[1].set_title('各方案验证成绩见对照柱状图')
    fig.savefig(directory / 'training_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(wavelength, example)
    axes[0, 0].set(title='保留原始反射率 R', ylabel='R', xlabel='波长 / nm')
    from spectral_inverse import SpectralConfig, SpectralViews
    views = SpectralViews(SpectralConfig())
    raw = views.raw_views(torch.tensor(example[None], dtype=torch.float32)).numpy()[0]
    axes[0, 1].plot(wavelength, raw[1], color=colors[2])
    axes[0, 1].set(title='11 点固定局部多项式一阶导数', ylabel=r'dR/dλ / nm$^{-1}$', xlabel='波长 / nm')
    axes[1, 0].plot(wavelength, raw[2], color=colors[3])
    axes[1, 0].set(title='二阶导数突出谷的曲率', ylabel=r'd²R/dλ² / nm$^{-2}$', xlabel='波长 / nm')
    fft = np.fft.rfft(example, norm='ortho')[:64]
    axes[1, 1].plot(np.arange(1, 64), fft.real[1:], label='实部')
    axes[1, 1].plot(np.arange(1, 64), fft.imag[1:], label='虚部')
    axes[1, 1].set(title='DFT 非直流系数 1–63，保留相位', xlabel='系数序号；沿波长采样轴变换', ylabel='正交归一化系数')
    axes[1, 1].text(.02, .95, f'DC={fft[0].real:.2f}；模型输入同时保留 DC',
                    transform=axes[1, 1].transAxes, va='top')
    axes[1, 1].legend()
    fig.savefig(directory / 'preprocessing.png', dpi=150)
    plt.close(fig)
    rows = []
    for ranking in selection['ranking']:
        n = ranking['variant']
        rows.append(f"| {n} | {ranking['parameters']:,} | {ranking['mean_periodic_validation_mse']:.6f} | "
                    f"{ranking['mean_equivalent_validation_mse']:.6f} ± {ranking['std']:.6f} |")
    test_rows = []
    comparisons = {}
    for n in dict.fromkeys(('baseline', selected)):
        items = [r['test'] for r in report['evaluated'] if r['variant'] == n and r['seed'] != 'ensemble']
        comparisons[n] = {m: float(np.mean([i[m]['mse'] for i in items])) for m in ('periodic', 'mirror_equivalent')}
        test_rows.append(f"| {n}，3 个种子均值 | {comparisons[n]['periodic']:.6f} | {comparisons[n]['mirror_equivalent']:.6f} |")
    reduction = 1 - comparisons[selected]['mirror_equivalent'] / comparisons['baseline']['mirror_equivalent']
    raw_reduction = 1 - comparisons[selected]['periodic'] / comparisons['baseline']['periodic']
    def change_label(value):
        return f'{"降低" if value >= 0 else "升高"} {100*abs(value):.1f}%'
    parameter_rows = []
    for index, label in enumerate(('P', 'L', 'H', 'dX', 'dY', '奇数角', '偶数角')):
        values = []
        for n in ('baseline', selected):
            entries = [r['test']['mirror_equivalent'] for r in report['evaluated'] if r['variant'] == n and r['seed'] != 'ensemble']
            values.append(np.mean([r['mae_nm'][index] if index < 5 else r['mae_deg'][index-5] for r in entries]))
        parameter_rows.append(f'| {label} | {values[0]:.3f} | {values[1]:.3f} | {"nm" if index < 5 else "°"} |')
    text = f'''# 1000 样本逆向模型：光谱视图与镜像等价优化

日期：2026-10-22。使用现有生成器 1.0 数据，Au / 有限厚 SiO2 / As2Se3，700–1700 nm 反射率 1000 点。数据 SHA256：`{report['dataset_sha256']}`。不重算 FDTD，不改变原始数据、原模型或原权重。

**验证集选中 `{selected}`。三个种子平均测试镜像等价 MSE 相对原模型{change_label(reduction)}；原始标签周期 MSE {change_label(raw_reduction)}。** 两种误差均报告，不能把只改变评价口径造成的下降称为网络改善。

## 实验与选择

固定训练 800、验证 100、测试 100；seed 42/43/44；每轮 batch=8、AdamW lr=3e-4、weight_decay=1e-3；最多 100 轮，20 轮停滞早停。所有标准化仅拟合训练集。基线 seed44 复用同一数据的历史冻结权重，其余从头训练。

每个训练运行按自身损失对应的验证误差选择 epoch；方案之间统一按三个种子的**平均镜像等价七参数 MSE**排名。`selection.json` 在任何本轮测试预测前落盘，只测试原模型与胜出方案。测试集此前已被检查，因此这是固定测试集上的后续实验，不是新的盲测。

先比较五个方案，随后在尚未打开测试预测时增加 compact_raw_mirror 与 shape_mirror，检查镜像损失的收益是否依赖傅里叶或导数分支。第一轮已有运行原样复用；最终在共七个方案之间按同一验证规则选择。完整配置为 `configs/inverse_feature_refinement.json`，这种验证集驱动的补充也会增加对验证集的适配，需以后用独立数据再确认。

| 方案 | 参数量 | 原始标签验证 MSE 均值 | 镜像等价验证 MSE 均值 ± 种子标准差 |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

![验证对照](validation_comparison.png)

## 预处理和架构

- compact_raw：96 维嵌入，4 层编码器，4 头、每头 24 维；100 个 10 点 patch；学习式加权池化。原模型为 128 维、10 层、10×64 注意力，约 400 万参数。
- shape：保留原始 R，同时计算 11 点、3 阶固定 Savitzky–Golay 一阶/二阶导数；10 点细分块与 50 点粗分块形成 100+20 token。这个对照同时增加形态通道与粗尺度，不能分拆归因。
- shape_fft：在 shape 上加入前 64 个复数 DFT 系数的实部和虚部，通过 128→96 投影形成 1 个全局 token。沿均匀波长轴计算，不解释为时间频谱或直接的光程；完整 R 始终保留，不是只用低频重建。
- shape_fft_mirror：相同网络，改为镜像等价联合周期损失。归一化角度变换为 `(u奇,u偶)→(1−u偶,1−u奇)`。两角**共同选择同一个等价分支**，不能各自挑最小误差。
- compact_raw_mirror / shape_mirror：分别在 compact_raw / shape 上仅更改镜像损失，形成必要的去分支对照。

![预处理示例，取训练集第一条，不用于挑选模型](preprocessing.png)

法向线偏振、平面各向同性层和四聚体镜像对称性给出该等价关系。它减少一种逆映射冲突，不保证结构唯一，也不意味着角度没有光谱作用。该表示与损失组合属于本项目探索，未据此宣称论文层面的原创性。

## 最终测试

| 模型 | 原始标签周期 MSE | 镜像等价 MSE |
| --- | ---: | ---: |
{chr(10).join(test_rows)}

| 参数 | 原模型 MAE，三种子均值 | 选中方案 MAE，三种子均值 | 单位 |
| --- | ---: | ---: | --- |
{chr(10).join(parameter_rows)}

上述 MAE 的角度按联合镜像匹配后统计，尺寸始终按原标签比较。另用训练标签拟合一个**不读取光谱**的镜像等价常数基线：测试等价 MSE 为 {constant_metrics['mirror_equivalent']['mse']:.6f}，两个角 MAE 分别为 {constant_metrics['mirror_equivalent']['mae_deg'][0]:.2f}° / {constant_metrics['mirror_equivalent']['mae_deg'][1]:.2f}°（`constant_baseline.json`）。角度是否从光谱中学到信息还应与此基线比较，不能只与旧模型的原始标签误差比较。

本轮选中方案的奇数角尚未超过该常数基线，偶数角有所改善；角度辨识仍是短板。dX 的平均误差也略有上升，没有把总指标改善表述为所有参数同时改善。

每个种子及预先定义的三模型集成结果均保存在 `summary.json`；主结论使用三种子单模型均值，不把集成与单模型混比。镜像对齐后对角度作圆均值的集成尚需实际回仿真验证，不能凭结构参数平均假定其光谱也平均。

![训练曲线](training_curves.png)

## 使用和核对

选中单模型使用验证表现居中的 seed `{selection['representative_seed']}`，权重 `models/inverse_features1000.pt`。本机 CPU 四线程、batch1 的暖机前向中位耗时 **{verification['cpu_batch1_median_ms']:.3f} ms**，原模型相同测法为 **{verification['baseline_cpu_batch1_median_ms']:.3f} ms**；包含预处理，不含文件/权重读取，不据此承诺其他机器的时延。保存重载与导出 GPU 预测最大归一化差 `{reload_difference:.3g}`；导出指标复算通过。测试预测的几何合法性计数见 `verification.json`。

部署用 seed44 在 100 个测试结构上全部通过几何检查；另一个种子 seed43 有 1 个预测不合法。未根据测试合法性改选种子。推理入口返回 `geometry_valid` 和具体错误，后续新目标仍须逐条检查。

```powershell
python optimize_spectral_inverse.py all --config configs/inverse_feature_refinement.json
python summarize_feature_study.py
python predict_spectral_inverse.py --sample-id 207
```

解释器沿用 `D:/Anaconda3/envs/transformer_env/python.exe`，无需新增依赖。PyCharm 配置 `11 Optimize spectrum features 1000`、`12 Predict optimized inverse`。原有配置 09/10 保留。全部新实验、缓存、权重写入 D 盘。

配置中的 `historical_checkpoint` 是本机复用历史 seed44 的可选路径；另一台机器不存在此文件时，训练器会从头训练该种子，不要求复制整个旧工作树。完成的实验计划会检查代码和数据 SHA；修改配置/训练代码后应使用新的 `output_dir`。

## 结论范围

这批数据已知 Au 损耗拟合严重缺失；本次成绩只证明同一历史数值数据集上的学习效果。新版厚基板/中间 Au 器件和修正材料不在此训练数据内。未运行新 FDTD，因此不把参数 MSE 换算成谷位/线宽误差，也未认证 Q 或 BIC。新材料与网格问题不妨碍这里的算法对照，但真实器件设计仍需可靠数据和回仿真。
'''
    (directory / 'report.md').write_text(text, encoding='utf-8')
    print(json.dumps({'selected': selected, 'test_mean': comparisons, 'verification': verification}, indent=2))


if __name__ == '__main__':
    main()
