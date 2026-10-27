"""Plot the frozen resonance-tokenization study; never trains or runs FDTD."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from resonance_sampling import build_extra_sampling_grid


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/topmetal_resonance'
LABELS = {'baseline': '原均匀分块', 'uniform_extra': '均匀加密对照', 'resonance': '共振局部细分'}
COLORS = {'baseline': '#687f91', 'uniform_extra': '#3b91ad', 'resonance': '#bb8252'}
SEED_COLORS = ['#2d7cae', '#ce7f3e', '#4a9567']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def change(before, after):
    percent = (after / before - 1) * 100
    return f'{"降低" if percent < 0 else "增加"}{abs(percent):.1f}%'


def plot_sampling(dataset):
    """Display the predeclared example without choosing it from model errors."""
    with np.load(dataset, allow_pickle=False) as data:
        index = np.flatnonzero(data['sample_ids'] == 207)
        if len(index) != 1:
            raise ValueError('The predeclared sample 207 must occur exactly once')
        spectrum = data['spectra'][index[0]].copy()
        wavelength = data['wavelength_nm'].copy()
    raw = torch.from_numpy(spectrum.astype(np.float32))[None]
    grids = {mode: build_extra_sampling_grid(raw, mode)[0].cpu().numpy()
             for mode in ('uniform_extra', 'resonance')}
    sample = {'sample_id': 207, 'purpose': 'fixed illustration, not selected from prediction errors',
              'smoothing_role': 'window location only; tokens sample the original reflectance',
              'interpolation_role': 'linear resampling does not create spectral resolution', 'modes': {}}
    token_wavelengths = {}
    for mode, grid in grids.items():
        positions = np.interp(grid, np.arange(len(wavelength)), wavelength)
        token_wavelengths[mode] = positions
        sample['modes'][mode] = {
            'extra_token_count': int(len(grid)),
            'centers_nm': positions.mean(axis=1).tolist(),
            'spans_nm': (positions[:, -1] - positions[:, 0]).tolist(),
            'sample_wavelengths_nm': positions.tolist(),
            'groups': [{'first_token': start, 'last_token': start + 15,
                        'minimum_wavelength_nm': float(positions[start:start + 16].min()),
                        'maximum_wavelength_nm': float(positions[start:start + 16].max())}
                       for start in (0, 16)],
        }
    (OUT / 'sampling_207.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.5), sharex=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.115, top=.91, hspace=.34)
    fig.suptitle('固定样本207：保留完整光谱，在输入相关的位置增加细分', fontsize=16, y=.97)
    low, high = float(spectrum.min()), float(spectrum.max())
    span = max(high - low, .1)
    ylim = (low - span * .23, high + span * .08)
    native = wavelength.reshape(100, 10)
    axes[0].plot(wavelength, spectrum, color='#283e51', lw=1.5)
    axes[0].scatter(native.mean(axis=1), np.full(100, low - span * .12),
                    marker='|', s=80, color=COLORS['baseline'])
    axes[0].set_title('原始输入：100个等长token，每个包含10个原网格点', loc='left', fontsize=11)
    for ax, mode in zip(axes[1:], ('uniform_extra', 'resonance')):
        positions = token_wavelengths[mode]
        ax.plot(wavelength, spectrum, color='#687b8c', lw=1.4)
        for group, color in enumerate(('#3b91ad', '#bb8252')):
            values = positions[group * 16:(group + 1) * 16]
            y = low - span * (.09 + group * .07)
            ax.hlines(np.full(16, y), values[:, 0], values[:, -1], color=color, lw=2)
            ax.scatter(values.mean(axis=1), np.full(16, y), marker='|', s=100, color=color)
            ax.vlines(values.mean(axis=1), y, low, color=color, alpha=.15, lw=.65)
        ax.set_title(f'{LABELS[mode]}：保留上面的100个全谱token，另加32个token',
                     loc='left', fontsize=11)
    for ax in axes:
        ax.set(ylim=ylim, ylabel='反射率 R', xlim=(wavelength[0], wavelength[-1]))
        ax.grid(alpha=.15)
    axes[-1].set_xlabel('波长 / nm')
    fig.text(.5, .04, '下方横线表示附加token的覆盖范围，短竖线表示中心；两种颜色仅区分各16个token。',
             ha='center', color='#526371', fontsize=10)
    fig.text(.5, .015, '平滑副本仅定位；原始反射率与绝对波长保留。连续插值不会增加原始光谱分辨率。',
             ha='center', color='#526371', fontsize=9)
    fig.savefig(OUT / 'sampling_207.png', dpi=175)
    plt.close(fig)
    return sample


def plot_comparison(plan, records, selection, histories):
    names = ['baseline'] + list(plan['variants'])
    ranked = {r['variant']: r for r in selection['ranking']}
    groups = {name: [r for r in records if r['variant'] == name] for name in names}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.12, top=.90, hspace=.36, wspace=.23)
    fig.suptitle('MetaformerPro：均匀分块与共振局部细分\n相同4层骨干与训练设置；每个方案3个种子', fontsize=17, y=.975)
    ax = axes.flat[0]
    means = [ranked[name]['mean_validation_mse'] for name in names]
    spreads = [ranked[name]['std_validation_mse'] for name in names]
    ax.bar(np.arange(len(names)), means, yerr=spreads, capsize=4,
           color=[COLORS[name] for name in names], alpha=.8)
    for i, name in enumerate(names):
        values = [r['validation']['mirror_equivalent']['mse'] for r in groups[name]]
        ax.scatter(i + np.linspace(-.12, .12, len(values)), values, color='#243c4d', s=25, zorder=4)
        ax.text(i, means[i] + spreads[i] + .0004, f'{means[i]:.5f}', ha='center', fontsize=10)
    ax.set_xticks(np.arange(len(names)), [LABELS[name] for name in names])
    ax.set(ylabel='联合镜像验证 MSE', title='三种子均值、标准差和每次结果',
           ylim=(0, max(m + s for m, s in zip(means, spreads)) * 1.20))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    maximum = max(max(max(h['train_mse'], h['validation']['mirror_equivalent']['mse'])
                      for h in history) for history in histories.values())
    for ax, name in zip(list(axes.flat)[1:], names):
        for row, color in zip(sorted(groups[name], key=lambda r: r['seed']), SEED_COLORS):
            history = histories[(name, row['seed'])]
            epochs = [h['epoch'] for h in history]
            ax.plot(epochs, [h['train_mse'] for h in history], '--', color=color, alpha=.55, lw=1.1)
            ax.plot(epochs, [h['validation']['mirror_equivalent']['mse'] for h in history],
                    color=color, alpha=.9, lw=1.3)
            best = next(h for h in history if h['epoch'] == row['best_epoch'])
            ax.scatter(row['best_epoch'], best['validation']['mirror_equivalent']['mse'],
                       s=25, color=color, edgecolor='white', linewidth=.4, zorder=5)
        ax.set(title=f'{LABELS[name]} · {groups[name][0]["parameters"]:,} 参数',
               xlabel='训练轮次', ylabel='联合镜像归一化 MSE',
               xlim=(0, plan['epochs']), ylim=(0, maximum * 1.05))
        ax.grid(alpha=.17)
    handles = [Line2D([0], [0], color=color, label=f'种子{seed}')
               for color, seed in zip(SEED_COLORS, plan['seeds'])]
    handles += [Line2D([0], [0], color='#333333', label='实线：验证'),
                Line2D([0], [0], color='#333333', linestyle='--', label='虚线：训练批损失（含dropout）'),
                Line2D([0], [0], color='#333333', linestyle='none', marker='o', label='保存轮次')]
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(.5, .04), fontsize=9)
    fig.text(.5, .018, '原基线曲线复用已有运行；两种附加采样各三次新训练。token数量影响用均匀加密组控制。',
             ha='center', fontsize=9, color='#526371')
    fig.savefig(OUT / 'resonance_comparison.png', dpi=170)
    plt.close(fig)


def plot_parameter_errors(test, selected):
    compared = ['baseline'] if selected == 'baseline' else ['baseline', selected]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.5), constrained_layout=True)
    for ax, labels, key, unit in ((axes[0], ['P', 'L', 'H', 'dX', 'dY'], 'mae_nm', 'nm'),
                                  (axes[1], ['奇数角', '偶数角'], 'mae_deg', '°')):
        width = .65 / len(compared)
        for i, name in enumerate(compared):
            values = np.asarray([r['test']['mirror_equivalent'][key] for r in test['runs']
                                 if r['variant'] == name])
            ax.bar(np.arange(len(labels)) + (i - (len(compared) - 1) / 2) * width,
                   values.mean(0), width, yerr=values.std(0), capsize=3,
                   color=COLORS[name], label=LABELS[name])
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylabel(f'MAE / {unit}')
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    axes[0].set_title('既有100条测试：三种子均值 ± 标准差')
    axes[1].set_title('两角：联合镜像等价评价')
    fig.savefig(OUT / 'parameter_errors.png', dpi=170)
    plt.close(fig)


def summarize():
    summary = read(OUT / 'summary.json')
    plan = read(OUT / 'plan.json')
    records = read(OUT / 'validation_results.json')
    test = read(OUT / 'test_metrics.json')
    verification = read(OUT / 'verification.json')
    selection = read(OUT / 'selection.json')
    selected, seed = selection['selected'], selection['representative_seed']
    candidates = sorted((r for r in records if r['variant'] == 'resonance'),
                        key=lambda r: r['validation']['mirror_equivalent']['mse'])
    candidate = candidates[len(candidates) // 2]
    candidate_path = ROOT / 'models/metaformerpro_resonance_candidate.pt'
    shutil.copyfile(candidate['checkpoint'], candidate_path)
    candidate_info = {'variant': 'resonance', 'seed': candidate['seed'],
                      'selection': 'median validation seed; experimental, not a deployment override',
                      'checkpoint': str(candidate_path),
                      'sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                      'full_test_evaluated': selected == 'resonance',
                      'fdtd_run_performed': False}
    (OUT / 'experimental_candidate.json').write_text(
        json.dumps(candidate_info, ensure_ascii=False, indent=2), encoding='utf-8')
    histories = {(r['variant'], r['seed']): read(OUT / 'validation_runs' /
                    f'{r["variant"]}_seed{r["seed"]}' / 'history.json') for r in records}
    plot_comparison(plan, records, selection, histories)
    plot_parameter_errors(test, selected)
    plot_sampling(Path(plan['dataset']))

    means = test['mean']
    base = means['baseline']['mirror_equivalent']
    chosen = means[selected]['mirror_equivalent']
    baseline_validation = next(row for row in selection['ranking'] if row['variant'] == 'baseline')
    uniform_validation = next(row for row in selection['ranking'] if row['variant'] == 'uniform_extra')
    local_validation = next(row for row in selection['ranking'] if row['variant'] == 'resonance')
    if selected == 'baseline':
        decision = ('两种新输入编码均未满足预设替换条件，**继续保留现用原均匀分块模型**。'
                    '本轮保存的冻结输出不是已证明有效的共振编码升级。')
    else:
        decision = (f'验证集按预设规则选中 **{LABELS[selected]}，seed {seed}**。'
                    f'相对原均匀分块，三种子平均镜像测试MSE{change(base["mse"], chosen["mse"])}，'
                    f'五尺寸测试MSE{change(base["length_mse"], chosen["length_mse"])}。')
        if chosen['mse'] >= base['mse']:
            decision += ' 测试未复现验证改善；冻结选择后没有依据测试结果换选其它方案。'
    direct_comparison = (
        f'共振局部细分相对相同132个token的均匀加密组，三种子平均验证MSE'
        f'{change(uniform_validation["mean_validation_mse"], local_validation["mean_validation_mse"])}；'
        f'相对原100个token基线{change(baseline_validation["mean_validation_mse"], local_validation["mean_validation_mse"])}。'
        '这是当前数据和固定配方上的观察，不能仅凭单个种子或总loss判断其稳定优越性。')
    if not selection['resonance_vs_uniform']['mechanism_threshold_passed']:
        direct_comparison += ' 本轮未通过预设的共振取样机制门槛，尚未证实按谷分配优于均匀附加取样。'
    train_means = {name: float(np.mean([r['train']['mirror_equivalent']['mse']
                                      for r in records if r['variant'] == name]))
                   for name in LABELS}
    validation_rows = '\n'.join(
        f'| {LABELS[r["variant"]]} | {100 if r["variant"] == "baseline" else 132} | '
        f'{r["parameters"]:,} | {train_means[r["variant"]]:.6f} | '
        f'{r["mean_validation_mse"]:.6f} ± {r["std_validation_mse"]:.6f} | '
        f'{r["length_mse"]:.6f} | {r["mae_nm"][3]:.3f} | {r["mae_nm"][4]:.3f} | '
        f'{"是" if r["eligible"] else "否"} / {"是" if r["replacement_threshold_passed"] else "否"} |'
        for r in selection['ranking'])
    epoch_rows = '\n'.join(
        f'| {LABELS[r["variant"]]} | {r["seed"]} | {r["best_epoch"]} | {r["epochs"]} | '
        f'{r["training_seconds"]:.1f} | {"复用已有运行" if r.get("reused_existing_run") else "本轮新训练"} |'
        for r in records)
    test_rows = '\n'.join(
        f'| {LABELS[name]} | {values["periodic"]["mse"]:.6f} | '
        f'{values["mirror_equivalent"]["mse"]:.6f} | {values["mirror_equivalent"]["length_mse"]:.6f} |'
        for name, values in means.items())
    parameter_rows = '\n'.join(
        f'| {label} | {before:.3f} | {after:.3f} | {unit} |'
        for label, before, after, unit in zip(['P', 'L', 'H', 'dX', 'dY', '奇数角', '偶数角'],
            base['mae_nm'] + base['mae_deg'], chosen['mae_nm'] + chosen['mae_deg'], ['nm'] * 5 + ['°'] * 2))
    geometry = verification['geometry'][f'{selected}_seed{seed}']
    new_records = [r for r in records if not r.get('reused_existing_run')]
    seconds = sum(r['training_seconds'] for r in new_records)
    report = f'''# MetaformerPro：共振局部细分输入编码试验

2026-10-27。独立分支 `codex/metaformerpro-resonance-tokenization`，工作树 `D:/projects/MetaformerPro_resonance/Metaformer_As2Se3`。原模型和生成器保留。

{decision}

## 这次具体改变了什么

现用100个全谱token保留；新增两种132-token输入进行对照：均匀加密组在全波段附加32个token，共振组在输入谱的可信谷附近附加32个token。二者共用原Conv10投影和同一个4层编码器，均无新增可训练参数。每个附加token保留实际波长中心、覆盖宽度的固定位置编码。没有外接ResNet、LSTM或正向神经网络。

仅在平滑副本上定位谷和窗口；窗口内取原反射率。窗口最多围绕两个可信谷，不强制所有谱都有两个谷；缺谷时按预定均匀规则补足token。连续插值用于固定输入形状，不会增加原始约1 nm采样的物理分辨率。局部细分是假设能帮助有限网络提取已有形状差异，不能创造缺失信息或打破镜像同谱。

固定定位规则为11点三阶SG、谷显著度≥0.01、间隔≥20个原网格点；按显著度选最多两谷后按波长排序。窗口总跨度为半突出度宽度的4倍，截在16–240个原网格间隔内，边界仅裁剪。单谷的另一块为全域16token，无谷为全域32token。原始反射率不做局部深度归一化；中心、跨度及附加类型使用固定编码。算法选出的谷不等于已确认的物理模式，宽度也不作为Q值。

![固定207号输入的全谱与附加token位置](sampling_207.png)

取样位置和跨度保存在 [sampling_207.json](sampling_207.json)。207号为此前固定的界面示例，不按本轮误差挑选。

## 数据与比较规则

沿用修正材料及层序的新1000条数据，固定800训练/100验证/100测试。数据SHA256：`{summary['dataset_sha256']}`。厚SiO₂ / 被动JC Au100 / As₂Se₃四聚体，基板延伸至下PML；金属局部8 nm网格，仍为 `pipeline_validation`，没有新增FDTD或网格收敛证明。

所有训练输入归一化统计只由800条训练谱拟合；谷窗口完全由当前输入谱确定，不使用真实几何或测试标签。3个种子均为42/43/44，维度96、编码器4层、注意力4头、前馈192、dropout0.1、圆周角度头；AdamW、batch8、学习率3e-4、weight decay0.001，最多100轮，20轮无足够改善早停。现用基线的三次已有运行复用，本轮新增{len(new_records)}次训练。

验证门槛沿用：三种子平均镜像MSE至少降低1%，五尺寸MSE最多增加2%，dX MAE最多增加5%，dY MAE最多增加2%；合格者中选最低验证误差，以该方案验证表现居中的种子冻结候选。**只在选择冻结后评估原基线及被选中的方案。** 既有100条测试此前已经查看过，属于重复比较，不是新增盲测。没有根据本轮测试结果再选方案或种子。

## 验证结果

{direct_comparison}

| 方案 | token数 | 参数量 | 保存权重训练eval MSE | 镜像验证MSE均值 ± 标准差 | 五尺寸验证MSE | dX MAE/nm | dY MAE/nm | 尺寸门槛 / 1%改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{validation_rows}

![验证比较与全部运行曲线](resonance_comparison.png)

新训练合计{seconds / 60:.1f}分钟，不含复用基线和整理脚本。种子标准差描述训练随机性，不代表物理准确度置信区间。训练eval表格来自保存权重；曲线虚线是包含dropout的逐轮训练批损失，二者口径不同。

| 方案 | 种子 | 保存轮次 | 总轮数 | 训练秒数 | 来源 |
|---|---:|---:|---:|---:|---|
{epoch_rows}

## 冻结后的既有测试

| 方案 | 原始标签周期MSE | 镜像等价MSE | 五尺寸MSE |
|---|---:|---:|---:|
{test_rows}

| 参数 | 原均匀分块MAE | 冻结候选MAE | 单位 |
|---|---:|---:|---|
{parameter_rows}

![逐参数测试误差](parameter_errors.png)

表格为三种子平均，单个推理候选使用seed {seed}。两个角共同选择同一个镜像代表，原始标签周期误差也保留。候选种子的100条测试结构中{geometry['valid']}/100通过几何合法性检查；该检查不等于满足最小加工间隙。CPU重载预测与归档结果最大归一化差为{verification['cpu_prediction_max_difference']:.2e}。几何合法和标签MAE不能证明候选已复现目标谱，本轮没有候选FDTD回验。

## 复现与使用

冻结输出：`{summary['model_path']}`，SHA256：`{summary['model_sha256']}`。现用原权重不覆盖。输入仍为700–1700 nm、1000个等间距反射率点；同一输入定位规则用于训练与推理。

共振方案另保留验证表现居中的seed {candidate['seed']}，文件为 `models/metaformerpro_resonance_candidate.pt`，仅供实验查看，不自动替换现用权重。该文件的身份及是否进入完整测试见 [experimental_candidate.json](experimental_candidate.json)。PyCharm运行项23加载此实验候选；运行项17继续加载现用模型。

本轮18项定向程序测试通过，覆盖采样/缺谷回退、绝对位置与跨度、梯度、132-token集成、旧模型兼容和权重重载。实验候选训练归一化与固定207号CPU/CUDA推理检查见 [candidate_verification.json](candidate_verification.json)。

```powershell
python train_metaformerpro_refine.py all --config configs/metaformerpro_topmetal_resonance.json
python summarize_resonance_study.py
python predict_metaformerpro.py --checkpoint models/metaformerpro_resonance_candidate.pt --spectrum inputs/target_topmetal.csv --output results/topmetal_resonance/prediction_207.json --show
```

![共振实验候选的固定207号推理示意，图中光谱为输入目标](prediction_207.png)

此图读取另行保留的共振实验权重，不代表验证冻结方案已切换。输入谱来自已有数值数据，图中没有候选的FDTD回验曲线。

逐运行记录见 [validation_results.json](validation_results.json)、[selection.json](selection.json)、[test_metrics.json](test_metrics.json)、[verification.json](verification.json)。
'''
    (OUT / 'README.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'README.md'), 'selected': selected, 'seed': seed,
                      'resonance_vs_uniform_validation_percent':
                          (local_validation['mean_validation_mse'] / uniform_validation['mean_validation_mse'] - 1) * 100,
                      'selected_vs_baseline_test_percent': (chosen['mse'] / base['mse'] - 1) * 100},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampling-only', action='store_true', help='Plot fixed207 sampling before training finishes')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    if args.sampling_only:
        plot_sampling(ROOT / 'examples/jc220_topmetal_1000/dataset.npz')
        print(OUT / 'sampling_207.png')
    else:
        summarize()


if __name__ == '__main__':
    main()
