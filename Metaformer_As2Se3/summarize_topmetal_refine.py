"""Render the frozen top-metal refinement results without loading neural models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/topmetal_refine'
LABELS = {'baseline': '原圆周头', 'standardized': '完整标准化',
          'position_readout': '保留分块位置', 'encoder192': '192维编码器'}
COLORS = {'baseline': '#6a7c8b', 'standardized': '#4c95b8',
          'position_readout': '#b47d4a', 'encoder192': '#4f917d'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relative_change(before, after):
    percent = (after / before - 1) * 100
    return f'{"降低" if percent < 0 else "增加"} {abs(percent):.1f}%'


def mean_train(records, variant):
    return float(np.mean([r['train']['mirror_equivalent']['mse']
                          for r in records if r['variant'] == variant]))


def main():
    summary = read_json(OUT/'summary.json')
    plan = read_json(OUT/'plan.json')
    records = read_json(OUT/'validation_results.json')
    test = read_json(OUT/'test_metrics.json')
    verification = read_json(OUT/'verification.json')
    diagnosis = read_json(OUT/'signal_diagnosis.json')
    coverage = read_json(ROOT/'results/topmetal1000/spectral_coverage.json')['all']
    selection = summary['selection']
    selected = selection['selected']
    seed = selection['representative_seed']
    ranking = selection['ranking']
    names = ['baseline'] + list(plan['variants'])
    indexed = {r['variant']: r for r in ranking}
    baseline = summary['test_mean']['baseline']
    chosen = summary['test_mean'][selected]
    base_mirror, chosen_mirror = baseline['mirror_equivalent'], chosen['mirror_equivalent']

    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)
    positions = np.arange(len(names))
    means = [indexed[n]['mean_validation_mse'] for n in names]
    spreads = [indexed[n]['std_validation_mse'] for n in names]
    axes[0].bar(positions, means, yerr=spreads, capsize=4,
                color=[COLORS[n] for n in names], alpha=.8)
    for i, name in enumerate(names):
        values = [r['validation']['mirror_equivalent']['mse']
                  for r in records if r['variant'] == name]
        axes[0].scatter(i+np.linspace(-.12, .12, len(values)), values,
                        s=24, color='#273d50', zorder=4)
        axes[0].text(i, means[i]+spreads[i]+.00035, f'{means[i]:.5f}', ha='center', fontsize=9)
    axes[0].set_xticks(positions, [LABELS[n] for n in names], rotation=12)
    axes[0].set(ylabel='联合镜像验证 MSE', title='三种子均值、标准差及各次结果')
    axes[0].set_ylim(0, max(m+s for m,s in zip(means,spreads))*1.18)
    row = next(r for r in records if r['variant'] == selected and r['seed'] == seed)
    history = read_json(OUT/'validation_runs'/f'{selected}_seed{seed}'/'history.json')
    epochs = [h['epoch'] for h in history]
    axes[1].plot(epochs, [h['train_mse'] for h in history], color='#4386ac', label='训练批损失（含 dropout）')
    axes[1].plot(epochs, [h['validation']['mirror_equivalent']['mse'] for h in history],
                 color='#b57b43', label='验证 MSE（eval）')
    axes[1].axvline(row['best_epoch'], color='#5c6d7d', linestyle='--', linewidth=1,
                    label=f'保存轮次 {row["best_epoch"]}')
    axes[1].set(xlabel='轮次', ylabel='联合镜像 MSE', title=f'冻结部署运行：{LABELS[selected]} / seed{seed}')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(OUT/'validation_comparison.png', dpi=170)
    plt.close(fig)

    compared = ['baseline'] if selected == 'baseline' else ['baseline', selected]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.5), constrained_layout=True)
    for ax, labels, key, unit in ((axes[0], ['P','L','H','dX','dY'], 'mae_nm', 'nm'),
                                 (axes[1], ['奇数角','偶数角'], 'mae_deg', '°')):
        width = .65 / len(compared)
        for i, name in enumerate(compared):
            values = np.asarray([r['test']['mirror_equivalent'][key]
                                 for r in test['runs'] if r['variant'] == name])
            ax.bar(np.arange(len(labels))+(i-(len(compared)-1)/2)*width,
                   values.mean(0), width, yerr=values.std(0), capsize=3,
                   color=COLORS[name], label=LABELS[name])
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylabel(f'MAE / {unit}')
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    axes[0].set_title('五尺寸：同一100条测试的三种子均值 ± 标准差')
    axes[1].set_title('两角：联合镜像等价评价')
    fig.savefig(OUT/'parameter_errors.png', dpi=170)
    plt.close(fig)

    if selected == 'baseline':
        decision = ('本轮三个候选均未满足预设替换门槛，**继续采用原圆周角度头**。'
                    '`models/metaformerpro_refined.pt` 保存被保留的基线权重，部署版本仍为0.2.0；原文件也继续保留。')
    else:
        decision = (f'验证集按预设规则选中 **{selected}（{LABELS[selected]}），seed {seed}**。'
                    f'同一既有测试集上的三种子平均镜像MSE{relative_change(base_mirror["mse"], chosen_mirror["mse"])}，'
                    f'五尺寸MSE{relative_change(base_mirror["length_mse"], chosen_mirror["length_mse"])}。')
        if chosen_mirror['mse'] >= base_mirror['mse']:
            decision += ' 测试未复现验证改善，不能据此宣称综合性能提升；本轮没有再用测试结果改选其它候选。'
    vrows = '\n'.join(
        f'| {r["variant"]} | {r["parameters"]:,} | {mean_train(records,r["variant"]):.6f} | '
        f'{r["mean_validation_mse"]:.6f} ± {r["std_validation_mse"]:.6f} | '
        f'{r["length_mse"]:.6f} | {r["mae_nm"][3]:.3f} | {r["mae_nm"][4]:.3f} | '
        f'{"是" if r["eligible"] else "否"} / {"是" if r["replacement_threshold_passed"] else "否"} |'
        for r in ranking)
    mrows = '\n'.join(
        f'| {name} | {values["periodic"]["mse"]:.6f} | {values["mirror_equivalent"]["mse"]:.6f} | '
        f'{values["mirror_equivalent"]["length_mse"]:.6f} |'
        for name,values in summary['test_mean'].items())
    base_errors = base_mirror['mae_nm']+base_mirror['mae_deg']
    chosen_errors = chosen_mirror['mae_nm']+chosen_mirror['mae_deg']
    prows = '\n'.join(f'| {label} | {before:.3f} | {after:.3f} | {after-before:+.3f} | {unit} |'
        for label,before,after,unit in zip(['P','L','H','dX','dY','奇数角','偶数角'],
                                         base_errors,chosen_errors,['nm']*5+['°']*2))
    parameter_table = ('| 参数 | 基线MAE | 冻结选择MAE | 变化（负数为降低） | 单位 |\n'
                       '|---|---:|---:|---:|---|\n'+prows)
    if selected == 'baseline':
        parameter_table = ('| 参数 | 保留模型的三种子平均MAE | 单位 |\n|---|---:|---|\n' +
            '\n'.join(f'| {label} | {value:.3f} | {unit} |' for label,value,unit in
            zip(['P','L','H','dX','dY','奇数角','偶数角'], chosen_errors, ['nm']*5+['°']*2)))
    trows = '\n'.join(
        f'| {r["variant"]} | {r["seed"]} | {r["best_epoch"]} | {r["epochs"]} | '
        f'{r["training_seconds"]:.1f} | {"沿用已有运行" if r.get("reused_existing_run") else "本轮训练"} |'
        for r in records)
    gap_rows = '\n'.join(
        f'| {name} | {mean_train(records,name):.6f} | {indexed[name]["mean_validation_mse"]:.6f} | '
        f'{indexed[name]["mean_validation_mse"]-mean_train(records,name):+.6f} |'
        for name in names)
    chosen_geometry = verification['geometry'][f'{selected}_seed{seed}']
    seconds = sum(r['training_seconds'] for r in records if not r.get('reused_existing_run'))
    floor_band = diagnosis['normalization']['1300_1700_nm']
    dy = diagnosis['training_length_correlations']['alpha_y']
    long_coverage = coverage['samples_with_ordinary_resolved_dip_above_1000_nm']
    supplement = diagnosis['angle_baseline_supplement']
    constant_angles = supplement['fitted_constant_validation']['mirror_equivalent']['mae_deg']
    canonical = supplement['ridge_canonical_validation']['mirror_equivalent']
    uncanonical = supplement['ridge_uncanonical_circular_validation']['mirror_equivalent']
    report = f'''# MetaformerPro：新层序数据上的进一步优化

2026-10-27。分支 `codex/metaformerpro-topmetal-refine`，独立工作树 `D:/projects/MetaformerPro_refine/Metaformer_As2Se3`。

{decision}

这轮只优化“反射谱→七个结构参数”的逆向模型。冻结选择后只测试基线和被选中的方案，不把没有胜出的候选逐一送到测试集继续挑选。原有100条测试样本已在前几轮查看过，**这是既有测试集上的重复比较，不是新增盲测**。三种子波动表示训练随机性，不是物理精度的置信区间。

## 数据和诊断

沿用同一1000条生成器2.2.0数值数据，固定800训练/100验证/100测试。数据SHA256：`{summary['dataset_sha256']}`。层序自下而上为厚SiO₂、100 nm被动JC Au、As₂Se₃四聚体，基板延伸至下PML，采用金属局部8 nm网格。材料、层序和逐样本完整性核验沿用[上一轮报告](https://github.com/CypherZzzyer05/Report/tree/main/reports/2026-10-24_metaformerpro_topmetal1000)。本批仍为 `pipeline_validation`，没有新增FDTD或网格收敛论证。

仅用训练/验证的[轻量诊断](signal_diagnosis.json)发现：1300–1700 nm有{floor_band['points_clamped_at_0_02']}/400个点受到原0.02标准差下限影响，但最小归一化标准差仍为{floor_band['minimum_normalized_std_at_0_02']:.3f}；降低下限的岭回归没有改善验证结果。dY与单个波长的训练集最大线性关联仅 |r|={dy['max_abs_pearson']:.3f}。这些结果提示dY的可提取信号较弱，不能据此证明物理上无法辨识，也不能把全部误差归因于归一化。

补充比较只在800条训练标签上拟合一个常数角对，验证集镜像等价角MAE已可达到 **{constant_angles[0]:.2f}° / {constant_angles[1]:.2f}°**，这部分改善并没有使用光谱。因而，相对原始角均值常数的全部提升不能都解释成网络学到了角度信息。同一α的圆周岭回归中，固定联合镜像代表使镜像验证MSE从{uncanonical['mse']:.6f}降至{canonical['mse']:.6f}；同时原始标签周期MSE变差，两种评价均保留在诊断JSON中。这是标签表示对照，不是本轮神经模型的性能提升。

原先严格双主谷标准在这批数据中仅覆盖{coverage['samples_with_strict_pair']}/1000条；两个严格主谷均>1000 nm的样本为{coverage['samples_with_strict_pair_both_dips_above_1000_nm']}条。{long_coverage}条在1000 nm以上存在普通明显谷，两种计数不能混用。本轮优化不能补出训练数据未覆盖的长波双谷目标。

## 三项有限改动

| 方案 | 改动 |
|---|---|
| baseline | 上一轮96维、4层、圆周头，原始0.02标准差下限；沿用三个已有运行 |
| standardized | 标准差下限改为0.005，其他结构相同；检验弱长波缩放是否有帮助 |
| position_readout | 同样0.005下限，将100个波长分块位置保留到输出层，避免先池化为一个向量 |
| encoder192 | 同样0.005下限，编码维度96→192，层数不变；检验扩大骨干容量 |

三个候选各用42/43/44种子从随机初始化训练；AdamW、batch8、学习率3e-4、weight decay 0.001，最多100轮，20轮无足够改善早停。预处理统计只由800条训练谱拟合并写入权重。三个已有基线重新载入后的验证预测已与归档对齐。

替换基线要求：三种子平均联合镜像验证MSE至少降低1%；五尺寸验证MSE最多增加2%、dX MAE最多增加5%、dY MAE最多增加2%。在满足全部条件者中选平均验证MSE最低者，部署其验证表现居中的种子；没有合格者则保留基线。选择冻结后再评估既有测试集。

| 方案 | 参数量 | 保存权重训练MSE均值 | 镜像验证MSE均值 ± 标准差 | 五尺寸验证MSE | dX MAE/nm | dY MAE/nm | 尺寸门槛 / 1%改善 |
|---|---:|---:|---:|---:|---:|---:|---|
{vrows}

![验证方案对照和部署运行曲线](validation_comparison.png)

## 收敛与训练验证差距

本轮新增九次训练合计{seconds/60:.1f}分钟；该数值不包含沿用的三次基线运行，也不包含脚本启动和图表整理。

| 方案 | 种子 | 保存轮次 | 总轮数 | 训练秒数 | 来源 |
|---|---:|---:|---:|---:|---|
{trows}

| 方案 | 保存权重训练eval MSE均值 | 验证MSE均值 | 验证−训练 |
|---|---:|---:|---:|
{gap_rows}

上表在保存的同一权重上以eval模式比较训练与验证。正差距说明训练集拟合优于验证集，不能仅凭训练损失继续下降认定泛化继续改善。图中逐轮训练曲线是含dropout的批损失，与表中的保存权重eval误差不是同一口径；虚线标出最终保存轮次。

## 冻结后的测试

| 方案 | 原始标签周期MSE | 镜像等价MSE | 五尺寸MSE |
|---|---:|---:|---:|
{mrows}

{parameter_table}

长度按真实nm还原；两个角采用**同一联合镜像代表**后计算周期误差，没有分别为每个角挑选有利标签。表中为三个种子的平均，实际部署为单个seed {seed}；原始标签周期MSE同时保留，避免将对称等价评价误读为精确标签恢复。

![逐参数测试误差](parameter_errors.png)

部署种子的100条测试结构中，{chosen_geometry['valid']}/100通过几何检查。CPU推理与保存的测试预测最大归一化差为{verification['cpu_prediction_max_difference']:.2e}。训练集标准化检查通过。几何合法与参数MAE都不能证明输出候选已经复现目标光谱；本轮没有候选FDTD回验。

## 使用本轮权重

原有 `models/metaformerpro_topmetal.pt` 继续保留，本轮冻结输出单独保存为 `models/metaformerpro_refined.pt`，SHA256：`{summary['model_sha256']}`。PyCharm在本项目中使用解释器 `D:/Anaconda3/envs/transformer_env/python.exe`，运行项19读取固定输入并展示结构；输入CSV仍要求700–1700 nm等间距1000点、反射率0–1。

```powershell
python train_metaformerpro_refine.py all
python summarize_topmetal_refine.py
python predict_metaformerpro.py --checkpoint models/metaformerpro_refined.pt --spectrum inputs/target_topmetal.csv --output results/topmetal_refine/prediction_207.json --show
```

207号继续作为预先固定的界面示例，不按本轮测试误差挑选“最佳”结构。目标谱是已有数据的输入曲线，模型输出仍为待回验候选。

![冻结选择的结构候选与输入目标谱](prediction_207.png)

逐运行指标、冻结选择及核验分别保存在 [validation_results.json](validation_results.json)、[selection.json](selection.json)、[test_metrics.json](test_metrics.json)、[verification.json](verification.json)。
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report': str(OUT/'report.md'), 'selected': selected, 'seed': seed,
                      'test_mse_change_percent': (chosen_mirror['mse']/base_mirror['mse']-1)*100},
                      ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
