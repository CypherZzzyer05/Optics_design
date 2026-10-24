"""Summarize the separately trained top-metal batch after validation selection."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from fdtd_pipeline.config import Geometry
from optimize_spectral_inverse import evaluate, metrics, sha, write_json
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformerpro import load_pro

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/topmetal1000'


def main():
    torch.set_num_threads(4)
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    trials = json.loads((OUT/'validation_results.json').read_text(encoding='utf-8'))
    coverage = json.loads((OUT/'spectral_coverage.json').read_text(encoding='utf-8'))
    assert sha(plan['dataset']) == summary['dataset_sha256']
    assert all(not trial['reference_reused'] for trial in trials)
    datasets = {s: SpectrumGeometryDataset(plan['dataset'], s, allow_validation_data=True)
                for s in ('train', 'test')}
    train, test = datasets['train'], datasets['test']
    widths = test.upper_values - test.lower_values
    constant = np.broadcast_to(train.labels.numpy().mean(0), test.labels.shape).copy()
    constant_score = metrics(constant, test.labels.numpy(), widths)
    selected, selected_seed = summary['selection']['selected'], summary['selection']['representative_seed']
    evaluated, legality = [], {}
    reload_difference = 0.
    for trial in trials:
        name = f'{trial["variant"]}_seed{trial["seed"]}'
        path = OUT/f'{name}_test.npz'
        model, saved = load_pro(trial['checkpoint'])
        assert saved['dataset_sha256'] == summary['dataset_sha256']
        assert saved['normalization_fit_sample_ids'] == train.sample_ids.tolist()
        simulation = saved['simulation_settings']
        assert simulation['metal_position'] == 'above_substrate'
        assert simulation['substrate_termination'] == 'pml'
        assert simulation['metal_thickness_nm'] == 100
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                prediction = archive['prediction'].copy()
                np.testing.assert_array_equal(archive['sample_ids'], test.sample_ids)
                np.testing.assert_allclose(archive['target'], test.labels.numpy(), atol=0, rtol=0)
            score = metrics(prediction, test.labels.numpy(), widths)
        else:
            # Both heads were declared for test evaluation before any training.
            prediction, score = evaluate(model, test, torch.device('cpu'))
            np.savez_compressed(path, prediction=prediction, target=test.labels.numpy(), sample_ids=test.sample_ids)
        if trial['variant'] == selected and trial['seed'] == selected_seed:
            cpu, _ = evaluate(model, test, torch.device('cpu'))
            reload_difference = float(np.max(np.abs(cpu-prediction)))
            assert reload_difference < 1e-5
            torch.testing.assert_close(model.normalizer.mean, train.spectra[:, None].mean(0, keepdim=True),
                                       rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(model.normalizer.scale,
                        train.spectra[:, None].std(0, unbiased=False, keepdim=True).clamp_min(.02),
                        rtol=1e-6, atol=1e-6)
        failures = []
        for index, values in enumerate(test.lower_values + prediction*widths):
            try:
                Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
            except ValueError as error:
                failures.append({'sample_id': int(test.sample_ids[index]), 'reason': str(error)})
        legality[name] = {'valid': len(test)-len(failures), 'total': len(test), 'failures': failures}
        evaluated.append({'variant': trial['variant'], 'seed': trial['seed'], 'test': score})
        destination = OUT/'validation_runs'/name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ('history.json', 'validation_predictions.npz'):
            shutil.copyfile(Path(trial['checkpoint']).parent/filename, destination/filename)
        del model
    means = {}
    for name in plan['variants']:
        scores = [r['test'] for r in evaluated if r['variant'] == name]
        means[name] = {mode: {key: np.mean([s[mode][key] for s in scores], axis=0).tolist()
                                   for key in ('mse', 'length_mse', 'angle_mse', 'mae_nm', 'mae_deg')}
                       for mode in ('periodic', 'mirror_equivalent')}
    write_json(OUT/'test_mean.json', means)
    write_json(OUT/'test_metrics.json', {'runs': evaluated, 'train_mean_constant': constant_score})
    write_json(OUT/'verification.json', {'reload_max_difference': reload_difference,
        'normalization_training_only': True, 'all_checkpoints_use_new_dataset': True,
        'all_checkpoints_use_topmetal_pml_stack': True, 'geometry': legality,
        'model_sha256': sha(plan['model_output'])})
    shutil.copyfile(Path(plan['output_dir'])/'study.log', OUT/'training.log')

    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    names, colors = ['pro_v01', 'circular'], ['#617487', '#287eb1']
    for name, color in zip(names, colors):
        rows = sorted([r for r in trials if r['variant']==name],
                      key=lambda r:r['validation']['mirror_equivalent']['mse'])
        row = rows[1]
        history = json.loads((Path(row['checkpoint']).parent/'history.json').read_text(encoding='utf-8'))
        epochs = [h['epoch'] for h in history]
        label = f'{name} / seed{row["seed"]}'
        axes[0].plot(epochs, [h['train_mse'] for h in history], color=color, label=label)
        axes[1].plot(epochs, [h['validation']['mirror_equivalent']['mse'] for h in history], color=color, label=label)
    for ax in axes:
        ax.set(xlabel='轮次', ylabel='联合镜像周期 MSE')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    axes[0].set_title('新批次训练损失（各方案验证居中种子）')
    axes[1].set_title('新批次验证损失')
    fig.savefig(OUT/'training_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, labels, key in [(axes[0], ['P','L','H','dX','dY'], 'mae_nm'),
                             (axes[1], ['奇数角','偶数角'], 'mae_deg')]:
        for i, (name, color) in enumerate(zip(names, colors)):
            ax.bar(np.arange(len(labels))+(i-.5)*.34, means[name]['mirror_equivalent'][key], .34,
                   label=name, color=color)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(means[n]['mirror_equivalent'][key]) for n in names)*1.3)
    axes[0].set(ylabel='MAE / nm', title='五个尺寸：三种子测试均值')
    axes[1].set(ylabel='MAE / °', title='两角：联合镜像匹配后的误差')
    fig.savefig(OUT/'parameter_errors.png', dpi=160)
    plt.close(fig)
    rows = '\n'.join(f'| {name} | {m["periodic"]["mse"]:.6f} | {m["mirror_equivalent"]["mse"]:.6f} | '
                     f'{m["mirror_equivalent"]["length_mse"]:.6f} |' for name,m in means.items())
    vrows = '\n'.join(f'| {r["variant"]} | {r["mean_validation_mse"]:.6f} | {r["length_mse"]:.6f} | '
                      f'{r["dx_mae_nm"]:.3f} | {r["eligible"]} |' for r in summary['selection']['ranking'])
    old, new = means['pro_v01']['mirror_equivalent'], means['circular']['mirror_equivalent']
    prows = '\n'.join(f'| {label} | {a:.3f} | {b:.3f} | {unit} |' for label,a,b,unit in zip(
        ['P','L','H','dX','dY','奇数角','偶数角'], old['mae_nm']+old['mae_deg'], new['mae_nm']+new['mae_deg'], ['nm']*5+['°']*2))
    trows = '\n'.join(f'| {r["variant"]} | {r["seed"]} | {r["best_epoch"]} | {r["epochs"]} | {r["training_seconds"]:.1f} |' for r in trials)
    chosen = means[selected]['mirror_equivalent']
    improvement = (1-chosen['mse']/constant_score['mirror_equivalent']['mse'])*100
    cov = coverage['all']
    coverage_rows = '\n'.join(
        f"| {split} | {values['samples']} | {values['samples_with_strict_pair']} | "
        f"{values['samples_with_strict_pair_both_dips_above_1000_nm']} | "
        f"{values['samples_with_ordinary_resolved_dip_above_1000_nm']} |"
        for split, values in coverage['by_split'].items())
    report = f'''# MetaformerPro：新层序1000样本训练与验证

2026-10-24汇总；训练于2026-10-23完成。独立分支 `codex/metaformerpro-topmetal-1000`。这次只使用新数据，两个输出头均从随机初始化训练，未复用旧层序权重。

验证规则选中 **{selected}，seed {selected_seed}**。所选方案三种子平均镜像等价测试MSE为 **{chosen['mse']:.6f}**，比训练集标签均值常数基线低 **{improvement:.1f}%**。这是本批数值数据内的比较，不与旧物理体系测试误差直接比较。

## 数据查收

1000/1000个样本，固定800训练/100验证/100测试；样本ID、原始NPZ、汇总数组、配置及材料身份核验见 [data_audit.json](data_audit.json)。数据SHA256：`{summary['dataset_sha256']}`。

层序自下而上：延伸至下PML的厚SiO₂、100 nm JC Au、As₂Se₃四聚体。金属区域z=-100..0 nm，柱体z=0..H，T监视器在SiO₂内。Au使用2.2.0的修正被动模型；金层z网格约7.733 nm，其他区域自动网格。全部达到自动关断，质量仍为pipeline_validation，未据此认定网格精度收敛。

## 光谱覆盖

有 **{cov['samples_with_ordinary_resolved_dip_above_1000_nm']}/1000** 条光谱在1000 nm以上存在普通明显谷，但按原先固定的严格双主谷标准，只有 **{cov['samples_with_strict_pair']}/1000** 条合格，且 **没有一条两个主谷都在1000 nm以上**。普通谷与严格双主谷是不同统计，不能混用。当前几何范围并未充分覆盖长波双主谷目标，仅增加同范围的随机样本不能保证解决这一问题。

普通谷：突出度≥0.1、半突出度宽度≥5个采样间隔。严格配对还要求各谷R≤0.2、突出度≥0.4、距波段边界>30 nm，谷间距≥30 nm、谷间隆起与较浅谷的反射率差≥0.2。枚举所有合格配对后统计，未只检查每条谱最强的一对。

| 分割 | 样本数 | 严格双主谷 | 两主谷均>1000 nm | 至少一个普通谷>1000 nm |
|---|---:|---:|---:|---:|
{coverage_rows}

逐谱结果见 [spectral_coverage.json](spectral_coverage.json)。该统计描述输入数据，不使用模型测试误差筛选展示样本。样本207预先固定作界面演示，只有一个满足普通阈值的谷（约770.07 nm），未作双谷、最佳结构或最佳预测的宣称。半突出度宽度不等同于模态线宽，未由此计算Q。

![新数据光谱分布](dataset_overview.png)

## 训练和选择

现有两个头：Sigmoid七输出、圆周角度输出；骨干均为4层96维，约32万参数。每个方案3种子，batch8，AdamW学习率3e-4，最多100轮、20轮无足够改善早停。标准化只使用800条训练谱。未在本轮增加新网络模块或容量扫描。

先要求五尺寸验证MSE不超过Sigmoid基线的1.02倍、dX MAE不超过其1.05倍，再选三种子平均镜像验证误差最低者，部署种子取该方案验证表现居中者。选择在本轮测试评估前冻结；两种头均按预先计划报告测试结果，不据测试结果更换选择。

| 方案 | 镜像验证MSE | 五尺寸验证MSE | dX MAE/nm | 通过尺寸门槛 |
|---|---:|---:|---:|---|
{vrows}

![训练与验证](training_curves.png)

| 方案 | 种子 | 最佳轮 | 总轮数 | 总耗时/秒 |
|---|---:|---:|---:|---:|
{trows}

## 测试结果

| 方案 | 原始标签周期MSE | 镜像等价MSE | 五尺寸MSE |
|---|---:|---:|---:|
{rows}

圆周头相对Sigmoid头，三种子平均镜像等价测试MSE降低 **{(1-new['mse']/old['mse'])*100:.1f}%**，五尺寸MSE降低 **{(1-new['length_mse']/old['length_mse'])*100:.1f}%**。奇数角MAE降低约{old['mae_deg'][0]-new['mae_deg'][0]:.2f}°，偶数角MAE增加约{new['mae_deg'][1]-old['mae_deg'][1]:.2f}°，并非所有指标都改善。dY误差仍接近训练均值常数预测，说明该参数尚未被可靠辨识。

训练均值常数基线的镜像等价MSE：{constant_score['mirror_equivalent']['mse']:.6f}。三种子均值用于方案比较，部署仍为单个验证选中的种子。

| 参数 | Sigmoid头 MAE | 圆周头 MAE | 单位 |
|---|---:|---:|---|
{prows}

![各参数误差](parameter_errors.png)

部署权重在100条测试结构中的合法数为 {legality[f'{selected}_seed{selected_seed}']['valid']}/100；CPU重载与保存预测最大归一化差为 {reload_difference:.2e}。训练标准化、样本ID、层序和数据身份检查通过。几何合法不代表已经复现目标光谱；本轮未运行新FDTD。

## 固定光谱到结构

PyCharm打开 `D:/projects/MetaformerPro_topmetal/Metaformer_As2Se3`，解释器 `D:/Anaconda3/envs/transformer_env/python.exe`。运行项16训练新批次；运行项17使用 `models/metaformerpro_topmetal.pt` 和 `inputs/target_topmetal.csv`，保存参数和正确层序的示意图。

```powershell
python train_metaformerpro.py all --config configs/metaformerpro_topmetal1000.json
python analyze_topmetal_spectra.py
python summarize_topmetal1000.py
python predict_metaformerpro.py --checkpoint models/metaformerpro_topmetal.pt --spectrum inputs/target_topmetal.csv --output results/topmetal1000/prediction_207.json --show
```

![新层序结构候选](prediction_207.png)

原始新数据包保持在用户提供的D盘目录；项目仅复制约10 MB汇总及任务配置。旧数据与旧权重保留。模型输出为候选结构，目标谱能否由其实现仍需后续电磁回验。
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'selected':selected, 'seed':selected_seed, 'test_mean':means,
                      'constant_baseline':constant_score, 'valid':legality[f'{selected}_seed{selected_seed}'],
                      'reload_difference':reload_difference}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
