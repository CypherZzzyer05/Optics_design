"""Plot the frozen encoder-depth study from saved JSON/NPZ results only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/topmetal_depth'
LABELS = {'baseline': '4层基线', 'depth8': '8层编码器', 'depth12': '12层编码器'}
COLORS = {'baseline': '#687f91', 'depth8': '#3b91ad', 'depth12': '#bb8252'}
SEED_COLORS = ['#2d7cae', '#ce7f3e', '#4a9567']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def change(before, after):
    value = (after/before-1)*100
    return f'{"降低" if value < 0 else "增加"}{abs(value):.1f}%'


def main():
    summary = read(OUT/'summary.json')
    plan = read(OUT/'plan.json')
    records = read(OUT/'validation_results.json')
    test = read(OUT/'test_metrics.json')
    verification = read(OUT/'verification.json')
    selection = summary['selection']
    selected, seed = selection['selected'], selection['representative_seed']
    names = ['baseline']+list(plan['variants'])
    ranked = {r['variant']:r for r in selection['ranking']}
    groups = {name:[r for r in records if r['variant'] == name] for name in names}
    train_means = {name:float(np.mean([r['train']['mirror_equivalent']['mse'] for r in rows]))
                   for name,rows in groups.items()}
    histories = {(r['variant'],r['seed']):read(OUT/'validation_runs'/
                    f'{r["variant"]}_seed{r["seed"]}'/'history.json') for r in records}

    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False, 'font.size':10})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.12, top=.90, hspace=.36, wspace=.23)
    fig.suptitle('MetaformerPro：只改变编码器层数\n同一800/100/100划分；每个深度3个种子', fontsize=17, y=.975)
    ax = axes.flat[0]
    means = [ranked[n]['mean_validation_mse'] for n in names]
    spreads = [ranked[n]['std_validation_mse'] for n in names]
    ax.bar(np.arange(len(names)), means, yerr=spreads, capsize=4,
           color=[COLORS[n] for n in names], alpha=.8)
    for i,name in enumerate(names):
        values = [r['validation']['mirror_equivalent']['mse'] for r in groups[name]]
        ax.scatter(i+np.linspace(-.12,.12,len(values)), values, color='#243c4d', s=25, zorder=4)
        ax.text(i,means[i]+spreads[i]+.0004,f'{means[i]:.5f}',ha='center',fontsize=10)
    ax.set_xticks(np.arange(len(names)),[LABELS[n] for n in names])
    ax.set(ylabel='联合镜像验证 MSE', title='三种子均值、标准差和每次结果',
           ylim=(0,max(m+s for m,s in zip(means,spreads))*1.20))
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    maximum = max(max(max(h['train_mse'],h['validation']['mirror_equivalent']['mse'])
                      for h in history) for history in histories.values())
    for ax,name in zip(list(axes.flat)[1:],names):
        for row,color in zip(sorted(groups[name],key=lambda r:r['seed']),SEED_COLORS):
            history = histories[(name,row['seed'])]
            epochs = [h['epoch'] for h in history]
            ax.plot(epochs,[h['train_mse'] for h in history],'--',color=color,alpha=.55,lw=1.1)
            ax.plot(epochs,[h['validation']['mirror_equivalent']['mse'] for h in history],
                    color=color,alpha=.9,lw=1.3)
            best = next(h for h in history if h['epoch'] == row['best_epoch'])
            ax.scatter(row['best_epoch'],best['validation']['mirror_equivalent']['mse'],s=25,
                       color=color,edgecolor='white',linewidth=.4,zorder=5)
        ax.set(title=f'{LABELS[name]} · {groups[name][0]["parameters"]:,} 参数',
               xlabel='训练轮次',ylabel='联合镜像归一化 MSE',xlim=(0,plan['epochs']),ylim=(0,maximum*1.05))
        ax.grid(alpha=.17)
    handles = [Line2D([0],[0],color=c,label=f'种子{s}') for c,s in zip(SEED_COLORS,plan['seeds'])]
    handles += [Line2D([0],[0],color='#333333',label='实线：验证'),
                Line2D([0],[0],color='#333333',linestyle='--',label='虚线：训练批损失（含dropout）'),
                Line2D([0],[0],color='#333333',linestyle='none',marker='o',label='保存轮次')]
    fig.legend(handles=handles,loc='lower center',ncol=6,bbox_to_anchor=(.5,.04),fontsize=9)
    fig.text(.5,.018,'4层曲线为已有运行的复用；8/12层各三次新训练。原早停长度保持，不补齐缺失轮次。',
             ha='center',fontsize=9,color='#526371')
    fig.savefig(OUT/'depth_comparison.png',dpi=170)
    plt.close(fig)

    compared = ['baseline'] if selected == 'baseline' else ['baseline',selected]
    fig, axes = plt.subplots(1,2,figsize=(11.7,4.5),constrained_layout=True)
    for ax,labels,key,unit in ((axes[0],['P','L','H','dX','dY'],'mae_nm','nm'),
                              (axes[1],['奇数角','偶数角'],'mae_deg','°')):
        width = .65/len(compared)
        for i,name in enumerate(compared):
            values = np.asarray([r['test']['mirror_equivalent'][key] for r in test['runs']
                                 if r['variant'] == name])
            ax.bar(np.arange(len(labels))+(i-(len(compared)-1)/2)*width,values.mean(0),width,
                   yerr=values.std(0),capsize=3,color=COLORS[name],label=LABELS[name])
        ax.set_xticks(np.arange(len(labels)),labels)
        ax.set_ylabel(f'MAE / {unit}')
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
    axes[0].set_title('既有100条测试：三种子均值 ± 标准差')
    axes[1].set_title('两角：联合镜像等价评价')
    fig.savefig(OUT/'parameter_errors.png',dpi=170)
    plt.close(fig)

    baseline = summary['test_mean']['baseline']
    chosen = summary['test_mean'][selected]
    base, best = baseline['mirror_equivalent'],chosen['mirror_equivalent']
    if selected == 'baseline':
        decision = ('8层和12层方案均未满足预设替换门槛，**继续使用原4层圆周角度头**。'
                    '`models/metaformerpro_depth.pt` 保存保留的4层权重，本轮未获得可部署的深度升级。')
    else:
        decision = (f'验证集按预设规则选中 **{LABELS[selected]}，seed {seed}**。'
                    f'三种子平均镜像测试MSE相对4层{change(base["mse"],best["mse"])}，'
                    f'五尺寸MSE{change(base["length_mse"],best["length_mse"])}。')
        if best['mse'] >= base['mse']:
            decision += (' 测试未复现验证改善，本轮未据测试结果再改选其它深度。'
                         '12层权重作为实验候选保留，现用4层权重不覆盖。')
    validation_rows = '\n'.join(f'| {LABELS[r["variant"]]} | {r["parameters"]:,} | '
        f'{train_means[r["variant"]]:.6f} | {r["mean_validation_mse"]:.6f} ± {r["std_validation_mse"]:.6f} | '
        f'{r["length_mse"]:.6f} | {r["mae_nm"][3]:.3f} | {r["mae_nm"][4]:.3f} | '
        f'{"是" if r["eligible"] else "否"} / {"是" if r["replacement_threshold_passed"] else "否"} |'
        for r in selection['ranking'])
    epoch_rows = '\n'.join(f'| {LABELS[r["variant"]]} | {r["seed"]} | {r["best_epoch"]} | {r["epochs"]} | '
        f'{r["training_seconds"]:.1f} | {"复用已有运行" if r.get("reused_existing_run") else "本轮新训练"} |'
        for r in records)
    test_rows = '\n'.join(f'| {LABELS[name]} | {values["periodic"]["mse"]:.6f} | '
        f'{values["mirror_equivalent"]["mse"]:.6f} | {values["mirror_equivalent"]["length_mse"]:.6f} |'
        for name,values in summary['test_mean'].items())
    parameter_rows = '\n'.join(f'| {label} | {before:.3f} | {after:.3f} | {unit} |' for label,before,after,unit in
        zip(['P','L','H','dX','dY','奇数角','偶数角'],base['mae_nm']+base['mae_deg'],
            best['mae_nm']+best['mae_deg'],['nm']*5+['°']*2))
    geometry = verification['geometry'][f'{selected}_seed{seed}']
    seconds = sum(r['training_seconds'] for r in records if not r.get('reused_existing_run'))
    report = f'''# MetaformerPro：4 / 8 / 12层编码器对照

2026-10-27。独立分支 `codex/metaformerpro-depth-study`，工作树 `D:/projects/MetaformerPro_depth/Metaformer_As2Se3`。

{decision}

## 对照设置

本轮只改变编码器层数。4层基线复用同一数据上已训练的42/43/44三个种子；8层和12层各从随机初始化训练三个种子，共六次新训练。三组均为96维嵌入、4个注意力头、192维前馈层、圆周角度输出、原始0.02标准差下限、dropout 0.1。训练仍为AdamW、batch8、学习率3e-4、weight decay 0.001，最多100轮，20轮无足够改善早停。

数据仍为生成器2.2.0的1000条修正材料与层序数据，固定800训练/100验证/100测试，SHA256：`{summary['dataset_sha256']}`。层序为厚SiO₂ / 100 nm被动JC Au / As₂Se₃四聚体，SiO₂延伸至下PML。金属局部8 nm网格的数据质量仍为 `pipeline_validation`；本轮没有新增FDTD或正向网络。输入预处理统计仅用800条训练谱拟合。

预设验证选型条件：三种子平均联合镜像验证MSE至少降低1%；五尺寸验证MSE最多增加2%，dX MAE最多增加5%，dY MAE最多增加2%。合格者中选验证误差最低的深度，以该深度验证表现居中的种子保存推理候选；无合格者则保留4层。**验证选择冻结后，仅评估4层基线和被选中的方案。** 这100条测试此前已被查看，本轮属于既有测试集上的重复比较，不是新增盲测。

## 验证与收敛

| 深度 | 参数量 | 保存权重训练eval MSE均值 | 镜像验证MSE均值 ± 种子标准差 | 五尺寸验证MSE | dX MAE/nm | dY MAE/nm | 尺寸门槛 / 1%改善 |
|---|---:|---:|---:|---:|---:|---:|---|
{validation_rows}

![三个深度的验证比较及全部9次运行曲线](depth_comparison.png)

六次新训练合计{seconds/60:.1f}分钟，不包含复用的4层训练与脚本整理。三种子标准差仅描述训练随机性，不是物理精度置信区间。

| 深度 | 种子 | 保存轮次 | 总轮数 | 训练秒数 | 来源 |
|---|---:|---:|---:|---:|---|
{epoch_rows}

表中训练eval和验证误差来自同一个保存权重，口径一致；曲线虚线为含dropout的逐轮训练批损失。训练损失继续降低而验证平台时，应按保存轮次评价，不能把末轮训练损失当作泛化提升。本轮结果只说明这些深度在固定配方和当前数据上的表现，不能单独证明样本量是误差主因，也不能排除其它训练配方的影响。

## 冻结后的测试

| 深度 | 原始标签周期MSE | 镜像等价MSE | 五尺寸MSE |
|---|---:|---:|---:|
{test_rows}

| 参数 | 4层基线MAE | 冻结选择MAE | 单位 |
|---|---:|---:|---|
{parameter_rows}

![逐参数测试误差](parameter_errors.png)

以上为三种子平均；实验候选推理使用单个seed {seed}。两角按同一个联合镜像代表计算周期误差，原始标签周期MSE同时保留。候选种子的100条测试结构中{geometry['valid']}/100通过几何检查，CPU与保存测试预测最大归一化差为{verification['cpu_prediction_max_difference']:.2e}，训练集标准化检查通过。参数误差与几何合法性不代表候选已复现输入光谱，本轮没有候选FDTD回验。

## 使用

原4层权重保留；本轮冻结候选为 `models/metaformerpro_depth.pt`，SHA256：`{summary['model_sha256']}`。PyCharm使用 `D:/Anaconda3/envs/transformer_env/python.exe`；运行项20执行本轮深度对照，运行项21读取 `inputs/target_topmetal.csv` 并展示实验候选结构。运行项17继续使用原4层模型。推理JSON的 `model_config` 明确记录层数和宽度。输入仍要求700–1700 nm等间距1000点，反射率0–1。

```powershell
python train_metaformerpro_refine.py all --config configs/metaformerpro_topmetal_depth.json
python summarize_metaformerpro_depth.py
python predict_metaformerpro.py --checkpoint models/metaformerpro_depth.pt --spectrum inputs/target_topmetal.csv --output results/topmetal_depth/prediction_207.json --show
```

207号仍是固定界面示例，没有按测试误差挑选“最佳”结构。图中的光谱为输入目标曲线。

![冻结选择的候选结构](prediction_207.png)

原始逐运行数据见 [validation_results.json](validation_results.json)、[selection.json](selection.json)、[test_metrics.json](test_metrics.json)、[verification.json](verification.json)。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.md'),'selected':selected,'seed':seed,
                      'test_mse_change_percent':(best['mse']/base['mse']-1)*100},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
