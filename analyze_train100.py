"""Analyze a completed held-out run without fitting or changing its model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.project.resolve()
sys.path.insert(0,str(root))
from fdtd_pipeline.config import Geometry
from train_metaformer import regression_metrics

pointer = json.loads((root/'runs/as2se3_au100_metal8_100/latest_run.json').read_text(encoding='utf-8'))
assert pointer['state']=='completed'
run = Path(pointer['run_dir'])
report = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
audit = json.loads((root/'docs/au100_metal8_100_data_audit.json').read_text(encoding='utf-8'))
with np.load(root/'data/as2se3_au100_metal8_100/dataset.npz',allow_pickle=False) as archive:
    data = {key:archive[key].copy() for key in archive.files}
assert hashlib.sha256((root/'data/as2se3_au100_metal8_100/dataset.npz').read_bytes()).hexdigest()==report['dataset_sha256']==audit['dataset_sha256']
names = list(data['parameter_names'])
train = data['geometry_01'][data['train_indices']].astype(float)
target = data['geometry_01'][data['test_indices']].astype(float)
pred_file = np.atleast_1d(np.genfromtxt(run/'test_predictions.csv',delimiter=',',names=True))
assert pred_file['sample_id'].astype(int).tolist()==data['sample_ids'][data['test_indices']].tolist()
prediction = np.column_stack([pred_file['pred_01_'+name] for name in names])
np.testing.assert_allclose(np.column_stack([pred_file['true_01_'+name] for name in names]),target,atol=6e-10,rtol=0)
recomputed = regression_metrics(prediction,target,data['lower_values'],data['upper_values'],names,'square_periodic')
assert abs(recomputed['normalized_mse']-report['metrics']['test']['normalized_mse'])<1e-9

def periodic_mean(values):
    """Exact squared-geodesic Frechet mean on a unit circle, using only train labels."""
    values = np.asarray(values,dtype=float)%1
    cuts = np.unique(np.r_[0.,(values+.5)%1,1.])
    candidates = list(cuts[:-1])
    for left,right in zip(cuts[:-1],cuts[1:]):
        midpoint = (left+right)/2
        unwrapped = midpoint-((midpoint-values+.5)%1-.5)
        candidates.append(float(np.clip(unwrapped.mean(),left,right))%1)
    loss = lambda c: float(np.mean(((c-values+.5)%1-.5)**2))
    best = min(candidates,key=loss)
    # A dense independent scan must not beat the analytic piecewise minimum.
    grid = np.linspace(0,1,10001,endpoint=False)
    dense_min = np.min(np.mean(((grid[:,None]-values[None,:]+.5)%1-.5)**2,axis=1))
    assert loss(best)<=dense_min+1e-12
    return best

constant = train.mean(axis=0)
constant[5:] = [periodic_mean(train[:,index]) for index in (5,6)]
periodic_baseline = regression_metrics(np.broadcast_to(constant,target.shape),target,
                    data['lower_values'],data['upper_values'],names,'square_periodic')
physical_prediction = data['lower_values']+prediction*(data['upper_values']-data['lower_values'])
geometry_checks = []
for sample_id,row in zip(pred_file['sample_id'].astype(int),physical_prediction):
    try:
        Geometry(*row[:5],theta_odd_deg=row[5],theta_even_deg=row[6])
        geometry_checks.append(dict(sample_id=int(sample_id),valid=True))
    except ValueError as error:
        geometry_checks.append(dict(sample_id=int(sample_id),valid=False,error=str(error)))
model_mse = report['metrics']['test']['normalized_mse']
base_mse = report['train_mean_baseline']['test']['normalized_mse']
results = dict(dataset_sha256=report['dataset_sha256'],run_directory=str(run),
    evaluation_recomputed_from_exported_predictions=True,model_test=recomputed,
    arithmetic_mean_baseline_test=report['train_mean_baseline']['test'],
    periodic_optimal_constant_baseline_test=periodic_baseline,
    periodic_optimal_constant_01=constant.tolist(),
    periodic_optimal_constant_physical=(data['lower_values']+constant*(data['upper_values']-data['lower_values'])).tolist(),
    improvement_vs_arithmetic_mean_percent=100*(1-model_mse/base_mse),
    improvement_vs_periodic_constant_percent=100*(1-model_mse/periodic_baseline['normalized_mse']),
    predicted_geometry_checks=geometry_checks,
    predicted_geometry_valid_count=sum(row['valid'] for row in geometry_checks),
    test_set_used_for_training_or_selection=False,additional_training_performed=False,
    predicted_structure_fdtd_run_performed=False)
errors = lambda values: np.column_stack([values[:,:5]-target[:,:5],(values[:,5:]-target[:,5:]+.5)%1-.5])
model_error = errors(prediction)
arithmetic = train.mean(axis=0)
baseline_error = errors(np.broadcast_to(arithmetic,target.shape))
paired_gain = np.mean(baseline_error**2-model_error**2,axis=1)
bootstrap_seed,bootstrap_samples = 20261016,50000
bootstrap_ids = np.random.default_rng(bootstrap_seed).integers(0,len(target),size=(bootstrap_samples,len(target)))
interval = np.percentile(paired_gain[bootstrap_ids].mean(axis=1),[2.5,97.5])
results['paired_bootstrap'] = dict(seed=bootstrap_seed,resamples=bootstrap_samples,
    unit='held-out structure',difference='arithmetic training-mean baseline minus model normalized MSE',
    mean_difference=float(paired_gain.mean()),percentile_95_interval=interval.tolist(),
    test_structures_improved=int(np.sum(paired_gain>0)),
    interpretation='Descriptive uncertainty over only 10 test structures; not a model selection procedure')
results['length_normalized_mse_improvement_percent'] = float(100*(1-np.mean(model_error[:,:5]**2)/np.mean(baseline_error[:,:5]**2)))
results['angle_normalized_mse_change_percent'] = float(100*(np.mean(model_error[:,5:]**2)/np.mean(baseline_error[:,5:]**2)-1))
out = root/'docs/au100_metal8_100_validation'
out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ('history.json','metrics.json','test_predictions.csv','training.log'):
    shutil.copy2(run/name,out/name)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
history = json.loads((run/'history.json').read_text())
fig,axes = plt.subplots(2,2,figsize=(13.8,9.3),layout='constrained')
fig.suptitle('100 样本首次模型验证：整体略有改善，旋转角仍未学好',fontsize=17,fontweight='bold')
fig.supxlabel('As2Se3 · Au 100 nm · 仅金属层 8 nm 网格 · 80/10/10 · 测试集仅 10 个结构',fontsize=10)
blue,gray,orange = '#2457bd','#9aa6b5','#d38420'
epochs = [row['epoch'] for row in history]
ax = axes[0,0]
ax.plot(epochs,[row['train_mse'] for row in history],color=blue,label='训练（含 dropout）')
ax.plot(epochs,[row['val_mse'] for row in history],color='#008a78',label='验证')
ax.axvline(report['best_epoch'],color=orange,linestyle='--',label=f"最佳第 {report['best_epoch']} 轮")
ax.set(title='训练与验证：后期训练继续下降，验证未改善',xlabel='轮数',ylabel='归一化参数 MSE')
ax.legend(fontsize=9)
ax.grid(alpha=.2)
ax = axes[0,1]
x = np.arange(5)
ax.bar(x-.18,[recomputed['mae_nm'][p] for p in names[:5]],.36,color=blue,label='模型')
ax.bar(x+.18,[report['train_mean_baseline']['test']['mae_nm'][p] for p in names[:5]],.36,color=gray,label='训练集均值基线')
ax.set(title='测试集长度误差：H 改善最明显',ylabel='MAE（nm；越小越好）',xticks=x,xticklabels=['P','L','H','dX','dY'])
ax.grid(axis='y',alpha=.2)
ax.legend(fontsize=9)
ax = axes[1,0]
x = np.arange(2)
ax.bar(x-.25,[recomputed['mae_deg'][p] for p in names[5:]],.25,color=blue,label='模型')
ax.bar(x,[report['train_mean_baseline']['test']['mae_deg'][p] for p in names[5:]],.25,color=gray,label='算术均值基线')
ax.bar(x+.25,[periodic_baseline['mae_deg'][p] for p in names[5:]],.25,color=orange,label='训练集周期最优常数')
ax.set(title='测试集旋转角误差：按 90° 周期计算',ylabel='周期 MAE（°；越小越好）',xticks=x,xticklabels=['奇数象限角','偶数象限角'],ylim=(0,45))
ax.grid(axis='y',alpha=.2)
ax.legend(fontsize=8)
ax = axes[1,1]
ax.axis('off')
text = (f"固定测试集总体 MSE\n\n模型：{model_mse:.6f}\n算术均值基线：{base_mse:.6f}\n"
        f"相对降低：{results['improvement_vs_arithmetic_mean_percent']:.2f}%\n"
        f"周期最优常数基线：{periodic_baseline['normalized_mse']:.6f}\n\n"
        f"完成 {report['epochs_completed']} 轮；最佳第 {report['best_epoch']} 轮；早停。\n"
        f"预测几何合法：{results['predicted_geometry_valid_count']}/10。\n"
        "模型结构未变；未用测试集调参。\n\n"
        "仅 10 个测试结构，改善尚不稳定。\n当前为 pipeline_validation 数据。\n尚未对预测结构回仿真验证光谱。")
ax.text(.03,.97,text,va='top',fontsize=11,linespacing=1.35)
fig.savefig(out/'validation_overview.png',dpi=160)
plt.close(fig)

rows = []
display = ['P','L','H','dX','dY','奇数象限角','偶数象限角']
for index,name in enumerate(names):
    group = 'mae_nm' if index<5 else 'mae_deg'
    m,b = recomputed[group][name],report['train_mean_baseline']['test'][group][name]
    rows.append(f"| {display[index]} | {m:.2f} | {b:.2f} | {100*(1-m/b):+.1f}% |")
note = f'''# 100样本首次 Metaformer 验证

日期：2026-10-16。固定训练/验证/测试为80/10/10，模型和超参数按预先准备配置运行一次，没有按测试结果重新训练或选模型。

本次测试总体归一化参数MSE为 **{model_mse:.6f}**，训练集算术均值常数基线为 **{base_mse:.6f}**，相对降低 **{results['improvement_vs_arithmetic_mean_percent']:.2f}%**。改善主要来自H、P、L；两个旋转角未优于原基线。首次结果表明模型学到部分尺寸关联，整体预测精度仍有限。

## 数据与检查

100条真实Lumerical v251（8.33.3999）反射光谱，每条700–1700 nm共1000点，标签为P/L/H/dX/dY/奇数角/偶数角。Au厚度100 nm，仅金属z网格8 nm，其他区域自动网格。100条唯一几何与固定种子采样计划完全一致，光谱唯一，数据有限；所有状态为2（自动关断完成）。84条使用4 ps上限，14条延长至8 ps，2条至16 ps；固定划分互斥且覆盖全部100条。原上传文件保持不变。

数据SHA256：`{report['dataset_sha256']}`。

收到dataset.npz及job.json，没有收到summary.json；嵌入元数据、配置指纹、全部标签和计划已独立核对。上传文件不包含各样本的原生日志和实际网格数组，本次没有重新证明网格收敛。

透射率范围为−0.15198%至0.79493%，其中10条谱共36点出现微小负值，保留原值并记录；训练仅使用反射谱，未裁剪或平滑。A由1−R−T定义，代数闭合不作为独立精度证明。Au损耗拟合仍未认证（max imaginary permittivity error约14.17），As₂Se₃最后12 nm采用端点延拓，数据质量保持pipeline_validation。

## 训练

现有ResNet1D + 10层Metaformer，7输出，4,081,415个参数；模型和训练实现未修改。固定种子42，batch=8，AdamW初始学习率3e−4、weight_decay=1e−3，验证集调学习率，早停耐心20轮，最多100轮；本机RTX3050 Ti混合精度。

实际完成{report['epochs_completed']}轮、{report['optimizer_steps']}次更新，优化及最终评估约{report['training_seconds']:.1f}秒（不含导入、数据审计及报告制作）。验证最佳第{report['best_epoch']}轮，验证MSE {report['best_val_mse']:.6f}。后期训练损失继续下降而验证未改善，按预定早停结束；最佳模型未出现训练误差远低于验证误差的典型严重过拟合，当前主要表现为小样本下学习收益有限。测试集只在验证选择结束后评估。

## 固定测试集表现

前五项MAE单位nm，角度项MAE单位°且按90°周期取最短误差；降低为正表示优于基线。

| 参数 | 模型MAE | 训练集算术均值基线MAE | MAE相对降低 |
|---|---:|---:|---:|
{chr(10).join(rows)}

训练/验证/测试在最佳权重下MSE分别为{report['metrics']['train']['normalized_mse']:.6f}、{report['metrics']['val']['normalized_mse']:.6f}、{model_mse:.6f}。总体MSE对七个归一化参数等权，不能当作反射率误差或“设计准确率”。

## 补充核验

保留原训练集算术均值基线，同时仅用80条训练标签求解90°周期平方误差的最优常数（前五项为普通均值，后两项为圆周Fréchet均值），再评估同一测试集。该基线MSE为{periodic_baseline['normalized_mse']:.6f}，模型相对降低{results['improvement_vs_periodic_constant_percent']:.2f}%。基线常数没有使用验证或测试标签拟合。

五长度归一化MSE改善{results['length_normalized_mse_improvement_percent']:.2f}%，两角度周期MSE则变差{results['angle_normalized_mse_change_percent']:.2f}%。10个测试结构中{results['paired_bootstrap']['test_structures_improved']}个总体误差优于原均值基线。按测试结构成对bootstrap（固定种子{bootstrap_seed}，{bootstrap_samples}次重采样）得到“基线−模型”平均MSE差的95%百分位区间[{interval[0]:.5f}, {interval[1]:.5f}]，跨过0。因此9.53%的总体改善尚不能视为稳定优势；该分析不改变模型或任何基线。

10个测试预测通过参数上下界之外的四聚体联合几何检查，包括旋转后方柱碰撞和周期边界：**{results['predicted_geometry_valid_count']}/10合法**。这仅证明几何可构造，尚未对预测结构运行FDTD，不能据此认定重构光谱符合输入。

## 结论与下一步

数据采集、合并、七输出训练、90°周期评价和检查点保存链路已跑通，H的学习改善最明显。dX/dY没有明显增益，两角度仍不优于原常数基线；当前不足以支持可靠七参数逆设计。仅10个测试结构、单个训练随机种子，9.53%的总体改善只是这次划分的初步结果，bootstrap区间也显示不确定性较大。

在扩大样本量前，应优先确定下一阶段目标：修正并验证材料拟合、分析旋转角在当前光谱输入中的可辨识性，或为关键参数和目标双谷结构设计更集中的采样。后续对预测结构的光谱回仿真单独作为新的验证环节；本次没有启动额外电磁求解。

## 文件

- `validation_overview.png`：曲线、长度误差、周期角误差及总体结果。
- `analysis.json`：独立复算、周期最优常数及几何合法性。
- `metrics.json` / `history.json`：原始训练记录。
- `test_predictions.csv`：10个留出样本的原始预测。
- `training.log`：原始训练日志。
- 本地最佳权重：`{run/'best.pt'}`。
- 训练配置：`configs/train_au100_metal8_100.json`。
'''
(out/'report.md').write_text(note,encoding='utf-8')
if Path(__file__).resolve() != root/'analyze_train100.py':
    shutil.copy2(Path(__file__),root/'analyze_train100.py')
print(json.dumps({key:results[key] for key in ('improvement_vs_arithmetic_mean_percent','improvement_vs_periodic_constant_percent','predicted_geometry_valid_count')},indent=2))
print('REPORT',out/'report.md')
