"""Export a completed pilot, verified checkpoint and all test plots for review."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from plot_forward_results import plot_results
from predict_forward import load_forward_checkpoint
from train_forward_metaformer import configure_torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(run, destination, model_path):
    run, destination, model_path = Path(run), Path(destination), Path(model_path)
    if destination.exists() or model_path.exists():
        raise FileExistsError('Choose fresh result and model paths; existing exports are preserved')
    if json.loads((run/'status.json').read_text())['state'] != 'completed':
        raise ValueError('Export requires a completed run')
    metrics = json.loads((run/'metrics.json').read_text())
    metadata = json.loads((run/'run.json').read_text())
    if sha(run/'best.pt') != metrics['best_checkpoint_sha256']:
        raise ValueError('Checkpoint checksum differs from completed evaluation')
    for name, digest in metadata['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Training source changed after this run: {name}')
    configure_torch()
    model, _ = load_forward_checkpoint(run/'best.pt')
    with np.load(run/'predictions.npz', allow_pickle=False) as arrays:
        x = torch.from_numpy(arrays['test_geometry_01'][:1].copy())
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        elapsed = []
        for _ in range(100):
            start = time.perf_counter()
            model(x)
            elapsed.append((time.perf_counter()-start)*1000)
    destination.mkdir(parents=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for name in ['run.json','history.json','metrics.json','training.log','predictions.npz']:
        shutil.copy2(run/name, destination/name)
        if name.endswith(('.json', '.log')):
            path = destination/name
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    shutil.copy2(run/'best.pt', model_path)
    plot_results(destination)
    test = metrics['metrics']['test']
    baseline = metrics['train_mean_baseline']['test']
    summary = {'task':'geometry_to_reflectance_only', 'dataset_sha256':metadata['dataset_sha256'],
               'quality_level':metadata['quality_level'], 'model_parameters':metadata['model_parameters'],
               'epochs_completed':metrics['epochs_completed'], 'best_epoch':metrics['best_epoch'],
               'optimizer_steps':metrics['optimizer_steps'], 'training_seconds':metrics['training_seconds'],
               'test_mse':test['mse'], 'test_mae':test['mae'], 'baseline_test_mse':baseline['mse'],
               'mse_improvement_percent':100*(1-test['mse']/baseline['mse']),
               'matched_valleys':test['matched_valleys'], 'target_valleys':test['target_valleys'],
               'matched_center_mae_nm':test['matched_center_mae_nm'],
               'matched_depth_mae':test['matched_depth_mae'],
               'cpu_checkpoint_max_abs_difference':metrics['cpu_checkpoint_max_abs_difference'],
               'cpu_forward_latency_ms':{'median':float(np.median(elapsed)),
                   'mean':float(np.mean(elapsed)), 'p95':float(np.percentile(elapsed,95)),
                   'batch_size':1,'threads':torch.get_num_threads(),'warmup':10,'repeats':100,
                   'scope':'FP32 neural forward only; excludes loading, normalization, CSV and FDTD'},
               'model_checkpoint_sha256':sha(model_path),
               'limitation':'Single seed, 10 test samples; narrow valleys remain substantially inaccurate. Not a reliable design surrogate yet.'}
    (destination/'summary.json').write_bytes(json.dumps(summary,indent=2).encode('utf-8'))
    readme = f'''# 正向 Metaformer 首轮结果

七参数结构到 1000 点反射光谱，{metadata['model_parameters']:,} 个参数；未接入逆向闭环，未训练 T。

固定 80/10/10 划分、seed=42。完成 {metrics['epochs_completed']} 轮、{metrics['optimizer_steps']} 次实际更新，验证集按 min_delta=1e-5 的改善阈值选择第 {metrics['best_epoch']} 轮。测试集只在模型选择结束后评估，未根据测试结果调参。

| 测试指标 | 数值 |
|---|---:|
| 整谱 MSE | {test['mse']:.8f} |
| 训练集均值光谱基线 MSE | {baseline['mse']:.8f} |
| 相对基线的整谱 MSE 降低 | {summary['mse_improvement_percent']:.2f}% |
| 反射率 MAE | {test['mae']:.6f} |
| 两个主要目标谷匹配数 | {test['matched_valleys']}/{test['target_valleys']} |
| 已匹配谷中心 MAE | {test['matched_center_mae_nm']:.3f} nm |
| 已匹配谷谷底反射率 MAE | {test['matched_depth_mae']:.3f} |
| CPU 检查点重载与 GPU 最大差值 | {metrics['cpu_checkpoint_max_abs_difference']:.3g} |

**首版已经学到部分结构相关的宽带变化，窄谷仍被明显抹平或遗漏。** 整谱平均误差的改善不能证明双谷设计成功，尚不适合作为高精度逆向约束。匹配误差只统计匹配成功的谷，缺失的 {test['missing_valleys']} 个谷另计，不能忽略。

![全部测试样本](test_spectra.png)

![固定前三条测试样本的谷附近细节](valley_details.png)

![训练曲线](learning_curve.png)

图中蓝线为 FDTD 标签、橙线为神经网络预测、灰线为训练集均值。全部 10 条测试样本按 ID 排列；局部图固定选择 ID 最小的三条，不挑选表现好的样本。

在当前电脑 CPU 上，batch=1、4 线程、FP32 的网络前向中位耗时 {summary['cpu_forward_latency_ms']['median']:.3f} ms。计时已预热 10 次、测量 100 次；不含加载、归一化和 CSV 写入。这是运行速度，不是精度承诺。

原始数据 SHA256：`{metadata['dataset_sha256']}`。这是历史 v1 Au100/有限 SiO2 数据，质量为 pipeline_validation，不能混同新版结构或认证 Au 耗散。完整数值、来源指纹、预测数组和代码哈希保存在本目录 JSON/NPZ 中；可移植权重位于 `../../models/forward_au100_seed42.pt`。
'''
    (destination/'README.md').write_bytes(readme.encode('utf-8'))
    manifest = {'model_checkpoint':{'path':str(model_path.relative_to(ROOT)), 'sha256':sha(model_path)},
                'files':{path.name:sha(path) for path in destination.iterdir() if path.is_file()}}
    (destination/'manifest.json').write_bytes(json.dumps(manifest,indent=2).encode('utf-8'))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=ROOT/'results/forward_au100_seed42')
    parser.add_argument('--model-path',type=Path,default=ROOT/'models/forward_au100_seed42.pt')
    args=parser.parse_args()
    print(json.dumps(export(args.run,args.destination,args.model_path),indent=2))


if __name__=='__main__':
    main()
