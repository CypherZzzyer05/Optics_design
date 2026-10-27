"""Data-only spectral-neighbor audit; no training, test evaluation or FDTD.

The two distance definitions and the example-selection rules were declared
before this script was written. Neighbors are diagnostic comparisons, not a
replacement for the deployed inverse model.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/spectral_information_audit'
DATASET = ROOT / 'examples/jc220_topmetal_1000/dataset.npz'
PARAMETERS = ('P', 'L', 'H', 'dX', 'dY', 'theta_odd', 'theta_even')
DISPLAY = ('P', 'L', 'H', 'dX', 'dY', '奇数角', '偶数角')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse_matrix(left, right):
    squared = ((left * left).sum(1)[:, None] + (right * right).sum(1)[None]
               - 2 * left @ right.T) / left.shape[1]
    return np.sqrt(np.maximum(squared, 0))


def geometry_error(prediction, target, lower, upper):
    """Mirror the target jointly, matching spectral_inverse.parameter_errors."""
    difference = prediction - target
    direct = (difference[5:] + 45) % 90 - 45
    alternate = (prediction[5:] + target[[6, 5]] + 45) % 90 - 45
    mirrored = bool(np.dot(alternate, alternate) < np.dot(direct, direct))
    difference = np.r_[difference[:5], alternate if mirrored else direct]
    return difference, difference / (upper - lower), mirrored


def self_check():
    points = np.array([[0., 0.], [3., 4.]])
    np.testing.assert_allclose(rmse_matrix(points, points),
                               [[0, 5 / np.sqrt(2)], [5 / np.sqrt(2), 0]])
    lower, upper = np.zeros(7), np.r_[np.ones(5), 90., 90.]
    truth = np.r_[np.zeros(5), 10., 20.]
    mirror = np.r_[np.zeros(5), -20., -10.]
    error, _, chosen = geometry_error(mirror, truth, lower, upper)
    np.testing.assert_array_equal(error, np.zeros(7))
    assert chosen
    periodic = np.r_[np.zeros(5), 100., 110.]
    error, _, _ = geometry_error(periodic, truth, lower, upper)
    np.testing.assert_array_equal(error, np.zeros(7))
    asymmetric = np.r_[np.zeros(5), -18., -5.]
    error, _, chosen = geometry_error(asymmetric, truth, lower, upper)
    np.testing.assert_array_equal(error[5:], [2., 5.])
    assert chosen


def check_error_convention(predictions, targets, lower, upper):
    """Check every row against the network's existing error implementation."""
    import torch
    from spectral_inverse import parameter_errors
    torch.set_num_threads(4)
    own = np.stack([geometry_error(p, t, lower, upper)[1]
                    for p, t in zip(predictions, targets)])
    prediction_01 = torch.from_numpy((predictions - lower) / (upper - lower))
    target_01 = torch.from_numpy((targets - lower) / (upper - lower))
    expected = parameter_errors(prediction_01, target_01, mirror=True).numpy()
    np.testing.assert_allclose(own, expected, rtol=1e-12, atol=1e-12)
    return {'rows': len(targets), 'normalized_max_abs_difference': float(np.max(np.abs(own - expected))),
            'passed': True}


def comparison_row(mode, query_id, neighbor_id, metric_distance, query, neighbor,
                   truth, prediction, lower, upper):
    difference, normalized, mirrored = geometry_error(prediction, truth, lower, upper)
    delta = neighbor - query
    return {
        'metric': mode, 'query_sample_id': int(query_id),
        'neighbor_sample_id': int(neighbor_id), 'metric_rmse': float(metric_distance),
        'raw_reflectance_rmse': float(np.sqrt(np.mean(delta * delta))),
        'raw_reflectance_max_abs_difference': float(np.max(np.abs(delta))),
        'joint_mirror_applied': mirrored,
        'joint_mirror_normalized_mse': float(np.mean(normalized * normalized)),
        **{f'abs_error_{name}_{"nm" if i < 5 else "deg"}': float(abs(difference[i]))
           for i, name in enumerate(PARAMETERS)},
    }


def quantiles(values):
    return {str(p): float(np.quantile(values, p)) for p in (0, .01, .05, .5, .95, 1)}


def plot_pair(ids, spectra, geometry, wavelength, lower, upper, rule):
    first, second = ids
    a, b = spectra
    delta = b - a
    maximum_index = int(np.argmax(np.abs(delta)))
    maximum_wavelength = float(wavelength[maximum_index])
    rmse = float(np.sqrt(np.mean(delta * delta)))
    max_abs = float(np.max(np.abs(delta)))
    stem = f'pair_{first}_{second}'
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10))
    fig.subplots_adjust(left=.085, right=.98, top=.91, bottom=.08, hspace=.43)
    fig.suptitle(f'训练谱对 {first} / {second}：相近不等于同谱', fontsize=17, y=.975)
    fig.text(.5, .935, f'原谱RMSE={100 * rmse:.3f}个百分点；最大差={100 * max_abs:.3f}个百分点',
             ha='center', color='#394d60', fontsize=11)
    for ax in (axes[0], axes[2]):
        ax.plot(wavelength, a, label=f'样本 {first}', color='#307ca3', lw=1.6)
        ax.plot(wavelength, b, label=f'样本 {second}', color='#c17d3b', lw=1.5, linestyle='--')
        ax.set_ylabel('反射率 R')
        ax.legend(loc='best')
    axes[0].set_title('完整输入光谱', loc='left', fontsize=11)
    axes[0].set_xlim(wavelength[0], wavelength[-1])
    axes[1].plot(wavelength, 100 * delta, color='#6d5b87', lw=1.4)
    axes[1].axhline(0, color='#6f7d88', lw=.7)
    axes[1].scatter([maximum_wavelength], [100 * delta[maximum_index]], color='#9d4f55', s=28)
    axes[1].set(ylabel='ΔR / 百分点', xlim=(wavelength[0], wavelength[-1]))
    axes[1].set_title(f'差谱：样本{second} − 样本{first}；最大差位于{maximum_wavelength:.1f} nm',
                      loc='left', fontsize=11)
    axes[2].set_xlim(max(float(wavelength[0]), maximum_wavelength - 20),
                     min(float(wavelength[-1]), maximum_wavelength + 20))
    axes[2].set_title('最大差附近局部放大：固定显示中心两侧各20 nm', loc='left', fontsize=11)
    axes[2].axvline(maximum_wavelength, color='#8b6571', lw=.8, alpha=.65)
    for ax in axes:
        ax.set_xlabel('波长 / nm')
        ax.grid(alpha=.16)
    fig.text(.5, .025, '全谱平均距离可能掩盖局部共振差异；本图未使用物理噪声门槛，也不是受控单参数对照。',
             ha='center', color='#526472', fontsize=10)
    fig.savefig(OUT / f'{stem}.png', dpi=175)
    plt.close(fig)
    difference, normalized, mirrored = geometry_error(geometry[1], geometry[0], lower, upper)
    rows = [{'parameter': name, 'unit': 'nm' if i < 5 else 'deg',
             f'sample_{first}': float(geometry[0, i]),
             f'sample_{second}_original': float(geometry[1, i]),
             'joint_mirror_absolute_difference': float(abs(difference[i]))}
            for i, name in enumerate(PARAMETERS)]
    write_csv(OUT / f'{stem}_parameters.csv', rows)
    return {'sample_ids': [int(first), int(second)], 'selection_rule': rule,
            'raw_rmse': rmse, 'raw_max_abs_difference': max_abs,
            'max_difference_wavelength_nm': maximum_wavelength,
            'joint_mirror_applied': mirrored,
            'joint_mirror_absolute_parameter_differences': np.abs(difference).tolist(),
            'joint_mirror_normalized_mse': float(np.mean(normalized * normalized)),
            'geometry_values': geometry.tolist(), 'figure': f'{stem}.png',
            'parameters_csv': f'{stem}_parameters.csv'}


def main():
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                         'font.size': 10})
    with np.load(DATASET, allow_pickle=False) as data:
        train_index, val_index = data['train_indices'], data['val_indices']
        train, val = data['spectra'][train_index].astype(np.float64), data['spectra'][val_index].astype(np.float64)
        geometry, val_geometry = data['geometry_values'][train_index], data['geometry_values'][val_index]
        ids, val_ids = data['sample_ids'][train_index], data['sample_ids'][val_index]
        wavelength = data['wavelength_nm'].copy()
        lower, upper = data['lower_values'], data['upper_values']
    assert train.shape == (800, 1000) and val.shape == (100, 1000)
    assert not set(ids.tolist()) & set(val_ids.tolist())
    scale = np.maximum(train.std(0, ddof=0), .02)
    train_rows, validation_rows, metrics, examples, convention_checks = [], [], {}, [], {}
    for mode, denominator in (('raw', np.ones(1000)), ('train_standardized', scale)):
        within = rmse_matrix(train / denominator, train / denominator)
        np.fill_diagonal(within, np.inf)
        nearest = within.argmin(1)
        train_rows.extend(comparison_row(mode, ids[i], ids[j], within[i, j], train[i], train[j],
                                        geometry[i], geometry[j], lower, upper)
                          for i, j in enumerate(nearest))
        between = rmse_matrix(val / denominator, train / denominator)
        val_nearest = between.argmin(1)
        convention_checks[mode] = {
            'train_1nn': check_error_convention(geometry[nearest], geometry, lower, upper),
            'validation_1nn': check_error_convention(geometry[val_nearest], val_geometry, lower, upper)}
        this_validation = [comparison_row(mode, val_ids[i], ids[j], between[i, j], val[i], train[j],
                                         val_geometry[i], geometry[j], lower, upper)
                           for i, j in enumerate(val_nearest)]
        validation_rows.extend(this_validation)
        errors = np.array([[row[f'abs_error_{name}_{"nm" if i < 5 else "deg"}']
                            for i, name in enumerate(PARAMETERS)] for row in this_validation])
        metrics[mode] = {'train_1nn_distance_quantiles': quantiles(within[np.arange(len(train)), nearest]),
                         'validation_1nn_distance_quantiles': quantiles(between[np.arange(len(val)), val_nearest]),
                         'validation_joint_mirror_normalized_mse': float(np.mean([row['joint_mirror_normalized_mse'] for row in this_validation])),
                         'validation_mae_nm': errors[:, :5].mean(0).tolist(),
                         'validation_mae_deg': errors[:, 5:].mean(0).tolist()}
        if mode == 'raw':
            unique = np.triu(np.ones(within.shape, dtype=bool), 1)
            conditions = ((unique, 'all_train_pairs_minimum_raw_rmse'),
                          (unique & (np.abs(geometry[:, 4, None] - geometry[None, :, 4]) >= 20),
                           'minimum_raw_rmse_among_train_pairs_with_abs_dY_difference_ge_20_nm'))
            for condition, rule in conditions:
                i, j = np.unravel_index(np.where(condition, within, np.inf).argmin(), within.shape)
                direct = np.sqrt(np.mean((train[i] - train[j]) ** 2))
                np.testing.assert_allclose(within[i, j], direct, rtol=1e-9, atol=1e-12)
                examples.append(plot_pair((ids[i], ids[j]), train[[i, j]], geometry[[i, j]],
                                          wavelength, lower, upper, rule))
    write_csv(OUT / 'train_1nn.csv', train_rows)
    write_csv(OUT / 'validation_1nn_errors.csv', validation_rows)
    baseline_path = ROOT / 'results/topmetal1000/validation_results.json'
    records = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline = [r['validation']['mirror_equivalent'] for r in records
                if r['variant'] == 'circular' and r['seed'] in (42, 43, 44)]
    assert len(baseline) == 3
    network = {key: np.mean([r[key] for r in baseline], axis=0).tolist()
               for key in ('mse', 'mae_nm', 'mae_deg')}
    validation_summary = {'split': 'validation only; no test data consumed',
                          'distance_metrics': metrics,
                          'error_convention_checks': convention_checks,
                          'network_three_seed_mean_validation': network,
                          'network_source': str(baseline_path),
                          'nearest_neighbor_role': 'diagnostic, not a deployed inverse predictor or error lower bound'}
    write_json(OUT / 'validation_1nn_errors.json', validation_summary)
    summary = {'scope': 'read-only dataset diagnostic; no training, FDTD or deployed model changes',
               'dataset': str(DATASET), 'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'split_counts': {'train': len(train), 'validation': len(val), 'test_consumed': 0},
               'parameter_names': list(PARAMETERS),
               'normalization': 'train float64 std(ddof=0), clipped below at 0.02; no fitted validation statistics',
               'threshold_policy': 'only descriptive distance quantiles; no physical/numerical noise floor claimed',
               'angle_policy': 'mirror target jointly, retain prediction odd/even coordinates, then square 90-degree periodic errors; matches spectral_inverse.parameter_errors',
               'distance_metrics': metrics, 'network_three_seed_mean_validation': network,
               'error_convention_checks': convention_checks,
               'examples': examples, 'analytic_self_checks_passed': True,
               'new_training_performed': False, 'new_fdtd_run_performed': False}
    write_json(OUT / 'summary.json', summary)
    write_report(summary)
    print(json.dumps({'report': str(OUT / 'README.md'), 'examples': [x['sample_ids'] for x in examples],
                      'val_1nn_dY_mae_nm': {name: m['validation_mae_nm'][4] for name, m in metrics.items()},
                      'network_validation_dY_mae_nm': network['mae_nm'][4]}, ensure_ascii=False, indent=2))


def write_report(summary):
    network, metrics = summary['network_three_seed_mean_validation'], summary['distance_metrics']
    labels = {'raw': '原反射谱距离', 'train_standardized': '训练集标准化距离'}
    rows = [(labels[name], item['validation_joint_mirror_normalized_mse'], item['validation_mae_nm'], item['validation_mae_deg'])
            for name, item in metrics.items()]
    rows.append(('现用网络（三种子均值）', network['mse'], network['mae_nm'], network['mae_deg']))
    errors = '\n'.join(f'| {name} | {mse:.6f} | ' + ' | '.join(f'{v:.3f}' for v in lengths + angles) + ' |'
                       for name, mse, lengths, angles in rows)
    quantile_rows = '\n'.join(f'| {labels[name]} | ' + ' | '.join(f'{item["train_1nn_distance_quantiles"][str(q)]:.6f}'
                                        for q in (0, .01, .05, .5, .95, 1)) + ' |'
                              for name, item in metrics.items())
    pair_sections = []
    for index, pair in enumerate(summary['examples']):
        a, b = pair['sample_ids']
        rule = '全部训练谱对中原谱RMSE最小' if index == 0 else '在ΔdY≥20 nm的训练谱对中原谱RMSE最小'
        parameter_rows = '\n'.join(f'| {name} | {pair["geometry_values"][0][i]:.3f} | '
            f'{pair["geometry_values"][1][i]:.3f} | {pair["joint_mirror_absolute_parameter_differences"][i]:.3f} | '
            f'{"nm" if i < 5 else "°"} |' for i, name in enumerate(DISPLAY))
        pair_sections.append(f'''### 训练谱对 {a} / {b}

选择规则：{rule}。原谱RMSE为 **{100 * pair['raw_rmse']:.3f}个百分点**，最大反射率差为 **{100 * pair['raw_max_abs_difference']:.3f}个百分点**，位于{pair['max_difference_wavelength_nm']:.1f} nm。较小的全谱平均差仍可伴随明显的局部差异，不能称为不可分辨或同谱。

![原谱、差谱及最大差位置放大]({pair['figure']})

| 参数 | 样本{a}原始值 | 样本{b}原始值 | 联合镜像后的绝对差 | 单位 |
|---|---:|---:|---:|---|
{parameter_rows}

两角共同选择目标（样本{a}）的原代表或镜像代表，再计算90°周期差，保持预测（样本{b}）的奇/偶坐标顺序，与现网络一致。表中前两列保留原始角标签，因此不能直接相减后与第四列混用。是否镜像目标：{pair['joint_mirror_applied']}。参数表另存为 [{pair['parameters_csv']}]({pair['parameters_csv']})。
''')
    report = f'''# 光谱近邻诊断：相近光谱下的结构差异

2026-10-27。仅使用修正材料及层序数据的800条训练和100条验证；没有读取测试索引或使用测试谱/标签。没有训练、FDTD或部署模型修改。数据SHA256：`{summary['dataset_sha256']}`。

**现用直接逆向网络的验证误差优于这两种1NN检索。** 本诊断没有把“给定光谱→结构”的任务改成查库，也没有据验证结果替换模型。它用于检查当前有限数据库里的光谱邻域和标签分散情况，不是本征不可辨识性或最优误差下界证明。

## 固定规则

- 原谱距离为1000点反射率差的RMSE，同时报告最大绝对差，避免全谱平均掩盖窄谷变化。
- 标准化距离用800条训练谱逐波长标准差，下限0.02，计算RMSE；均值在差分中抵消。计算为float64，标准差为总体口径 `ddof=0`，没有拟合验证统计。
- 训练内部1NN排除自身；验证1NN只查询训练库。两种距离分别展示，不按验证结果挑“最优距离”。
- 保持预测的奇/偶角顺序不变，给目标两角共同选择原代表或联合镜像 `(-θ偶,-θ奇)` 代表，再算90°周期差；五个长度不变。这与现网络 `spectral_inverse.parameter_errors` 的逐角MAE口径一致。七参MSE按原配置参数范围归一化。
- 两个示例分别按“全部训练谱对中原谱RMSE最小”和“ΔdY≥20 nm条件下原谱RMSE最小”选取。20 nm是已声明的示例筛选条件，不是物理可辨识阈值。最大差放大窗固定为中心两侧各20 nm。

## 邻域距离与验证比较

下表为800条训练谱各自最近邻距离的分位数。原谱距离单位是反射率，标准化距离无量纲。这些只是当前数据密度的描述，**不代表测量噪声或FDTD误差地板**。

| 距离定义 | 最小 | 1%分位 | 5%分位 | 中位数 | 95%分位 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
{quantile_rows}

验证100条的七参数误差如下。网络列沿用 `results/topmetal1000/validation_results.json` 的三个圆周头种子，仅读取validation字段，没有运行新的模型评价。这100条验证数据此前已用于模型选型和早停，当前比较不是新的盲测；这里只比较两种预先固定距离的1NN，不代表所有检索方法。

| 诊断/模型 | 联合镜像MSE | P MAE/nm | L MAE/nm | H MAE/nm | dX MAE/nm | dY MAE/nm | 奇角MAE/° | 偶角MAE/° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{errors}

两种1NN的dY误差为{metrics['raw']['validation_mae_nm'][4]:.2f}、{metrics['train_standardized']['validation_mae_nm'][4]:.2f} nm，现网络为{network['mae_nm'][4]:.2f} nm。网络优于直接近邻检索，说明其性能不能等同于简单查库。1NN较差也可能来自稀疏采样及距离定义，并不是神经网络可达到精度的下界。

{''.join(pair_sections)}

## 能与不能得出的结论

在示例80/925中，P、L、H、dX差异相对较小，但dY和角度同时变化，不能把谱相近完全归因于dY不敏感。此类非受控样本只提示可能的参数补偿或有限样本邻域分散；稀疏的七维数据库、选择距离和数值误差也会影响观察。

本批仍为金属局部8 nm、`pipeline_validation` 数值数据，完整器件的网格精度未得到普遍收敛认证。这里没有把既有单个器件的网格误差套给全部样本，也没有设定一个未经验证的噪声地板。当前谱对的最大差达到数个百分点，不能据平均RMSE直接宣称同谱或实验不可分辨。

要判断单个参数是否缺少可靠信息，需要固定其余参数的物理对照，以及可信数值/测量误差尺度。当前结果不证明增加观测一定有效，也不证明单谱能够唯一恢复七参；已经验证的镜像等价与这里的近似谱邻域应分别解释。

## 复现与文件

```powershell
python diagnose_spectral_neighbors.py
```

使用现项目已有NumPy、Matplotlib和PyTorch，逐行误差核对调用既有 `spectral_inverse.parameter_errors`。脚本内含解析距离、非对称镜像偏差和周期角度自检；所选最小谱对距离另与逐点直接RMSE核对。两种距离的800条训练1NN与100条验证1NN均独立核对逐参数归一化误差，结果保存在 `error_convention_checks`。完整输出见 [summary.json](summary.json)、[train_1nn.csv](train_1nn.csv)、[validation_1nn_errors.csv](validation_1nn_errors.csv)、[validation_1nn_errors.json](validation_1nn_errors.json)。未修改模型或生成器。
'''
    (OUT / 'README.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
