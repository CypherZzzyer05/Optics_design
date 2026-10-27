"""Lightweight train/validation signal diagnostics; no torch, test, or FDTD."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'examples/jc220_topmetal_1000/dataset.npz'
OUT = ROOT / 'results/topmetal_refine'


def errors(prediction, target, widths):
    """Same joint mirror (odd,even)->(1-even,1-odd) as the NN metrics."""
    def residual(t):
        diff = prediction-t
        diff[:, 5:] = np.remainder(diff[:, 5:]+.5, 1.)-.5
        return diff
    ordinary = residual(target)
    mirrored = target.copy()
    mirrored[:, 5:] = 1-target[:, [6, 5]]
    alternative = residual(mirrored)
    use = np.mean(alternative**2, 1) < np.mean(ordinary**2, 1)
    mirror = np.where(use[:, None], alternative, ordinary)
    return {name: {'mse': float(np.mean(e**2)),
                   'length_mse': float(np.mean(e[:, :5]**2)),
                   'mae_nm': (np.abs(e[:, :5])*widths[:5]).mean(0).tolist(),
                   'mae_deg': (np.abs(e[:, 5:])*widths[5:]).mean(0).tolist()}
            for name, e in [('periodic', ordinary), ('mirror_equivalent', mirror)]}


def cyclic_targets(labels, canonicalize=True):
    # Select one of the TWO JOINT mirror representatives, never per-angle minima.
    canonical = labels.copy()
    if canonicalize:
        use = labels[:, 5:].sum(1) < 1.
        canonical[use, 5:] = 1-labels[use][:, [6, 5]]
    phase = canonical[:, 5:] * (2*np.pi)
    pairs = np.stack((np.cos(phase), np.sin(phase)), -1).reshape(len(labels), 4)
    return np.concatenate((canonical[:, :5], pairs), 1)


def decode(values):
    pairs = values[:, 5:].reshape(-1, 2, 2)
    angles = np.remainder(np.arctan2(pairs[..., 1], pairs[..., 0])/(2*np.pi), 1.)
    return np.concatenate((np.clip(values[:, :5], 0., 1.), angles), 1)


def supplement(result, x, vx, y, vy, widths):
    """Fit constant angles on train only; compare two ridge labels at fixed alpha."""
    training_mean = y.mean(0)
    base_train = np.broadcast_to(training_mean, y.shape).copy()

    def objective(angles):
        base_train[:, 5:] = angles
        return errors(base_train, y, widths)['mirror_equivalent']['mse']

    fit = differential_evolution(objective, [(0., 1.), (0., 1.)], seed=20261027,
                                 maxiter=100, tol=1e-10, polish=True)
    fitted = training_mean.copy()
    fitted[5:] = fit.x
    original_validation = np.broadcast_to(training_mean, vy.shape)
    fitted_validation = np.broadcast_to(fitted, vy.shape)
    row = next(r for r in result['ridge'] if r['representation']=='standardized_floor_0_02')
    alpha = row['selected_by_validation']['alpha']
    mean, scale = x.mean(0), np.maximum(x.std(0), .02)
    tx, validation_x = (x-mean)/scale, (vx-mean)/scale
    gram = tx@tx.T
    values, vectors = eigh(gram, check_finite=False)
    values = np.maximum(values, 0.)
    labels = cyclic_targets(y, canonicalize=False)
    label_mean = labels.mean(0)
    coefficients = vectors@((vectors.T@(labels-label_mean))/(values[:, None]+alpha))
    prediction = decode((validation_x@tx.T)@coefficients+label_mean)
    result['angle_baseline_supplement'] = {
        'policy': 'Added train/validation diagnostic only. Optimize one constant angle pair on 800 training labels under joint mirror-periodic MSE; five dimensions remain training means. Validation never fits the constant. This separates symmetry-quotient gain from spectrum-dependent information.',
        'optimization': {'method': 'scipy differential_evolution plus polish', 'seed': 20261027,
                         'success': bool(fit.success), 'evaluations': int(fit.nfev),
                         'training_joint_mirror_mse': float(fit.fun)},
        'training_mean_angles_deg': (training_mean[5:]*90-45).tolist(),
        'fitted_constant_angles_deg': (fitted[5:]*90-45).tolist(),
        'training_mean_constant_validation': errors(original_validation, vy, widths),
        'fitted_constant_validation': errors(fitted_validation, vy, widths),
        'ridge_same_alpha_policy': 'Existing ridge already uses the canonical joint representative. Add uncanonicalized circular labels once at the exact existing alpha; no further alpha search.',
        'ridge_representation': 'standardized_floor_0_02', 'ridge_alpha': alpha,
        'ridge_canonical_validation': row['selected_by_validation']['validation'],
        'ridge_uncanonical_circular_validation': errors(prediction, vy, widths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement-only', action='store_true',
                        help='Reuse existing diagnostics and add the fixed constant-angle comparison')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA, allow_pickle=False) as archive:
        # Only these two split index arrays are consumed. The test split stays unused.
        ti, vi = archive['train_indices'], archive['val_indices']
        x = archive['spectra'][ti].astype(np.float64)
        vx = archive['spectra'][vi].astype(np.float64)
        y, vy = archive['geometry_01'][ti].astype(np.float64), archive['geometry_01'][vi].astype(np.float64)
        wavelengths = archive['wavelength_nm'].copy()
        widths = archive['upper_values']-archive['lower_values']
        ids, vids = archive['sample_ids'][ti], archive['sample_ids'][vi]
        names = archive['parameter_names'].tolist()
    if args.supplement_only:
        result = json.loads((OUT/'signal_diagnosis.json').read_text(encoding='utf-8'))
        supplement(result, x, vx, y, vy, widths)
        (OUT/'signal_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result['angle_baseline_supplement'], ensure_ascii=False, indent=2))
        return
    mean, std = x.mean(0), x.std(0)
    labels, label_mean = cyclic_targets(y), cyclic_targets(y).mean(0)
    constant = np.broadcast_to(y.mean(0), vy.shape)
    result = {'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'split_policy': '800 train spectra/labels fit all scalers and regressions; 100 validation only evaluates diagnostics; test indices, spectra and labels are not consumed.',
        'train_sample_ids': ids.tolist(), 'validation_sample_ids': vids.tolist(),
        'angle_policy': 'Ridge targets canonicalize the JOINT mirror by u_odd+u_even>=1, then regress cos/sin(2*pi*u) for each square-period angle. Decode atan2, evaluate BOTH original-periodic and JOINT mirror equivalent errors. This is a diagnostic linear baseline, not the NN training objective.',
        'ridge_policy': 'Three fixed representations, four fixed regularization ratios; alpha=ratio*mean(diag(X_train@X_train.T)); select one ratio per representation by validation joint-mirror MSE. No test evaluation or production model selection.',
        'interpretation': 'Weak linear/nearest-neighbor prediction or nearby spectra with different geometries do not prove physical non-identifiability. These checks only guide subsequent neural train/validation ablations.',
        'constant_validation': errors(constant, vy, widths), 'normalization': {}, 'ridge': [], 'nearest_neighbor': []}
    for lo, hi in [(700, 1000), (1000, 1300), (1300, 1701)]:
        mask = (wavelengths >= lo) & (wavelengths < hi)
        result['normalization'][f'{lo}_{min(hi,1700)}_nm'] = {
            'points': int(mask.sum()), 'std_min': float(std[mask].min()),
            'std_median': float(np.median(std[mask])), 'std_max': float(std[mask].max()),
            'points_clamped_at_0_02': int((std[mask]<.02).sum()),
            'minimum_normalized_std_at_0_02': float((std[mask]/np.maximum(std[mask], .02)).min())}
    centered = x-mean
    label_centered = y[:, :5]-y[:, :5].mean(0)
    correlation = centered.T@label_centered / (len(x)*std[:, None]*label_centered.std(0))
    result['training_length_correlations'] = {
        name: {'max_abs_pearson': float(np.abs(correlation[:, i]).max()),
               'peak_wavelength_nm': float(wavelengths[np.abs(correlation[:, i]).argmax()]),
               'max_abs_pearson_above_1300_nm': float(np.abs(correlation[wavelengths>1300, i]).max())}
        for i, name in enumerate(names[:5])}
    for representation, scale in [('raw_centered', np.ones_like(std)),
                                  ('standardized_floor_0_02', np.maximum(std, .02)),
                                  ('standardized_floor_0_002', np.maximum(std, .002))]:
        tx, valid_x = centered/scale, (vx-mean)/scale
        gram = tx@tx.T
        eigenvalues, vectors = eigh(gram, check_finite=False)
        eigenvalues = np.maximum(eigenvalues, 0.)
        projection = vectors.T@(labels-label_mean)
        cross = valid_x@tx.T
        scale_alpha = np.diag(gram).mean()
        trials = []
        for ratio in (.001, .01, .1, 1.):
            alpha = ratio*scale_alpha
            coefficients = vectors@(projection/(eigenvalues[:, None]+alpha))
            prediction = decode(cross@coefficients+label_mean)
            trials.append({'regularization_ratio': ratio, 'alpha': float(alpha),
                           'validation': errors(prediction, vy, widths)})
        chosen = min(trials, key=lambda row: row['validation']['mirror_equivalent']['mse'])
        result['ridge'].append({'representation': representation, 'trials': trials,
                                'selected_by_validation': chosen})
        distances = (valid_x**2).sum(1, keepdims=True)+(tx**2).sum(1)[None]-2*cross
        neighbor = distances.argmin(1)
        result['nearest_neighbor'].append({'representation': representation,
            'validation': errors(y[neighbor], vy, widths),
            'pairs': [{'validation_sample_id': int(v), 'train_neighbor_id': int(ids[n]),
                       'spectrum_rmse_raw': float(np.sqrt(np.mean((vx[j]-x[n])**2)))}
                      for j, (v, n) in enumerate(zip(vids, neighbor))]})
    supplement(result, x, vx, y, vy, widths)
    (OUT/'signal_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(wavelengths, std, label='训练谱标准差')
    axes[0].axhline(.02, color='#b46a43', ls='--', label='现有下限 0.02')
    axes[0].set(xlabel='波长 / nm', ylabel='R 标准差', title='仅800条训练谱：方差与标准化下限')
    for i, name in enumerate(['P', 'L', 'H', 'dX', 'dY']):
        axes[1].plot(wavelengths, np.abs(correlation[:, i]), label=name)
    axes[1].set(xlabel='波长 / nm', ylabel='|Pearson r|', title='训练谱各波长与五尺寸的线性关联')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(OUT/'signal_diagnosis.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'normalization': result['normalization'],
        'correlations': result['training_length_correlations'],
        'constant': result['constant_validation'],
        'ridge': [{k:v for k,v in r.items() if k!='trials'} for r in result['ridge']],
        'nearest_neighbor': [{k:v for k,v in r.items() if k!='pairs'} for r in result['nearest_neighbor']]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
