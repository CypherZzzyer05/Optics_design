"""Spectrum and explicitly matched valley metrics; no Q-factor claims."""
import numpy as np
from scipy.signal import find_peaks, peak_widths


def valleys(spectrum, wavelength_nm, prominence=.05, limit=None):
    spectrum = np.asarray(spectrum, dtype=float)
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    indices, properties = find_peaks(-spectrum, prominence=prominence)
    if limit is not None:
        order = np.argsort(-properties['prominences'], kind='stable')[:limit]
        indices = indices[order]
    indices = np.sort(indices)
    if not len(indices):
        return []
    _, _, left, right = peak_widths(-spectrum, indices, rel_height=.5)
    positions = np.arange(len(spectrum))
    widths = np.interp(right, positions, wavelength_nm) - np.interp(left, positions, wavelength_nm)
    return [{'center_nm': float(wavelength_nm[i]), 'reflectance': float(spectrum[i]),
             'half_prominence_width_nm': float(width), 'index': int(i)}
            for i, width in zip(indices, widths)]


def spectral_metrics(prediction, target, wavelength_nm, sample_ids=None):
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    wave = np.asarray(wavelength_nm, dtype=float)
    if (prediction.shape != target.shape or target.ndim != 2 or not len(target)
            or wave.shape != (target.shape[1],) or not np.all(np.diff(wave) > 0)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('Metrics require finite matching spectra and increasing wavelengths')
    ids = np.arange(len(target)) if sample_ids is None else np.asarray(sample_ids)
    if ids.shape != (len(target),):
        raise ValueError('One sample ID is required per spectrum')
    squared = (prediction - target) ** 2
    records, center_errors, depth_errors, width_errors, windows = [], [], [], [], []
    missing = total = 0
    for row, (pred, truth) in enumerate(zip(prediction, target)):
        reference = valleys(truth, wave, prominence=.05, limit=2)
        candidates = valleys(pred, wave, prominence=.02)
        used = set()
        for valley in reference:
            total += 1
            mask = abs(wave - valley['center_nm']) <= 15
            windows.extend(squared[row, mask].tolist())
            choices = [(abs(v['center_nm'] - valley['center_nm']), j, v)
                       for j, v in enumerate(candidates) if j not in used
                       and abs(v['center_nm'] - valley['center_nm']) <= 15]
            record = {'sample_id': int(ids[row]), 'target': valley, 'predicted': None}
            if choices:
                distance, j, matched = min(choices, key=lambda item: item[:2])
                used.add(j)
                record['predicted'] = matched
                center_errors.append(distance)
                depth_errors.append(abs(matched['reflectance'] - valley['reflectance']))
                width_errors.append(abs(matched['half_prominence_width_nm'] /
                                        valley['half_prominence_width_nm'] - 1))
            else:
                missing += 1
            records.append(record)
    average = lambda values: float(np.mean(values)) if len(values) else None
    return {'samples': len(target), 'mse': float(squared.mean()),
            'rmse': float(np.sqrt(squared.mean())), 'mae': float(np.abs(prediction-target).mean()),
            'per_sample_mse': squared.mean(axis=1).tolist(),
            'target_valleys': total, 'matched_valleys': total-missing, 'missing_valleys': missing,
            'valley_window_mse': average(windows),
            'matched_center_mae_nm': average(center_errors),
            'matched_depth_mae': average(depth_errors),
            'matched_width_relative_mae': average(width_errors),
            'valley_definition': 'Two strongest reference minima with prominence>=0.05; nearest unused predicted minimum with prominence>=0.02 within 15 nm. Width is half-prominence, not modal FWHM.',
            'valley_matches': records}
