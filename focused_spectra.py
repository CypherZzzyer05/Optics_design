"""Select real frequency samples while preserving both reference troughs."""
import itertools
import numpy as np
from scipy.signal import find_peaks


def focused_indices(wavelength, reflection, extra_wavelengths=()):
    from explore_double_dips import double_pair
    w, r = np.asarray(wavelength), np.asarray(reflection)
    pair = double_pair(w, r)['dominant_pair']
    if pair is None:
        raise ValueError('Focused refinement requires a resolved reference double dip')
    mask = np.zeros(len(w), dtype=bool)
    mask[np.linspace(0, len(w)-1, 31, dtype=int)] = True
    for dip in pair['dips']:
        mask |= abs(w-dip['wavelength_nm']) <= 24
    # Retain baseline extrema so prominence/width comparisons use the same ridges.
    extrema = np.concatenate([find_peaks(sign*r, prominence=.01)[0] for sign in (-1, 1)])
    for index in extrema:
        mask |= abs(w-w[index]) <= 4
    for value in extra_wavelengths:
        index = int(np.argmin(abs(w-value)))
        if abs(w[index]-value) > 1e-5:
            raise ValueError('Power-check frequencies must be reference samples')
        mask[index] = True
    return np.flatnonzero(mask)


def common_spectra(reference_w, reference_r, fine_w, fine_r, allow_focused=False):
    from explore_double_dips import double_pair
    reference_w, fine_w = np.asarray(reference_w), np.asarray(fine_w)
    reference_r, fine_r = np.asarray(reference_r), np.asarray(fine_r)
    if np.array_equal(reference_w, fine_w):
        return reference_w, reference_r, fine_r, {'kind': 'identical_full_grid', 'points': len(reference_w)}
    if not allow_focused:
        raise ValueError('Use the same spectral grid for a spatial mesh comparison')
    indices = np.argmin(abs(fine_w[:, None]-reference_w[None, :]), axis=1)
    error = float(np.max(abs(fine_w-reference_w[indices])))
    if (error > 1e-5 or np.any(np.diff(indices) <= 0)
            or indices[0] != 0 or indices[-1] != len(reference_w)-1):
        raise ValueError('Focused samples must be a direct full-band subset of the reference grid')
    full_pair = double_pair(reference_w, reference_r)['dominant_pair']
    if full_pair is None:
        raise ValueError('Full reference grid has no resolved pair')
    for dip in full_pair['dips']:
        required = np.flatnonzero(abs(reference_w-dip['wavelength_nm']) <= 20)
        if not np.isin(required, indices).all():
            raise ValueError('Focused grid omits dense samples near a reference trough')
    selected_w, selected_r = reference_w[indices], reference_r[indices]
    found = double_pair(selected_w, selected_r)['dips']
    if len(found) < 2:
        raise ValueError('Reference pair disappears under focused sampling')
    matched = min(itertools.permutations(found, 2),
                  key=lambda dips: sum(abs(a['wavelength_nm']-b['wavelength_nm'])
                                       for a, b in zip(full_pair['dips'], dips)))
    biases = []
    for full, selected in zip(full_pair['dips'], matched):
        bias = {'center_change_nm': abs(selected['wavelength_nm']-full['wavelength_nm']),
                'R_min_change': abs(selected['reflectance_min']-full['reflectance_min']),
                'relative_width_change': abs(selected['half_prominence_width_nm']/full['half_prominence_width_nm']-1),
                'prominence_change': abs(selected['prominence']-full['prominence'])}
        if (bias['center_change_nm'] > 1e-5 or bias['R_min_change'] > 1e-12
                or bias['relative_width_change'] > .01 or bias['prominence_change'] > .01):
            raise ValueError('Focused sampling changes the reference trough measurements beyond its bias limits')
        biases.append(bias)
    return selected_w, selected_r, fine_r, {
        'kind': 'direct_subset_with_dense_trough_windows', 'reference_points': len(reference_w),
        'shared_points': len(selected_w), 'maximum_wavelength_matching_error_nm': error,
        'reference_sampling_bias_by_trough': biases,
        'scope': 'Both dense trough windows and selected full-band samples; whole-spectrum differences apply only to shared points'}
