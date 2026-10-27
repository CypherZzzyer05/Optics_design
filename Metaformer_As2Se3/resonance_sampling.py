"""Fixed-budget sampling of global spectra and input-detected resonance windows.

The grid contains continuous indices in the existing 1000-point wavelength
axis. Linear interpolation changes where the model reads that axis; it does
not increase the physical or numerical resolution of the source spectrum.
Only a smoothed copy is used to locate windows. Their values are sampled from
the caller's original, training-set-normalized reflectance.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
import torch


def _window_grid(left, right, tokens):
    return np.linspace(left, right, tokens * 10, dtype=np.float32).reshape(tokens, 10)


def build_extra_sampling_grid(raw_spectrum, mode):
    """Return 32 extra 10-point token grids per raw input spectrum.

    ``raw_spectrum`` is [batch,1000] on CPU or GPU. ``uniform_extra`` adds
    uniform coverage; ``resonance`` locates up to two prominent valleys using
    an 11-point, third-order Savitzky-Golay copy. Detection uses no labels or
    fitted dataset statistics. Each valley receives 16 tokens over four times
    its half-prominence width, with a pre-clipping span of 16..240 indices.
    Windows are clipped at the data boundaries without shifting their center.
    Missing windows receive predetermined global coverage. Output is float32
    [batch,32,10] on the input device and does not backpropagate through peak
    detection.
    """
    if raw_spectrum.ndim != 2 or raw_spectrum.shape[-1] != 1000:
        raise ValueError("Expected raw reflectance [batch,1000]")
    if mode not in ("uniform_extra", "resonance"):
        raise ValueError("Sampling mode must be uniform_extra or resonance")
    uniform = torch.linspace(0, 999, 320, dtype=torch.float32,
                             device=raw_spectrum.device).reshape(1, 32, 10)
    if mode == "uniform_extra":
        return uniform.expand(raw_spectrum.shape[0], -1, -1)

    raw = raw_spectrum.detach().to(device="cpu", dtype=torch.float32).numpy()
    smooth = savgol_filter(raw, 11, 3, axis=-1, mode="interp")
    grids = np.broadcast_to(_window_grid(0, 999, 32), (len(raw), 32, 10)).copy()
    for row, spectrum in enumerate(smooth):
        peaks, properties = find_peaks(-spectrum, prominence=.01, distance=20)
        if not len(peaks):
            continue
        # Strongest two first, then wavelength order. Stable tie handling is
        # fixed by the peak index ordering returned by scipy.
        chosen = np.argsort(properties["prominences"], kind="stable")[-2:]
        chosen_peaks = np.sort(peaks[chosen])
        widths = peak_widths(-spectrum, chosen_peaks, rel_height=.5)[0]
        for block, (peak, width) in enumerate(zip(chosen_peaks, widths)):
            span = np.clip(4 * width, 16, 240)
            left, right = max(0, peak - span / 2), min(999, peak + span / 2)
            grids[row, block * 16:(block + 1) * 16] = _window_grid(left, right, 16)
        if len(chosen_peaks) == 1:
            grids[row, 16:] = _window_grid(0, 999, 16)
    return torch.from_numpy(grids).to(device=raw_spectrum.device)


def linear_sample(normalized, grid):
    """Interpolate [batch,1,1000] values at [batch,tokens,points] indices.

    Sampling is differentiable with respect to the original values. Peak
    selection itself is intentionally detached in build_extra_sampling_grid.
    """
    if normalized.ndim != 3 or normalized.shape[1:] != (1, 1000):
        raise ValueError("Expected normalized reflectance [batch,1,1000]")
    if grid.ndim != 3 or grid.shape[0] != normalized.shape[0]:
        raise ValueError("Grid must be [batch,tokens,points] with matching batch")
    locations = grid.to(device=normalized.device, dtype=torch.float32).clamp(0, 999)
    flat = locations.flatten(1)
    lower = flat.floor().long()
    upper = (lower + 1).clamp_max(999)
    fraction = flat - lower
    values = normalized[:, 0].float()
    sampled = values.gather(1, lower) * (1 - fraction) + values.gather(1, upper) * fraction
    return sampled.reshape(grid.shape)


def continuous_position_embedding(center_indices, spans, dim):
    """Encode absolute token centers and index spans without trainable weights.

    Inputs are [batch,tokens]. A native 10-point token has center 4.5+10*j
    and span 9, giving exactly the original sinusoidal position of token j.
    Span uses a second sinusoid of log(span/9), with its zero-input encoding
    subtracted so native spans add zero.
    """
    centers = center_indices.float()
    spans = spans.to(device=centers.device, dtype=torch.float32)
    even_indices = torch.arange(0, dim, 2, dtype=torch.float32, device=centers.device)
    frequencies = torch.exp(-even_indices * (
        torch.log(torch.tensor(10000., device=centers.device)) / dim))
    position_angles = ((centers - 4.5) / 10)[..., None] * frequencies
    span_angles = torch.log(spans / 9)[..., None] * frequencies
    encoding = torch.empty(*centers.shape, dim, device=centers.device, dtype=torch.float32)
    encoding[..., 0::2] = position_angles.sin() + span_angles.sin()
    encoding[..., 1::2] = (position_angles[..., :dim // 2].cos()
                           + (span_angles[..., :dim // 2].cos() - 1))
    return encoding
