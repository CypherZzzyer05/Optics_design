"""Small inverse transformer with explicit spectral views and joint angle symmetry.

Inputs remain raw reflectance on the saved wavelength grid. All fitted statistics
belong to the training split and are stored in the checkpoint.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_coeffs
import torch
from torch import nn
from torch.nn import functional as F

from metaformer import EncoderBlock, MetaformerConfig, SinusoidalPositionEncoding


def mirror_labels(target):
    """Normalized angles: (odd, even) -> (1-even, 1-odd)."""
    return torch.cat((target[..., :5], 1 - target[..., [6, 5]]), dim=-1)


def parameter_errors(prediction, target, mirror=False):
    """Choose ONE mirror representative jointly for the two angle coordinates."""
    difference = prediction - target
    angles = torch.remainder(difference[..., 5:] + .5, 1.) - .5
    if mirror:
        alternate = prediction[..., 5:] - mirror_labels(target)[..., 5:]
        alternate = torch.remainder(alternate + .5, 1.) - .5
        choose = alternate.square().sum(-1) < angles.square().sum(-1)
        angles = torch.where(choose[..., None], alternate, angles)
    return torch.cat((difference[..., :5], angles), dim=-1)


def inverse_loss(prediction, target, mirror=False):
    return parameter_errors(prediction.float(), target.float(), mirror).square().mean()


@dataclass(frozen=True)
class SpectralConfig:
    view: str = 'shape_fft'
    mirror_loss: bool = True
    spectrum_points: int = 1000
    embedding_dim: int = 96
    encoder_layers: int = 4
    dropout: float = .1
    wavelength_step_nm: float = 1000 / 999

    def __post_init__(self):
        if self.view not in ('raw', 'shape', 'shape_fft'):
            raise ValueError('Unknown spectral view')
        if self.spectrum_points != 1000 or self.wavelength_step_nm <= 0:
            raise ValueError('This experiment requires 1000 uniform wavelength points')


class SpectralViews(nn.Module):
    """R plus fixed 11-point SG slope/curvature; originals are never replaced."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = 1 if config.view == 'raw' else 3
        kernels = np.stack([savgol_coeffs(11, 3, deriv=order,
                           delta=config.wavelength_step_nm, use='dot') for order in (1, 2)])
        self.register_buffer('derivative_kernels', torch.tensor(kernels[:, None], dtype=torch.float32))
        self.register_buffer('mean', torch.zeros(1, channels, 1000))
        self.register_buffer('scale', torch.ones(1, channels, 1000))
        self.register_buffer('fft_mean', torch.zeros(1, 128))
        self.register_buffer('fft_scale', torch.ones(1, 128))
        self.register_buffer('fitted', torch.tensor(False))

    def raw_views(self, spectrum):
        values = spectrum[:, None]
        if self.config.view == 'raw':
            return values
        derivatives = F.conv1d(F.pad(values, (5, 5), mode='replicate'), self.derivative_kernels)
        return torch.cat((values, derivatives), dim=1)

    def fourier(self, spectrum):
        # Complex coefficients preserve phase; this is a wavelength-domain DFT,
        # not a temporal spectrum or a direct optical path length measurement.
        coefficients = torch.fft.rfft(spectrum.float(), norm='ortho')[:, :64]
        return torch.cat((coefficients.real, coefficients.imag), dim=-1)

    @torch.no_grad()
    def fit(self, training_spectra):
        values = self.raw_views(training_spectra.float())
        floors = torch.tensor([.02, .001, .0001][:values.shape[1]], device=values.device)[None, :, None]
        self.mean.copy_(values.mean(0, keepdim=True))
        self.scale.copy_(torch.maximum(values.std(0, unbiased=False, keepdim=True), floors))
        coefficients = self.fourier(training_spectra)
        self.fft_mean.copy_(coefficients.mean(0, keepdim=True))
        self.fft_scale.copy_(coefficients.std(0, unbiased=False, keepdim=True).clamp_min(.02))
        self.fitted.fill_(True)

    def forward(self, spectrum):
        if spectrum.ndim != 2 or spectrum.shape[1] != 1000:
            raise ValueError('Expected [batch,1000] raw reflectance')
        if not self.fitted:
            raise RuntimeError('Fit preprocessing on the training split first')
        # Keep finite differences and FFT in float32 even under training AMP.
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            values = (self.raw_views(spectrum.float()) - self.mean) / self.scale
            coefficients = (self.fourier(spectrum) - self.fft_mean) / self.fft_scale
        return values, coefficients


class SpectralInverse(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SpectralConfig()
        c = self.config
        channels = 1 if c.view == 'raw' else 3
        dim = c.embedding_dim
        self.views = SpectralViews(c)
        self.fine = nn.Conv1d(channels, dim, kernel_size=10, stride=10)
        self.fine_position = SinusoidalPositionEncoding(100, dim)
        self.fine_type = nn.Parameter(torch.zeros(1, 1, dim))
        if c.view != 'raw':
            self.coarse = nn.Conv1d(channels, dim, kernel_size=50, stride=50)
            self.coarse_position = SinusoidalPositionEncoding(20, dim)
            self.coarse_type = nn.Parameter(torch.zeros(1, 1, dim))
        if c.view == 'shape_fft':
            self.fft_projection = nn.Sequential(nn.Linear(128, dim), nn.GELU(), nn.LayerNorm(dim))
            self.fft_type = nn.Parameter(torch.zeros(1, 1, dim))
        block_config = MetaformerConfig(embedding_dim=dim, attention_heads=4, head_dim=dim // 4,
                      encoder_layers=c.encoder_layers, mlp_dim=dim * 2, dropout=c.dropout)
        self.blocks = nn.ModuleList(EncoderBlock(block_config) for _ in range(c.encoder_layers))
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, 1)
        self.head = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(c.dropout),
                                  nn.Linear(dim * 2, 7), nn.Sigmoid())

    def fit_input_normalization(self, training_spectra):
        self.views.fit(training_spectra)

    def forward(self, spectrum):
        values, coefficients = self.views(spectrum)
        tokens = [self.fine_position(self.fine(values).transpose(1, 2)) + self.fine_type]
        if self.config.view != 'raw':
            tokens.append(self.coarse_position(self.coarse(values).transpose(1, 2)) + self.coarse_type)
        if self.config.view == 'shape_fft':
            tokens.append(self.fft_projection(coefficients)[:, None] + self.fft_type)
        tokens = torch.cat(tokens, dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = (self.pool(tokens).softmax(dim=1) * tokens).sum(dim=1)
        return self.head(pooled)
