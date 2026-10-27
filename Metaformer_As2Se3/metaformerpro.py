"""MetaformerPro: inverse spectral transformer with square-angle symmetry."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from metaformer import EncoderBlock, MetaformerConfig, SinusoidalPositionEncoding


@dataclass(frozen=True)
class MetaformerProConfig:
    readout: str = 'global'
    angle_head: str = 'sigmoid'
    overlap: bool = False
    embedding_dim: int = 96
    encoder_layers: int = 4
    dropout: float = .1
    normalization_floor: float = .02
    sampling: str = 'uniform'

    def __post_init__(self):
        if self.readout not in ('global', 'queries', 'flatten') or self.angle_head not in ('sigmoid', 'circular'):
            raise ValueError('Unknown MetaformerPro readout or angle head')
        if self.normalization_floor <= 0:
            raise ValueError('Normalization floor must be positive')
        if self.sampling not in ('uniform', 'uniform_extra', 'resonance'):
            raise ValueError('Unknown spectral sampling policy')
        if self.sampling != 'uniform' and (self.overlap or self.readout != 'global'):
            raise ValueError('Extra spectral samples use nonoverlapping patches and global readout')


def decode_circular(raw):
    """Decode five length logits and two (cos(phi), sin(phi)) pairs.

    Phase is 2*pi*u, where u=(theta+45)/90. The fixed pi phase offset
    relative to 4theta is included in the training targets implicitly.
    """
    raw = raw.float()
    pairs = raw[..., 5:].reshape(*raw.shape[:-1], 2, 2)
    angles = torch.remainder(torch.atan2(pairs[..., 1], pairs[..., 0]) / (2 * math.pi), 1.)
    return torch.cat((raw[..., :5].sigmoid(), angles), dim=-1)


class ReflectanceNormalizer(nn.Module):
    def __init__(self, floor=.02):
        super().__init__()
        self.floor = floor
        self.register_buffer('mean', torch.zeros(1, 1, 1000))
        self.register_buffer('scale', torch.ones(1, 1, 1000))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, spectra):
        spectra = spectra.float()[:, None]
        self.mean.copy_(spectra.mean(0, keepdim=True))
        self.scale.copy_(spectra.std(0, unbiased=False, keepdim=True).clamp_min(self.floor))
        self.fitted.fill_(True)

    def forward(self, spectra):
        if spectra.ndim != 2 or spectra.shape[-1] != 1000:
            raise ValueError('Expected raw reflectance [batch,1000]')
        if not self.fitted:
            raise RuntimeError('Fit normalization on training spectra first')
        return (spectra.float()[:, None] - self.mean) / self.scale


class MetaformerPro(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MetaformerProConfig()
        c, dim = self.config, self.config.embedding_dim
        self.normalizer = ReflectanceNormalizer(c.normalization_floor)
        self.tokenizer = nn.Conv1d(1, dim, kernel_size=20 if c.overlap else 10, stride=10)
        self.positions = SinusoidalPositionEncoding(100, dim)
        self.token_type = nn.Parameter(torch.zeros(1, 1, dim))
        if c.sampling != 'uniform':
            marker = torch.zeros(1, 1, dim)
            marker[..., 0] = 1.
            self.register_buffer('extra_token_marker', marker, persistent=False)
        block_config = MetaformerConfig(embedding_dim=dim, attention_heads=4, head_dim=dim // 4,
                         encoder_layers=c.encoder_layers, mlp_dim=dim * 2, dropout=c.dropout)
        self.blocks = nn.ModuleList(EncoderBlock(block_config) for _ in range(c.encoder_layers))
        self.norm = nn.LayerNorm(dim)
        if c.readout in ('global', 'flatten'):
            if c.readout == 'global':
                self.pool = nn.Linear(dim, 1)
            head_input = dim if c.readout == 'global' else 100 * dim
            self.head = nn.Sequential(nn.Linear(head_input, dim * 2), nn.GELU(), nn.Dropout(c.dropout),
                                     nn.Linear(dim * 2, 9 if c.angle_head == 'circular' else 7))
            if c.angle_head == 'sigmoid':
                self.head.append(nn.Sigmoid())
            else:
                with torch.no_grad():
                    self.head[-1].bias[5:].copy_(torch.tensor([-1., 0., -1., 0.]))
        else:
            self.queries = nn.Parameter(torch.randn(7, dim) * .02)
            self.query_trunk = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(c.dropout))
            self.length_head = nn.Linear(dim * 2, 1)
            self.angle_projection = nn.Linear(dim * 2, 2 if c.angle_head == 'circular' else 1)
            if c.angle_head == 'circular':
                with torch.no_grad():
                    self.angle_projection.bias.copy_(torch.tensor([-1., 0.]))

    def fit_input_normalization(self, training_spectra):
        self.normalizer.fit(training_spectra)

    def forward(self, spectrum):
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            normalized = self.normalizer(spectrum)
        if self.config.overlap:
            normalized = F.pad(normalized, (5, 5), mode='replicate')
        tokens = self.positions(self.tokenizer(normalized).transpose(1, 2)) + self.token_type
        if self.config.sampling != 'uniform':
            from resonance_sampling import (build_extra_sampling_grid, linear_sample,
                                             continuous_position_embedding)
            grid = build_extra_sampling_grid(spectrum, self.config.sampling)
            with torch.autocast(device_type=spectrum.device.type, enabled=False):
                patches = linear_sample(normalized, grid)
                coordinates = continuous_position_embedding(
                    grid.mean(-1), grid[..., -1] - grid[..., 0], self.config.embedding_dim)
            extra = self.tokenizer(patches.reshape(len(spectrum), 1, -1)).transpose(1, 2)
            extra = extra + coordinates.to(extra.dtype) + self.token_type + self.extra_token_marker
            tokens = torch.cat((tokens, extra), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        if self.config.readout == 'global':
            pooled = (self.pool(tokens).softmax(1) * tokens).sum(1)
            raw = self.head(pooled)
            return decode_circular(raw) if self.config.angle_head == 'circular' else raw
        if self.config.readout == 'flatten':
            raw = self.head(tokens.flatten(1))
            return decode_circular(raw) if self.config.angle_head == 'circular' else raw
        weights = torch.einsum('bnd,pd->bpn', tokens, self.queries) / math.sqrt(tokens.shape[-1])
        pooled = torch.einsum('bpn,bnd->bpd', weights.softmax(-1), tokens)
        hidden = self.query_trunk(pooled)
        lengths = self.length_head(hidden[:, :5]).squeeze(-1)
        angles = self.angle_projection(hidden[:, 5:]).flatten(1)
        raw = torch.cat((lengths, angles), dim=-1)
        return decode_circular(raw) if self.config.angle_head == 'circular' else raw.sigmoid()


def import_v01_state(legacy_state):
    """One-time exact conversion of yesterday's compact_raw_mirror weights."""
    result = {}
    for name, value in legacy_state.items():
        if name in ('views.mean', 'views.scale', 'views.fitted'):
            result[name.replace('views.', 'normalizer.')] = value
        elif name.startswith(('blocks.', 'norm.', 'pool.', 'head.')):
            result[name] = value
        elif name.startswith('fine.'):
            result[name.replace('fine.', 'tokenizer.')] = value
        elif name == 'fine_type':
            result['token_type'] = value
    return result
