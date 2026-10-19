"""Forward Metaformer: seven normalized geometry values -> 1000-point R.

The inverse model is unchanged. Encoder blocks are reused, with independent
weights. Each wavelength token predicts one contiguous 10-point spectral patch.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from metaformer import EncoderBlock, MetaformerConfig, SinusoidalPositionEncoding


@dataclass(frozen=True)
class ForwardMetaformerConfig:
    spectrum_points: int = 1000
    patch_count: int = 100
    embedding_dim: int = 128
    attention_heads: int = 4
    head_dim: int = 64
    encoder_layers: int = 3
    mlp_dim: int = 256
    dropout: float = 0.1

    def encoder_config(self):
        return MetaformerConfig(**asdict(self), geometry_dim=7,
                                angle_symmetry='square_periodic')

    def __post_init__(self):
        self.encoder_config()  # Reuse the existing shape/dropout validation.


def encode_geometry(geometry):
    """Five normalized lengths plus sin/cos of each square's periodic angle.

    u = (theta + 45) / 90; phase = 4*theta in radians. The feature count is
    nine, while the physical degrees of freedom remain seven.
    """
    if geometry.ndim != 2 or geometry.shape[1] != 7:
        raise ValueError('Expected normalized geometry [batch,7]')
    phase = 2 * math.pi * (geometry[:, 5:] - .5)
    circular = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
    return torch.cat((geometry[:, :5], circular), dim=1)


class ForwardMetaformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ForwardMetaformerConfig()
        c = self.config
        self.geometry_embedding = nn.Linear(9, c.embedding_dim)
        self.position_encoding = SinusoidalPositionEncoding(c.patch_count, c.embedding_dim)
        self.blocks = nn.ModuleList(EncoderBlock(c.encoder_config())
                                    for _ in range(c.encoder_layers))
        self.spectrum_head = nn.Sequential(
            nn.LayerNorm(c.embedding_dim),
            nn.Linear(c.embedding_dim, c.spectrum_points // c.patch_count),
            nn.Sigmoid(),
        )

    def forward(self, geometry):
        features = self.geometry_embedding(encode_geometry(geometry))
        tokens = features[:, None, :].expand(-1, self.config.patch_count, -1)
        tokens = self.position_encoding(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.spectrum_head(tokens).flatten(1)
