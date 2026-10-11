"""Inverse Metaformer: reflectance spectrum -> lengths and optional common rotation.

The paper specifies 1000 spectral samples, 100 patches, a 128-dimensional
embedding, 10 attention heads, and d_k=64. Since 128 is not divisible by 10,
attention uses explicit 128 -> (10 * 64) projections instead of PyTorch's
``nn.MultiheadAttention``.
"""

# 核心流程：反射光谱 → 可选 1D ResNet → 光谱分块/token → 位置编码
#          → 多层注意力 Encoder → 平均池化 → 五个归一化长度及可选旋转角。
# 阅读时约定：B=批量大小，P=token 数量，C=卷积通道数，D=特征维度。
# 本文件仅定义网络；数据读取、训练更新和 nm 单位还原位于另外两个模块。

from dataclasses import dataclass
from math import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MetaformerConfig:
    spectrum_points: int = 1000   # 一条光谱含 1000 个按波长升序排列的反射率值。
    patch_count: int = 100        # 每条光谱形成 100 个 token，每个窗口含 10 个采样位置。
    embedding_dim: int = 128      # 每个 token 的学习特征维度；各维没有预设物理含义。
    attention_heads: int = 10     # 每个头都读取全部 token，通过不同投影学习关联。
    head_dim: int = 64            # 每头 Q/K/V 各 64 维；先投影到 640 维再拆成 10 头。
    encoder_layers: int = 10      # 编码层串联，层间结构相同但参数各自独立。
    mlp_dim: int = 256            # Encoder 内 MLP 和最终回归头的隐藏维度。
    dropout: float = 0.1          # 训练时随机丢弃部分元素；eval() 时关闭。
    geometry_dim: int = 5         # 默认五个长度；旋转任务设为 6，末列追加 rotation_deg。
    spectral_frontend: str = "linear"  # linear 保持原模型；resnet1d 启用新增卷积前端。
    resnet_channels: int = 32     # ResNet 保留 1000 个位置，每个位置有 32 通道特征。
    resnet_blocks: int = 3        # 三个残差块的 dilation 分别为 1、2、4。
    resnet_kernel_size: int = 7   # 奇数卷积核便于对称补零并保持序列长度。
    angle_symmetry: str = "none"  # square_mirror：六参数角度任务将 +/-theta 视为等效。

    def __post_init__(self) -> None:
        # 要求点数能被 token 数整除，防止分块时漏掉末尾的光谱点。
        if self.spectrum_points <= 0 or self.patch_count <= 0:
            raise ValueError("Spectrum and patch sizes must be positive")
        if self.spectrum_points % self.patch_count:
            raise ValueError("spectrum_points must be divisible by patch_count")
        if min(self.embedding_dim, self.attention_heads, self.head_dim,
               self.encoder_layers, self.mlp_dim, self.geometry_dim) <= 0:
            raise ValueError("All network dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.spectral_frontend not in ("linear", "resnet1d"):
            raise ValueError("spectral_frontend must be linear or resnet1d")
        for name in ("resnet_channels", "resnet_blocks", "resnet_kernel_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.resnet_kernel_size % 2 != 1:
            raise ValueError("resnet_kernel_size must be odd to preserve spectral length")
        if self.angle_symmetry not in ("none", "square_mirror"):
            raise ValueError("angle_symmetry must be none or square_mirror")
        if self.angle_symmetry != "none" and self.geometry_dim != 6:
            raise ValueError("Square mirror angle loss requires six outputs")

    @property
    def patch_size(self) -> int:
        # 当前配置下 1000 / 100 = 10；划分按固定位置进行，不按峰的位置进行。
        return self.spectrum_points // self.patch_count


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, length: int, dim: int) -> None:
        super().__init__()
        # 位置是 token 的序号；它与固定波长网格对应，不是直接输入波长数值。
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        frequencies = torch.exp(-even_indices * (torch.log(torch.tensor(10000.0)) / dim))
        angles = positions * frequencies
        encoding = torch.zeros(1, length, dim)
        # 偶数特征维使用 sin，奇数维使用 cos；不同维度使用不同频率。
        encoding[0, :, 0::2] = torch.sin(angles)
        encoding[0, :, 1::2] = torch.cos(angles[:, : dim // 2])
        # buffer 会随模型迁移到 CPU/GPU，但不是可训练参数；这里也不写入权重文件。
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        # [1,P,D] 广播到 [B,P,D]，为特征增加位置信息，token 数和维度保持不变。
        return tokens + self.encoding[:, : tokens.shape[1]].to(tokens.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        # 不把 128 维直接均分成 10 头；对 Q/K/V 分别投影成 10×64=640 维。
        # 合并为一次 Linear(128,1920)，计算上等价于三个独立的带偏置线性投影。
        self.qkv = nn.Linear(dim, 3 * inner_dim)
        # 拼接十个头后的 640 维特征再映射回 128 维，方便与输入做残差加和。
        self.output = nn.Linear(inner_dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, return_attention: bool = False):
        # 输入是 Encoder 中 LayerNorm 后的 token，当前形状 [B,100,128]。
        batch, patches, _ = tokens.shape
        # Linear 后为 [B,100,1920]，再整理为 [B,100,3,10,64]。
        qkv = self.qkv(tokens).reshape(batch, patches, 3, self.heads, self.head_dim)
        # 调整轴并拆出 Q/K/V，每个张量为 [B,10,100,64]。
        # Q 用来查询匹配特征，K 提供匹配线索，V 提供实际汇集的特征信息。
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # QK^T / sqrt(64)：每个头都比较全部 100×100 对 token，没有因果掩码。
        # 矩阵第 i 行、第 j 列表示 query token i 与 key token j 的匹配分数。
        weights = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # 对 key 轴做 Softmax；每个 query 对全部 key 的权重之和为 1。
        # 这些权重随输入重新计算；训练更新的是 qkv/output 的投影参数等网络参数。
        weights = weights.softmax(dim=-1)
        # 按权重汇集 V 特征，输出 [B,10,100,64]，加权对象是特征向量。
        # 训练时 Dropout 作用在权重上，其输出每行不必严格和为 1；推理时关闭。
        attended = self.attention_dropout(weights) @ value
        # 将 head 轴移到 token 后面，再拼接各头，形成 [B,100,640]。
        attended = attended.transpose(1, 2).reshape(batch, patches, -1)
        result = self.output_dropout(self.output(attended))
        # 返回的 weights 是 Dropout 前的 [B,10,100,100]，可用于画注意力图。
        # 权重大表示该层该头读取较多信息，不能直接当作物理重要性或因果证据。
        return (result, weights) if return_attention else result


class EncoderBlock(nn.Module):
    def __init__(self, config: MetaformerConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        # Pre-LayerNorm：先归一化特征，再计算子层；每个子层之后都有残差路径。
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(
            dim, config.attention_heads, config.head_dim, config.dropout
        )
        self.mlp_norm = nn.LayerNorm(dim)
        # 对各 token 独立使用同一个 MLP：128→256→128，GELU 提供非线性。
        # token 之间的信息交互发生在注意力子层，MLP 沿最后的特征轴处理。
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor, return_attention: bool = False):
        if return_attention:
            update, weights = self.attention(self.attention_norm(tokens), True)
        else:
            update = self.attention(self.attention_norm(tokens))
        # 第一条残差：保留原 token 表示，同时叠加跨波长的信息交互结果。
        tokens = tokens + update
        # 第二条残差：在已更新的 token 上加入 MLP 的非线性特征变换。
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return (tokens, weights) if return_attention else tokens


class SpectralResidualBlock(nn.Module):
    """Local peak-shape features with an identity path and length-preserving convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # dilation 扩大卷积采样间隔；配套 padding 使输出仍有 1000 个位置。
        padding = dilation * (kernel_size - 1) // 2
        # 32 通道时分 8 组；GroupNorm 对每个样本处理，不依赖批量统计量。
        groups = gcd(channels, 8)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        # Start as an identity branch; the residual contribution learns from the data.
        # 仅初始化时将末端归一化参数置零，使起始输出等于输入；之后仍可训练。
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, spectrum_features: Tensor) -> Tensor:
        # 残差支路：Conv → GroupNorm → GELU → Conv → GroupNorm。
        update = self.activation(self.norm1(self.conv1(spectrum_features)))
        # 恒等支路与残差支路逐元素相加；实现中加和后没有额外激活函数。
        return spectrum_features + self.norm2(self.conv2(update))


class SpectralResNet(nn.Module):
    """1D ResNet stem followed by dilated blocks; preserve the ordered 1000-point grid."""

    def __init__(self, config: MetaformerConfig):
        super().__init__()
        # 原始单通道反射率 [B,1,1000] → 峰形特征 [B,32,1000]。
        self.stem = nn.Sequential(
            nn.Conv1d(1, config.resnet_channels, config.resnet_kernel_size,
                      padding=config.resnet_kernel_size // 2),
            nn.GELU(),
        )
        # 串联三个权重独立的残差块，dilation=1/2/4，在分块前融合邻域特征。
        self.blocks = nn.Sequential(*[
            SpectralResidualBlock(config.resnet_channels, config.resnet_kernel_size, 2 ** i)
            for i in range(config.resnet_blocks)
        ])

    def forward(self, spectrum: Tensor) -> Tensor:
        return self.blocks(self.stem(spectrum))


class InverseMetaformer(nn.Module):
    """Map [batch, 1000] reflectance to [batch, geometry_dim] values in [0, 1].

    Geometry order: p, l, h, alpha_x, alpha_y, optionally rotation_deg.
    Convert normalized outputs to physical units using training-set parameter
    bounds. This module does not infer units or perform that conversion.
    """

    def __init__(self, config: MetaformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MetaformerConfig()
        c = self.config
        # linear 路径直接分块原光谱；resnet1d 路径先处理整条光谱的特征。
        self.spectral_frontend = SpectralResNet(c) if c.spectral_frontend == "resnet1d" else nn.Identity()
        # kernel=stride=10：1000 个位置形成 100 个不重叠的特征窗口。
        # 原路径每窗 1×10→128；ResNet 路径每窗 32×10→128，各窗口共用权重。
        # 卷积窗口不重叠，但 ResNet 已融合邻域信息，token 的信息范围可跨过窗口边界。
        self.patch_embedding = nn.Conv1d(
            c.resnet_channels if c.spectral_frontend == "resnet1d" else 1,
            c.embedding_dim, kernel_size=c.patch_size, stride=c.patch_size
        )
        self.position_encoding = SinusoidalPositionEncoding(c.patch_count, c.embedding_dim)
        # 每次构造一个新 Encoder 实例，十层不共享参数。
        self.blocks = nn.ModuleList(EncoderBlock(c) for _ in range(c.encoder_layers))
        self.final_norm = nn.LayerNorm(c.embedding_dim)
        # 回归五个长度及可选角度；Sigmoid 将每列输出限制在 0～1。
        # nm/deg 单位还原见 SpectrumGeometryDataset.denormalize()。
        self.regressor = nn.Sequential(
            nn.Linear(c.embedding_dim, c.mlp_dim),
            nn.GELU(),
            nn.Linear(c.mlp_dim, c.geometry_dim),
            nn.Sigmoid(),
        )

    def forward(self, spectrum: Tensor, return_attention: bool = False):
        # 同时接受 [B,1000] 和 [B,1,1000]，内部统一为前者。
        if spectrum.ndim == 3 and spectrum.shape[1] == 1:
            spectrum = spectrum[:, 0, :]
        if spectrum.ndim != 2 or spectrum.shape[1] != self.config.spectrum_points:
            raise ValueError(
                f"Expected spectrum [batch, {self.config.spectrum_points}] "
                f"or [batch, 1, {self.config.spectrum_points}]"
            )
        # 加入通道轴后提取特征：原路径 [B,1,1000]，ResNet 路径 [B,32,1000]。
        features = self.spectral_frontend(spectrum.unsqueeze(1))
        # Conv 输出 [B,128,100]；转置成注意力所用的 [B,100,128]。
        tokens = self.patch_embedding(features).transpose(1, 2)
        tokens = self.position_encoding(tokens)
        attention_maps = []
        # 每层仍有 100 个 token，但 token 的内容会结合整条光谱逐层更新。
        for block in self.blocks:
            if return_attention:
                tokens, weights = block(tokens, True)
                attention_maps.append(weights)
            else:
                tokens = block(tokens)
        # 先按特征轴 LayerNorm，再沿 token 轴平均：[B,100,128] → [B,128]。
        # 回归头输出 [B,geometry_dim]；这里尚未还原为 nm/deg。
        prediction = self.regressor(self.final_norm(tokens).mean(dim=1))
        return (prediction, attention_maps) if return_attention else prediction


def geometry_mse(prediction: Tensor, target: Tensor, angle_symmetry: str = "none") -> Tensor:
    """MSE for geometry values normalized to the same [0, 1] bounds."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    if angle_symmetry == "square_mirror":
        if prediction.shape[-1] != 6:
            raise ValueError("Mirror-equivalent rotation loss requires six values")
        # theta = -45 + 90*u；|2*u-1| 是 |theta|/45，处理镜像等效及 +/-45 端点。
        # 原始有符号标签仍保存；网络输出的是一个等效角候选，而非唯一可辨识的符号。
        prediction = torch.cat((prediction[..., :5], (2 * prediction[..., 5:] - 1).abs()), dim=-1)
        target = torch.cat((target[..., :5], (2 * target[..., 5:] - 1).abs()), dim=-1)
    elif angle_symmetry != "none":
        raise ValueError("Unknown angle symmetry")
    # 对批量和全部结构参数的平方误差取平均；预测与标签使用同一套上下界。
    # 该损失衡量尺寸回归误差，不能直接代表预测结构重新仿真后的光谱误差。
    return F.mse_loss(prediction, target)
