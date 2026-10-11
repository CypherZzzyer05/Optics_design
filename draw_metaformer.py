"""Draw the implemented ResNet + inverse Metaformer, with runtime shape checks.

Run from PyCharm or the project directory:
    python draw_metaformer.py

The PNG, editable SVG, PDF, and shape report are written to docs/figures.
This script only inspects architecture; it does not train or change weights.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from metaformer import InverseMetaformer, MetaformerConfig


ROOT = Path(__file__).resolve().parent
COLORS = {
    "ink": "#17304C", "muted": "#5F7084", "line": "#60748B",
    "blue": "#E9F2FC", "green": "#E6F5EF", "purple": "#F0ECFA",
    "orange": "#FFF1DC", "gray": "#EEF1F5", "panel": "#FAFBFD",
    "border": "#C9D5E3", "accent": "#247969",
}


def architecture_report(config: MetaformerConfig) -> dict:
    """Measure the actual forward path, including the ten attention tensors."""
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = InverseMetaformer(config).eval()
    observed = {}
    hooks = []

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            observed[name] = list(value.shape)
        return hook

    modules = dict(model.named_modules())
    names = ["spectral_frontend.stem"]
    names += [f"spectral_frontend.blocks.{i}" for i in range(config.resnet_blocks)]
    names += ["spectral_frontend", "patch_embedding", "position_encoding"]
    names += [f"blocks.{i}" for i in range(config.encoder_layers)]
    names += ["final_norm", "regressor.0", "regressor.1", "regressor.2", "regressor.3"]
    for name in names:
        hooks.append(modules[name].register_forward_hook(capture(name)))
    batch = 8
    try:
        with torch.inference_mode():
            prediction, attention = model(torch.zeros(batch, config.spectrum_points), True)
    finally:
        for hook in hooks:
            hook.remove()

    c = config
    assert observed["spectral_frontend"] == [batch, c.resnet_channels, c.spectrum_points]
    assert observed["patch_embedding"] == [batch, c.embedding_dim, c.patch_count]
    assert observed["position_encoding"] == [batch, c.patch_count, c.embedding_dim]
    for i in range(c.encoder_layers):
        assert observed[f"blocks.{i}"] == [batch, c.patch_count, c.embedding_dim]
    assert prediction.shape == (batch, c.geometry_dim)
    assert len(attention) == c.encoder_layers
    assert all(a.shape == (batch, c.attention_heads, c.patch_count, c.patch_count) for a in attention)
    baseline = InverseMetaformer(replace(c, spectral_frontend="linear"))
    return {
        "model_config": asdict(c),
        "probe_mode": "eval / inference_mode; synthetic zeros for shape checks only",
        "probe_input_shape": [batch, c.spectrum_points],
        "module_output_shapes": observed,
        "mean_pool_shape": [batch, c.embedding_dim],
        "prediction_shape": list(prediction.shape),
        "attention_shapes": [list(a.shape) for a in attention],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "baseline_parameter_count": sum(p.numel() for p in baseline.parameters()),
    }


def draw(config: MetaformerConfig, training: dict, bounds: np.ndarray,
         wavelengths: np.ndarray, report: dict, output: Path) -> None:
    c = config
    matplotlib.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(26, 17.5))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 260), ylim=(0, 175), aspect="equal")
    ax.axis("off")

    def label(x, y, text, size=11, weight="normal", color=None, ha="center", va="center"):
        return ax.text(x, y, text, fontsize=size, fontweight=weight,
                       color=color or COLORS["ink"], ha=ha, va=va, linespacing=1.45, zorder=5)

    def box(x, y, w, h, text, fill="blue", size=11, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                   facecolor=COLORS.get(fill, fill), edgecolor=COLORS["border"],
                                   linewidth=1.15, zorder=2))
        label(x + w / 2, y + h / 2, text, size, weight)

    def arrow(points, dashed=False, color=None):
        for i in range(len(points) - 1):
            last = i == len(points) - 2
            ax.add_patch(FancyArrowPatch(points[i], points[i + 1],
                                        arrowstyle="-|>" if last else "-",
                                        mutation_scale=12, linewidth=1.35,
                                        linestyle="--" if dashed else "-",
                                        color=color or COLORS["line"],
                                        shrinkA=0, shrinkB=0, zorder=3))

    def add(x, y):
        ax.add_patch(Circle((x, y), 2.25, facecolor="white", edgecolor=COLORS["accent"],
                            linewidth=1.5, zorder=4))
        label(x, y, "+", 17, "bold", COLORS["accent"])

    def panel(x, title):
        ax.add_patch(FancyBboxPatch((x, 25), 80, 97, boxstyle="round,pad=0.2,rounding_size=1.6",
                                   facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                                   linewidth=1.1, zorder=0))
        label(x + 40, 117, title, 14, "bold")

    label(130, 170, "ResNet + MetaFormer 逆向设计网络 · 实际实现结构", 23, "bold")
    rotated = c.geometry_dim == 6
    label(130, 164,
          f"光谱 → 五个尺寸{' + 共同旋转角' if rotated else ''}   |   B 为批量大小（训练 B={training['batch_size']}）"
          f"   |   参数量 {report['parameter_count']:,}   |   ResNet 为本项目增加的前端", 11,
          color=COLORS["muted"])

    start_um, stop_um = wavelengths[[0, -1]] / 1000
    stages = [
        ("blue", f"反射光谱 R(λ)\n{start_um:g}–{stop_um:g} μm；{c.spectrum_points} 点\n[B, {c.spectrum_points}]"),
        ("green", f"① 1D ResNet\nStem + {c.resnet_blocks} 个残差块\n[B, {c.resnet_channels}, {c.spectrum_points}]"),
        ("purple", f"Patch Embedding\nConv1d {c.resnet_channels}→{c.embedding_dim}\nk={c.patch_size}, stride={c.patch_size}；转置\n[B, {c.patch_count}, {c.embedding_dim}]"),
        ("purple", f"位置编码\n固定正弦 / 余弦\n与 token 逐元素相加\n[B, {c.patch_count}, {c.embedding_dim}]"),
        ("orange", f"② Encoder × {c.encoder_layers}\n{c.attention_heads} heads；dₖ={c.head_dim}\n每层含两条残差\n[B, {c.patch_count}, {c.embedding_dim}]"),
        ("blue", f"LayerNorm\n按每个 token 的\n{c.embedding_dim} 个特征归一化\n[B, {c.patch_count}, {c.embedding_dim}]"),
        ("blue", f"Mean Pool\n对 {c.patch_count} 个 token\n沿序列轴取平均\n[B, {c.embedding_dim}]"),
        ("green", f"尺寸回归头\nLinear {c.embedding_dim}→{c.mlp_dim}\nGELU → Linear {c.mlp_dim}→{c.geometry_dim}\nSigmoid；[B, {c.geometry_dim}]"),
        ("gray", f"单位还原（网络外）\np, l, h, αₓ, αᵧ{'，θ' if rotated else ''}\n[B, {c.geometry_dim}]；{'nm ×5 + deg' if rotated else 'nm'}\ng = lower + ĝ₀₁ × range"),
    ]
    for i, (fill, text) in enumerate(stages):
        x = 4 + i * 28
        box(x, 137, 24, 22, text, fill, 10.5)
        if i < len(stages) - 1:
            arrow([(x + 24.4, 148), (x + 27.6, 148)])

    arrow([(16, 137), (16, 132.5), (72, 132.5), (72, 137)], True)
    label(44, 128, f"基线版本跳过 ResNet；Patch Conv 改为 1→{c.embedding_dim}", 9,
          color=COLORS["muted"])
    names = ["p", "l", "h", "αₓ", "αᵧ"] + (["θ"] if rotated else [])
    ranges = "；".join(f"{name} {lo:g}–{hi:g}" for name, (lo, hi) in zip(names, bounds))
    label(169, 128, f"范围（前五个 nm{'；θ 用 deg' if rotated else ''}）：{ranges}", 9 if rotated else 9.5, color=COLORS["muted"])

    panel(4, "① 1D ResNet：局部光谱特征与残差")
    panel(90, f"② 单个 Encoder：重复 {c.encoder_layers} 次，权重独立")
    panel(176, "③ 多头自注意力：每个 Encoder 内部")

    # The CNN block has no activation after the residual addition.
    label(44, 108,
          f"Stem：Conv1d 1→{c.resnet_channels}，k={c.resnet_kernel_size}，stride=1 → GELU", 10.5)
    label(44, 103,
          "残差块串联：" + " → ".join(f"d={2 ** i}" for i in range(c.resnet_blocks)), 10.5)
    label(47, 96, f"x：[B, {c.resnet_channels}, {c.spectrum_points}]", 11)
    padding = (c.resnet_kernel_size - 1) // 2
    box(23, 80, 48, 11,
        f"Conv1d {c.resnet_channels}→{c.resnet_channels}\nk={c.resnet_kernel_size}，dilation=d，padding={padding}d", "green", 10.5)
    arrow([(47, 94), (47, 91.4)])
    box(23, 64, 48, 11, f"GroupNorm（{gcd(c.resnet_channels, 8)} 组）\nGELU", "green", 11)
    arrow([(47, 79.6), (47, 75.4)])
    box(23, 48, 48, 11,
        f"Conv1d {c.resnet_channels}→{c.resnet_channels}（k={c.resnet_kernel_size}，d）\nGroupNorm（{gcd(c.resnet_channels, 8)} 组）", "green", 10.5)
    arrow([(47, 63.6), (47, 59.4)])
    arrow([(47, 47.6), (47, 41.5)])
    arrow([(47, 94), (14, 94), (14, 39), (44.5, 39)], color=COLORS["accent"])
    label(12, 67, "恒等\n分支", 10, color=COLORS["accent"])
    add(47, 39)
    arrow([(47, 36.5), (47, 33.8)])
    label(47, 32, f"输出仍为 [B, {c.resnet_channels}, {c.spectrum_points}]；加和后无激活", 9.5)
    label(44, 27.5, "末端 GroupNorm 的 γ、β 仅在初始化时置零", 9.5, color=COLORS["muted"])

    # Pre-norm attention and feed-forward sublayers with two separate skip paths.
    label(130, 110, f"x：[B, {c.patch_count}, {c.embedding_dim}]", 11)
    box(107, 97, 46, 8, f"LayerNorm（{c.embedding_dim}）", "blue", 11)
    arrow([(130, 108), (130, 105.4)])
    box(107, 84, 46, 9, f"③ Multi-Head Self-Attention\n{c.attention_heads} heads × {c.head_dim} 维 → {c.embedding_dim} 维", "orange", 10.5)
    arrow([(130, 96.6), (130, 93.4)])
    arrow([(130, 83.6), (130, 80.5)])
    arrow([(130, 108), (99, 108), (99, 78), (127.5, 78)], color=COLORS["accent"])
    add(130, 78)
    label(151, 81, "x + Attention(LN(x))", 9, ha="center", color=COLORS["accent"])
    box(107, 63, 46, 8, f"LayerNorm（{c.embedding_dim}）", "blue", 11)
    arrow([(130, 75.5), (130, 71.4)])
    box(107, 44, 46, 14,
        f"MLP：Linear {c.embedding_dim}→{c.mlp_dim}\nGELU → Dropout({c.dropout:g})\nLinear {c.mlp_dim}→{c.embedding_dim} → Dropout({c.dropout:g})", "purple", 10.5)
    arrow([(130, 62.6), (130, 58.4)])
    arrow([(130, 43.6), (130, 40.5)])
    arrow([(132.5, 78), (162, 78), (162, 38), (132.5, 38)], color=COLORS["accent"])
    add(130, 38)
    arrow([(130, 35.5), (130, 32.2)])
    label(130, 30, f"输出：[B, {c.patch_count}, {c.embedding_dim}]", 10.5)
    label(130, 26.5, "两个子层均为 Pre-LayerNorm + 残差加和", 9.5, color=COLORS["muted"])

    # Expanded Q/K/V path. Attention dropout acts on the weights before A @ V.
    inner = c.attention_heads * c.head_dim
    label(216, 109, f"输入：LN(x)，[B, {c.patch_count}, {c.embedding_dim}]", 11)
    box(184, 94, 64, 11,
        f"Linear {c.embedding_dim}→{3 * inner}；拆为 Q、K、V\n每个 reshape 为 [B, {c.attention_heads}, {c.patch_count}, {c.head_dim}]", "orange", 10.5)
    arrow([(216, 107), (216, 105.4)])
    for x, name in [(184, "Q"), (208, "K"), (232, "V")]:
        box(x, 81, 16, 8, name, "orange", 12, "bold")
        arrow([(216, 93.6), (216, 91.5), (x + 8, 91.5), (x + 8, 89.4)])
    box(184, 67, 44, 9,
        f"QKᵀ / √{c.head_dim}\n[B, {c.attention_heads}, {c.patch_count}, {c.patch_count}]", "orange", 10.5)
    arrow([(192, 80.6), (192, 76.4)])
    arrow([(216, 80.6), (216, 76.4)])
    box(184, 54, 44, 9,
        f"沿 key 轴 Softmax\nA = Dropout({c.dropout:g})(weights)", "orange", 10.5)
    arrow([(206, 66.6), (206, 63.4)])
    box(184, 41, 64, 9,
        f"A × V → 拼接 {c.attention_heads} 个头\n[B, {c.patch_count}, {inner}]", "orange", 10.5)
    arrow([(206, 53.6), (206, 50.4)])
    arrow([(240, 80.6), (251, 80.6), (251, 45.5), (248.4, 45.5)])
    label(250, 61, "V", 10, color=COLORS["muted"])
    box(184, 28, 64, 9,
        f"Linear {inner}→{c.embedding_dim} → Dropout({c.dropout:g})\n输出：[B, {c.patch_count}, {c.embedding_dim}]", "orange", 10.5)
    arrow([(216, 40.6), (216, 37.4)])

    # The loss uses normalized geometry, not nanometer predictions.
    ax.add_patch(FancyBboxPatch((4, 3), 252, 17, boxstyle="round,pad=0.2,rounding_size=1.3",
                               facecolor=COLORS["gray"], edgecolor=COLORS["border"], linewidth=1))
    label(18, 11.5, "训练支路\n（网络外）", 11, "bold")
    box(34, 6, 55, 11, f"预测 ĝ₀₁ + 标签 g₀₁\n两者均为 [B, {c.geometry_dim}]，范围 0–1", "blue", 10.5)
    arrow([(89.4, 11.5), (97.6, 11.5)])
    box(98, 6, 43, 11, "MSE + 镜像等效角\n角度项比较 |θ| / 45°" if rotated else "MSE Loss\n在归一化尺寸上计算", "purple", 10.5)
    arrow([(141.4, 11.5), (149.6, 11.5)])
    box(150, 6, 53, 11, f"反向传播 + AdamW\n学习率 {training['learning_rate']:g}；更新权重", "green", 10.5)
    splits = report['split_sizes']
    label(230, 11.5, f"当前数据 {sum(splits.values())} 个\n训练 / 验证 / 测试\n{splits['train']} / {splits['val']} / {splits['test']}", 10)

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "metaformer_architecture"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=180, facecolor="white")
    # A second, enlarged view of the internals is convenient in a document or slide.
    extent = matplotlib.transforms.Bbox.from_extents(0.4, 2.4, 25.6, 12.3)
    fig.savefig(output / "metaformer_blocks.png", dpi=220, bbox_inches=extent, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_resnet_100.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    config_path = args.config.resolve()
    training = json.loads(config_path.read_text(encoding="utf-8"))
    config = MetaformerConfig(**training["model"])
    if config.spectral_frontend != "resnet1d" or config.geometry_dim not in (5, 6):
        parser.error("This figure describes five- or six-output ResNet + Metaformer configurations.")
    dataset_path = (config_path.parent / training["dataset"]).resolve()
    with np.load(dataset_path, allow_pickle=False) as data:
        rotated = config.geometry_dim == 6
        bounds = np.stack([data["lower_values" if rotated else "lower_nm"],
                           data["upper_values" if rotated else "upper_nm"]], axis=1)
        wavelengths = data["wavelength_nm"].copy()
        split_sizes = {split: int(len(data[f"{split}_indices"])) for split in ("train", "val", "test")}
    report = architecture_report(config)
    report.update({
        "source": "metaformer.py",
        "source_sha256": sha256((ROOT / "metaformer.py").read_bytes()).hexdigest(),
        "training_config": str(config_path), "dataset": str(dataset_path),
        "split_sizes": split_sizes, "parameter_bounds_nm": bounds[:5].tolist(),
        "parameter_bounds_values": bounds.tolist(), "parameter_units": ['nm'] * 5 + (['deg'] if rotated else []),
        "wavelength_range_nm": [float(wavelengths[0]), float(wavelengths[-1])],
        "input_scaling": "raw reflectance on the dataset wavelength grid; no per-curve min-max scaling",
        "loss": "MSE on normalized lengths and mirror-equivalent |theta|/45" if rotated else "MSE on normalized geometry",
    })
    draw(config, training, bounds, wavelengths, report, args.output_dir)
    (args.output_dir / "metaformer_architecture.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {config.encoder_layers} encoder shapes and attention tensors.")
    print(f"Parameters: ResNet + Metaformer={report['parameter_count']:,}; "
          f"baseline={report['baseline_parameter_count']:,}")
    print(f"Saved figures and architecture report to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
