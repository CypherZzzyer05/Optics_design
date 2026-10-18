"""Read-only preprocessing illustration for an existing FDTD spectrum.

Run with --dataset /path/to/dataset.npz. No model or solver is invoked.
The background scale is an illustrative choice, not a validated training setting.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import grey_closing
from scipy.signal import find_peaks, peak_widths, savgol_filter


BLUE, ORANGE, INK = "#2465A5", "#DB7926", "#23364A"
COLORS = [BLUE, ORANGE]
SOURCES = {
    "background": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_closing.html",
    "width": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html",
    "smoothing": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.facecolor": "white",
    })


def transform(x, raw, background_points):
    if background_points < 3 or background_points % 2 != 1 or background_points > len(x):
        raise ValueError("Background window must be odd and between 3 and the spectrum length")
    indices, props = find_peaks(-raw, prominence=.01)
    if len(indices) < 2:
        raise ValueError("This illustration requires two detectable valleys; none are fabricated")
    chosen = sorted(np.argsort(props["prominences"])[-2:], key=lambda i: indices[i])
    background = grey_closing(raw, size=background_points, mode="nearest")
    contrast = background - raw
    # Closing applies two neighborhood operations. Exclude padding-dependent
    # boundary features rather than presenting them as measured resonances.
    valid_background = np.ones(len(x), dtype=bool)
    margin = background_points - 1
    valid_background[:margin] = False
    valid_background[-margin:] = False
    derivative = np.gradient(raw, x, edge_order=2)
    u = np.linspace(-3, 3, 33)
    valleys, windows, window_wavelengths = [], [], []
    smoothing = {n: savgol_filter(raw, n, 2, mode="interp") for n in (5, 7, 11)}
    smoothing_differences = []
    for order, j in enumerate(chosen):
        idx = int(indices[j])
        widths, heights, left, right = peak_widths(-raw, [idx], rel_height=.5)
        left_nm, right_nm = np.interp([left[0], right[0]], np.arange(len(x)), x)
        width = float(right_nm - left_nm)
        depth = float(contrast[idx])
        if depth <= 0 or width <= 0:
            raise ValueError("The detected valley has no usable contrast or width")
        target_wavelength = x[idx] + u * width
        if target_wavelength[0] < x[0] or target_wavelength[-1] > x[-1]:
            raise ValueError("Valley window is truncated by the measured band; a validity mask is needed")
        windows.append(np.interp(target_wavelength, x, contrast) / depth)
        window_wavelengths.append(target_wavelength)
        mask = (x >= target_wavelength[0]) & (x <= target_wavelength[-1])
        if not np.all(valid_background[mask]):
            raise ValueError("Local background depends on edge padding; a reliable local window is unavailable")
        half_mask = (x >= left_nm) & (x <= right_nm)
        record = {
            "order": order + 1,
            "sample_index": idx,
            "center_nm": float(x[idx]),
            "minimum_reflectance": float(raw[idx]),
            "raw_prominence": float(props["prominences"][j]),
            "half_prominence_width_nm": width,
            "half_prominence_level_R": float(-heights[0]),
            "half_prominence_bounds_nm": [float(left_nm), float(right_nm)],
            "native_samples_inside_half_prominence_width": int(half_mask.sum()),
            "estimated_background_R": float(background[idx]),
            "background_contrast_depth": depth,
            "native_samples_inside_local_window": int(mask.sum()),
        }
        valleys.append(record)
        near = (x >= x[idx] - 2 * width) & (x <= x[idx] + 2 * width)
        native_indices = np.flatnonzero(near)
        for n, smooth in smoothing.items():
            k = int(native_indices[np.argmin(smooth[near])])
            smoothing_differences.append({
                "valley_order": order + 1, "window_points": n, "polynomial_order": 2,
                "smoothed_local_minimum_nm": float(x[k]),
                "smoothed_local_minimum_R": float(smooth[k]),
                "minimum_R_change_percentage_points": float((smooth[k] - raw[idx]) * 100),
            })
    arrays = {
        "wavelength_nm": x, "raw_R": raw, "estimated_background_R": background,
        "contrast_R": contrast, "dR_dlambda_per_nm": derivative,
        "valid_background_mask": valid_background,
        "local_u": u, "local_wavelength_nm": np.array(window_wavelengths),
        "local_contrast_divided_by_depth": np.array(windows),
        "local_metadata": np.array([[v["center_nm"], v["half_prominence_width_nm"],
                                     v["minimum_reflectance"], v["background_contrast_depth"]]
                                    for v in valleys]),
        **{f"savgol_{n}_R": y for n, y in smoothing.items()},
    }
    assert np.min(contrast) >= -1e-12
    assert np.max(np.abs(background - contrast - raw)) < 1e-12
    return arrays, valleys, smoothing_differences


def plot_views(a, valleys, sample_id, path, background_points):
    x, raw = a["wavelength_nm"], a["raw_R"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    fig.subplots_adjust(left=.075, right=.97, bottom=.13, top=.85, hspace=.47, wspace=.23)
    fig.suptitle(f"同一条真实光谱的四种表达 · 1.0 样本 {sample_id}", fontsize=19, y=.972, fontweight="bold")
    fig.text(.075, .911, "700–1700 nm / 1000 点 / 原始反射率保留；其余表达作为候选辅助信息", fontsize=11, color="#5C6C7C")

    ax = axes[0, 0]
    ax.plot(x, raw, lw=1.55, color=BLUE, label="原始 R")
    ax.plot(x, np.where(a["valid_background_mask"], a["estimated_background_R"], np.nan),
            lw=1.15, ls="--", color="#7A8795", label="估计局部背景（有效区域）")
    for i, v in enumerate(valleys):
        ax.scatter(v["center_nm"], v["minimum_reflectance"], s=35, color=COLORS[i], zorder=4)
        ax.annotate(f"谷 {i+1}: {v['center_nm']:.2f} nm\nR={v['minimum_reflectance']:.3f}",
                    (v["center_nm"], v["minimum_reflectance"]),
                    xytext=(v["center_nm"]+55, .24 if i == 0 else .49), fontsize=10,
                    arrowprops={"arrowstyle": "->", "color": COLORS[i]})
    ax.set(title="A  原始光谱：保留位置、深度和背景", ylabel="反射率 R", ylim=(0, 1.04))
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    ax = axes[0, 1]
    ax.plot(x, np.where(a["valid_background_mask"], a["contrast_R"], np.nan), lw=1.45, color=ORANGE)
    valid_indices = np.flatnonzero(a["valid_background_mask"])
    ax.axvspan(x[0], x[valid_indices[0]], color="#D9DFE6", alpha=.5)
    ax.axvspan(x[valid_indices[-1]], x[-1], color="#D9DFE6", alpha=.5)
    ax.set(title="B  去背景：把局部反射谷表示为向上的特征", ylabel="背景 − R（反射率差）")
    ax.text(.97, .92, f"背景用 {background_points} 点闭运算估计\n两端灰区不参与背景特征；弱谷仍保留", transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color="#5C6C7C")

    ax = axes[1, 0]
    ax.plot(x, a["dR_dlambda_per_nm"], lw=1.25, color="#308B74")
    ax.axhline(0, color="#ABB5C0", lw=.7)
    ax.set(title="C  一阶导数：突出谷的下降沿和上升沿", ylabel="dR/dλ（每 nm）")
    ax.text(.98, .92, "直接对原始数据求差分\n没有预先平滑；导数可能放大噪声", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#5C6C7C")

    ax = axes[1, 1]
    for i, v in enumerate(valleys):
        idx = v["sample_index"]
        u_native = (x - x[idx]) / v["half_prominence_width_nm"]
        mask = np.abs(u_native) <= 3
        ax.plot(u_native[mask], a["contrast_R"][mask] / v["background_contrast_depth"],
                "o", ms=3.7, color=COLORS[i], alpha=.7)
        ax.plot(a["local_u"], a["local_contrast_divided_by_depth"][i], lw=1.5,
                color=COLORS[i], label=f"谷 {i+1}: W={v['half_prominence_width_nm']:.2f} nm")
    ax.set(title="D  谷形对齐：比较形状，同时另存位置、宽度和深度",
           xlabel="相对坐标 u = (λ − 谷中心) / W", ylabel="局部对比 / 该谷深度", xlim=(-3, 3))
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.16)
        if ax is not axes[1, 1]:
            ax.set(xlim=(x[0], x[-1]), xlabel="波长 λ（nm）")
    fig.text(.075, .061, "W 为半突出度宽度，不直接当作物理 Q 值。D 图：圆点是原始采样；实线为 33 点线性插值，不增加真实分辨率。", fontsize=10, color="#5C6C7C")
    fig.text(.075, .031, f"{background_points} 点背景尺度仅作演示；局部归一化后，中心波长、W、谷底 R 和深度均作为独立信息保留。", fontsize=10, color="#5C6C7C")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_smoothing(a, valleys, path):
    x = a["wavelength_nm"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))
    fig.subplots_adjust(left=.075, right=.97, bottom=.19, top=.78, wspace=.2)
    fig.suptitle("预处理前先检查：平滑会不会改变我们要学习的窄谷？", fontsize=18, y=.95, fontweight="bold")
    for ax, v in zip(axes, valleys):
        mask = np.abs(x-v["center_nm"]) <= 12
        ax.plot(x[mask], a["raw_R"][mask], "o-", color=BLUE, ms=4, lw=1.7, label="原始采样")
        for n, color in ((5, "#36A286"), (7, ORANGE), (11, "#C85865")):
            ax.plot(x[mask], a[f"savgol_{n}_R"][mask], lw=1.4, color=color, label=f"SG {n} 点 / 二阶")
        ax.set(title=f"谷 {v['order']} · {v['center_nm']:.2f} nm · W={v['half_prominence_width_nm']:.2f} nm",
               xlabel="波长 λ（nm）", ylabel="反射率 R")
        ax.grid(alpha=.16)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(.075, .06, "同一条原始光谱，不把平滑曲线作为训练真值；平滑窗口需相对于最窄特征检查，不能直接沿用时序信号的窗口。", fontsize=10, color="#5C6C7C")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--background-points", type=int, default=41)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output/preprocessing_demo")
    args = parser.parse_args()
    source = args.dataset.resolve()
    before = file_hash(source)
    with np.load(source, allow_pickle=False) as d:
        rows = np.flatnonzero(d["sample_ids"] == args.sample_id)
        if len(rows) != 1:
            raise ValueError("Sample ID must identify exactly one existing row")
        row = int(rows[0])
        x = d["wavelength_nm"].astype(float)
        raw = d["spectra"][row].astype(float)
        status = int(d["status"][row])
        geometry = d["geometry_values"][row].astype(float)
    if status != 2 or raw.shape != x.shape or not np.isfinite(raw).all():
        raise ValueError("Need a successful finite spectrum on the wavelength grid")
    if not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
        raise ValueError("Wavelength grid must be finite and increasing")
    if not np.allclose(np.diff(x), np.diff(x).mean(), rtol=1e-8):
        raise ValueError("The point-based background example requires a uniform grid")
    arrays, valleys, changes = transform(x, raw, args.background_points)
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    stem = f"sample{args.sample_id}"
    plot_views(arrays, valleys, args.sample_id, args.output / f"{stem}_preprocessing.png", args.background_points)
    plot_smoothing(arrays, valleys, args.output / f"{stem}_smoothing.png")
    assert file_hash(source) == before
    report = {
        "dataset": str(source), "dataset_sha256": before, "sample_id": args.sample_id,
        "geometry_values": geometry.tolist(), "status": status,
        "source_unchanged": True, "solver_run_performed": False, "model_training_performed": False,
        "raw_grid_step_nm": float(np.diff(x).mean()),
        "background_method": "grey_closing; nearest boundary padding; flat 1D footprint",
        "background_window_points": args.background_points,
        "background_window_span_nm": float((args.background_points-1)*np.diff(x).mean()),
        "background_boundary_mask": f"Exclude first and last {args.background_points-1} points from derived background features; raw spectrum retained",
        "derivative": "numpy.gradient(R, wavelength_nm, edge_order=2); no smoothing",
        "local_view": "33 linear-interpolation samples over +-3 half-prominence widths; no added resolution",
        "local_metadata_columns": ["center_nm", "half_prominence_width_nm", "minimum_R", "contrast_depth"],
        "valleys": valleys, "smoothing_differences": changes,
        "reconstruction_max_absolute_error": float(np.max(np.abs(arrays["estimated_background_R"]-arrays["contrast_R"]-raw))),
        "illustration_only": "Background scale and valley rules are not validated training hyperparameters; raw spectrum is retained",
        "scipy_version": scipy.__version__, "sources": SOURCES,
        "script_sha256": file_hash(Path(__file__)),
    }
    np.savez_compressed(args.output / f"{stem}_preprocessing.npz", **arrays)
    (args.output / f"{stem}_preprocessing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"sample_id": args.sample_id, "valleys": valleys, "output": str(args.output.resolve()), "source_unchanged": True}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
