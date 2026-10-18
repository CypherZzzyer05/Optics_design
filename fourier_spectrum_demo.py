"""Plot an existing spectrum, its reversible DFT, and the matching geometry.

This is a read-only data demonstration. It never launches an FDTD solver.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import find_peaks, peak_widths

from fdtd_pipeline.config import Geometry


BLUE = "#1768AC"
ORANGE = "#D16B28"
INK = "#23354A"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def prism(polygon, z_min, z_max, color):
    lower = np.column_stack([polygon, np.full(len(polygon), z_min)])
    upper = np.column_stack([polygon, np.full(len(polygon), z_max)])
    faces = [lower, upper]
    faces.extend(np.array([lower[i], lower[(i + 1) % len(polygon)],
                           upper[(i + 1) % len(polygon)], upper[i]])
                 for i in range(len(polygon)))
    return faces, [color] * len(faces)


def plot_structure(g, simulation, output, sample_id):
    fig = plt.figure(figsize=(13.6, 7.1))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.12], wspace=.13)
    ax = fig.add_subplot(grid[0])
    p = g.p
    ax.add_patch(Rectangle((-p/2, -p/2), p, p, facecolor="#F5F8FB",
                           edgecolor="#526D82", linewidth=1.4, linestyle="--"))
    ax.axhline(0, color="#ABB7C4", lw=.8)
    ax.axvline(0, color="#ABB7C4", lw=.8)
    quadrants = ["III", "II", "IV", "I"]
    for vertices, center, angle, quadrant in zip(
            g.vertices(), g.centers(), g.angles(), quadrants):
        color = BLUE if quadrant in ("I", "III") else ORANGE
        ax.add_patch(Polygon(vertices, facecolor=color, edgecolor=INK, lw=1))
        ax.text(*center, f"{quadrant}\n{angle:+.2f}°", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.annotate("", (-p/2, p/2 + 28), (p/2, p/2 + 28),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text(0, p/2 + 39, f"P = {p:.2f} nm", ha="center", va="bottom")
    ax.set(xlim=(-p/2-35, p/2+35), ylim=(-p/2-35, p/2+92),
           xlabel="x (nm)", ylabel="y (nm)", aspect="equal")
    ax.set_title("A  俯视图：四个子体绕各自中心旋转", loc="left", pad=20)
    ax.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(grid[1], projection="3d")
    substrate = float(simulation["substrate_depth_nm"])
    gold = float(simulation["metal_thickness_nm"])
    cell = np.array([[-p/2, -p/2], [p/2, -p/2], [p/2, p/2], [-p/2, p/2]])
    faces, colors = [], []
    blocks = [(cell, -substrate-gold, -substrate, "#D5AC4C"),
              (cell, -substrate, 0, "#D9E6EE")]
    for vertices in g.vertices():
        blocks.append((vertices, 0, g.h, "#4D8FC1"))
    for block in blocks:
        block_faces, block_colors = prism(*block)
        faces.extend(block_faces)
        colors.extend(block_colors)
    # Sort all surfaces together, so the substrate cannot incorrectly hide
    # the bottoms of pillars as separate collection-level depth sorting can.
    ax3.add_collection3d(Poly3DCollection(
        faces, facecolors=colors, edgecolors="#34465A", linewidths=.5))
    ax3.set(xlim=(-p/2, p/2), ylim=(-p/2, p/2),
            zlim=(-substrate-gold, g.h), xlabel="x (nm)", ylabel="y (nm)",
            zlabel="z (nm)")
    ax3.set_xticks([-300, 0, 300])
    ax3.set_yticks([-300, 0, 300])
    ax3.set_zticks([-substrate-gold, -substrate, 0, round(g.h)])
    ax3.set_box_aspect([p, p, g.h+substrate+gold])
    ax3.view_init(elev=23, azim=-62)
    ax3.tick_params(labelsize=9, pad=0)
    ax3.set_title("B  三维结构：单周期截取", loc="left", pad=17)
    fig.text(.54, .135,
             f"As2Se3 高 {g.h:.2f} nm  /  SiO2 厚 {substrate:g} nm  /  Au 厚 {gold:g} nm",
             fontsize=10)
    fig.suptitle(f"与下方光谱对应的真实几何参数 · 样本 {sample_id}", fontsize=17, y=.97,
                 fontweight="bold")
    fig.text(.06, .085,
             f"P={g.p:.2f}   L={g.l:.2f}   H={g.h:.2f}   dX={g.alpha_x:.2f}   dY={g.alpha_y:.2f} nm"
             f"     |     奇数象限 {g.theta_odd_deg:+.2f}°   偶数象限 {g.theta_even_deg:+.2f}°",
             fontsize=11)
    fig.text(.06, .043,
             "正角为逆时针；俯视图颜色仅区分角度组。dX/dY 是未旋转时的名义边缘间隙，旋转后实际间隙会变化。",
             fontsize=10, color="#5D6B79")
    fig.subplots_adjust(top=.83, bottom=.19, left=.07, right=.96)
    fig.savefig(output / f"structure_sample{sample_id}.png", dpi=180)
    with (output / f"structure_sample{sample_id}.svg").open("w", encoding="utf-8", newline="\n") as stream:
        fig.savefig(stream, format="svg")
    plt.close(fig)


def plot_spectrum(wavelength, reflectance, frequency, amplitude, valleys, mean_r, output, sample_id):
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.1))
    fig.subplots_adjust(hspace=.53, top=.86, bottom=.13, left=.09, right=.96)
    fig.suptitle("真实反射光谱 → 傅里叶幅度谱", fontsize=18, y=.97, fontweight="bold")
    fig.text(.09, .905, f"样本 {sample_id}  ·  As2Se3 / SiO2 / Au (100 nm)  ·  700–1700 nm，1000 点",
             fontsize=11, color="#5D6B79")
    ax = axes[0]
    ax.plot(wavelength, reflectance, color=BLUE, lw=1.5)
    for i, valley in enumerate(valleys):
        x, y = valley["wavelength_nm"], valley["reflectance"]
        ax.scatter([x], [y], s=35, color=ORANGE, zorder=4)
        ax.annotate(f"谷 {i+1}: {x:.2f} nm\nR = {y:.3f}", xy=(x, y),
                    xytext=(x+40, .33 if i == 0 else .48),
                    arrowprops=dict(arrowstyle="->", color=ORANGE), fontsize=11,
                    bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#E0E5EB"))
    ax.set(xlim=(wavelength[0], wavelength[-1]), ylim=(0, 1.03),
           xlabel="波长 λ (nm)", ylabel="反射率 R")
    ax.set_title("A  原始光谱：保留所有细节", loc="left", pad=10)
    ax.grid(alpha=.2)

    ax = axes[1]
    ax.semilogy(frequency[1:], amplitude[1:], color=BLUE, lw=1.1)
    ax.set(xlim=(0, frequency[-1]), ylim=(8e-5, .12),
           xlabel="沿波长轴的傅里叶频率 (cycles/μm)", ylabel="单边幅度（对数坐标）")
    ax.set_title("B  傅里叶变换：先减均值，再作 rFFT", loc="left", pad=10)
    ax.grid(alpha=.2, which="both")
    ax.text(.97, .94, f"已移除直流均值 mean(R) = {mean_r:.4f}\n此图仅显示幅度；完整复系数另存",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#E0E5EB"))
    fig.text(.09, .04,
             "未平滑、未加窗、未截断。变换横轴是波长变化的倒数尺度，不是光学频率或时间；两处谷不会一一变成两个峰。",
             fontsize=10, color="#5D6B79")
    fig.savefig(output / "spectrum_and_fft.png", dpi=180)
    with (output / "spectrum_and_fft.svg").open("w", encoding="utf-8", newline="\n") as stream:
        fig.savefig(stream, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "docs/fourier_preprocessing_demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        row = np.flatnonzero(archive["sample_ids"] == args.sample_id)
        if len(row) != 1:
            raise ValueError("sample-id must identify exactly one existing sample")
        row = int(row[0])
        wavelength = archive["wavelength_nm"].astype(np.float64)
        reflectance = archive["spectra"][row].astype(np.float64)
        geometry_values = archive["geometry_values"][row].copy()
    job_path = args.dataset.parent / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8-sig"))
    simulation = job["simulation"]
    # The embedded packed metadata omits metal thickness. Use the original job
    # sidecar for this explicit dimension, never a historical class default.
    if "metal_thickness_nm" not in simulation:
        raise ValueError("The original job.json must state the gold thickness")
    x = wavelength / 1000  # micrometres -> Fourier frequency in cycles/um
    dx = float(np.diff(x).mean())
    if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-14):
        raise ValueError("This demonstration requires uniformly sampled wavelengths")
    if not np.isfinite(reflectance).all():
        raise ValueError("Spectrum contains non-finite values")
    mean_r = float(reflectance.mean())
    coefficients = rfft(reflectance - mean_r)
    frequency = rfftfreq(len(x), d=dx)
    amplitude = 2 * np.abs(coefficients) / len(x)
    amplitude[0] /= 2
    if len(x) % 2 == 0:
        amplitude[-1] /= 2
    reconstructed = irfft(coefficients, n=len(x)) + mean_r
    reconstruction_error = float(np.max(np.abs(reflectance - reconstructed)))
    if reconstruction_error > 1e-12:
        raise RuntimeError("Full-coefficient inverse transform failed to reconstruct input")
    indices, properties = find_peaks(-reflectance, prominence=.1)
    if len(indices) < 2:
        raise ValueError("Selected sample does not have two sufficiently prominent valleys")
    # Label the two largest prominences, while retaining all weaker features.
    selected = np.argsort(properties["prominences"])[-2:]
    indices = np.sort(indices[selected])
    widths = peak_widths(-reflectance, indices, rel_height=.5)[0] * dx * 1000
    valleys = [dict(wavelength_nm=float(wavelength[i]), reflectance=float(reflectance[i]),
                    half_prominence_width_nm=float(width))
               for i, width in zip(indices, widths)]
    g = Geometry(*geometry_values[:5], theta_odd_deg=geometry_values[5],
                 theta_even_deg=geometry_values[6])
    result = {
        "sample_id": args.sample_id,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
        "quality_level": job["quality_level"],
        "points": len(x), "wavelength_step_nm": dx * 1000,
        "mean_reflectance": mean_r,
        "endpoint_reflectance_difference": float(reflectance[-1] - reflectance[0]),
        "transform": "scipy.fft.rfft(R - mean(R)); no smoothing/windowing/truncation/padding",
        "frequency_unit": "cycles per micrometre along the wavelength coordinate",
        "amplitude_normalization": "2*abs(rfft)/N, with DC and even-N Nyquist not doubled",
        "inverse_max_absolute_error": reconstruction_error,
        "major_valleys": valleys,
        "geometry_names": ["P", "L", "H", "dX", "dY", "theta_odd_deg", "theta_even_deg"],
        "geometry_values": geometry_values.tolist(),
        "centres_nm": np.array(g.centers()).tolist(),
        "polygon_vertices_nm": [v.tolist() for v in g.vertices()],
        "substrate_depth_nm": simulation["substrate_depth_nm"],
        "metal_thickness_nm": simulation["metal_thickness_nm"],
        "resonator_material": simulation["resonator_material"],
        "substrate_material": "SiO2 (Glass) - Palik",
        "metal_material": simulation["metal_material"],
    }
    np.savez_compressed(args.output / f"sample{args.sample_id}_full_fft.npz", wavelength_nm=wavelength,
                        reflectance=reflectance, mean_reflectance=mean_r,
                        frequency_cycles_per_um=frequency, fft_complex=coefficients,
                        single_sided_amplitude=amplitude, reconstructed_reflectance=reconstructed,
                        geometry_values=geometry_values)
    (args.output / "analysis.json").write_bytes(
        (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    style()
    plot_structure(g, simulation, args.output, args.sample_id)
    plot_spectrum(wavelength, reflectance, frequency, amplitude, valleys, mean_r,
                  args.output, args.sample_id)
    # Matplotlib's SVG path writer leaves spaces at line ends; normalize only
    # these two generated text exports for a clean, portable Git diff.
    for svg in (args.output / f"structure_sample{args.sample_id}.svg",
                args.output / "spectrum_and_fft.svg"):
        svg.write_bytes(("\n".join(line.rstrip() for line in
                                  svg.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8"))
    print(json.dumps({k: result[k] for k in ("sample_id", "points", "major_valleys",
                                             "inverse_max_absolute_error")}, indent=2))
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
