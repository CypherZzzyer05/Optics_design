"""Real FDTD check: fixed square centers, zero/positive/negative rotation spectra."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from fdtd_pipeline.config import Geometry, PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator
from fdtd_pipeline.pipeline import atomic_json, atomic_npz


def validate_rotation(config_path, output_dir, angle=20):
    config = load_config(config_path)
    if not 0 < angle < 45:
        raise ValueError("The check angle must be between 0 and 45 degrees")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Geometry(3200, 900, 1000, 250, 300)
    results, native_objects = {}, {}
    with FDTDSimulator(config.simulation, config.lumapi_path) as simulator:
        for name, degrees in (("zero", 0), ("positive", angle), ("negative", -angle)):
            geometry = replace(base, rotation_deg=degrees)
            print(f"Real FDTD: {name}, rotation={degrees:g} deg", flush=True)
            result = simulator.run(geometry, output_dir / (name + ".fsp"))
            objects = []
            for i, center in enumerate(base.centers()):
                x = float(simulator.fdtd.getnamed(f"si_{i}", "x")) * 1e9
                y = float(simulator.fdtd.getnamed(f"si_{i}", "y")) * 1e9
                native_angle = float(simulator.fdtd.getnamed(f"si_{i}", "rotation 1"))
                if not np.allclose([x, y], center, atol=1e-6) or not np.isclose(native_angle, -degrees):
                    raise ValueError("Native solver rotation or square center differs from the requested geometry")
                objects.append({"name": f"si_{i}", "center_nm": [x, y], "native_rotation_deg": native_angle})
            native_objects[name] = objects
            atomic_npz(output_dir / (name + ".npz"), **result,
                       geometry_nm=base.array(), rotation_deg=np.float64(degrees))
            results[name] = result
    mirror_r = float(np.max(np.abs(results['positive']['reflectance'] - results['negative']['reflectance'])))
    mirror_t = float(np.max(np.abs(results['positive']['transmittance'] - results['negative']['transmittance'])))
    rotation_change = float(np.max(np.abs(results['positive']['reflectance'] - results['zero']['reflectance'])))
    report = {
        "backend": "real Lumerical FDTD", "angle_deg": angle,
        "rotation_mode": "each_square_own_center_common_z_angle",
        "project_angle_convention": "counterclockwise viewed from +z; native rotation 1 is its negative",
        "geometry_nm": base.array().tolist(), "native_objects": native_objects,
        "mirror_max_reflectance_difference": mirror_r,
        "mirror_max_transmittance_difference": mirror_t,
        "rotation_max_reflectance_change_from_zero": rotation_change,
        "mirror_tolerance": .005, "mirror_equivalence_passed": max(mirror_r, mirror_t) < .005,
        "statuses": {key: int(result['status']) for key, result in results.items()},
        "simulation_settings": config.metadata()['simulation'],
        "note": "Coarse-mesh engineering validation; predicted sign is an equivalent candidate, not uniquely identifiable from this reflection input.",
    }
    atomic_json(output_dir / 'validation_report.json', report)
    plot_rotation(base, angle, results, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report['mirror_equivalence_passed']:
        raise ValueError("Mirror spectra disagree beyond tolerance; inspect mesh and model before training")
    return report


def plot_rotation(base, angle, results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle('各方块绕自身中心旋转 · 真实 FDTD 对照', fontsize=17)
    for ax, degrees in zip(axes[0], (0, angle)):
        geometry = replace(base, rotation_deg=degrees)
        ax.add_patch(Rectangle((-base.p / 2, -base.p / 2), base.p, base.p,
                               facecolor='#edf2f7', edgecolor='#61718a', linestyle='--'))
        for vertices in geometry.vertices():
            ax.add_patch(Polygon(vertices, facecolor='#315f9c', edgecolor='#17395f'))
        centers = np.array(base.centers())
        ax.scatter(centers[:, 0], centers[:, 1], s=12, color='#d58b23', label='固定中心')
        ax.set(xlim=(-base.p * .55, base.p * .55), ylim=(-base.p * .55, base.p * .55),
               aspect='equal', xlabel='x (nm)', ylabel='y (nm)', title=f'θ = {degrees:g}°')
        ax.legend(fontsize=9)
    for name, text, style in (('zero', '0°', '-'), ('positive', f'+{angle:g}°', '-'), ('negative', f'−{angle:g}°', '--')):
        axes[1, 0].plot(results[name]['wavelength_nm'] / 1000, results[name]['reflectance'], style, label=text)
    axes[1, 0].set(xlabel='波长 (μm)', ylabel='反射率 R', title='旋转后的光谱与镜像对照')
    axes[1, 0].legend()
    difference = results['positive']['reflectance'] - results['negative']['reflectance']
    axes[1, 1].plot(results['zero']['wavelength_nm'] / 1000, difference)
    axes[1, 1].set(xlabel='波长 (μm)', ylabel='R(+θ) − R(−θ)', title='正负角度反射率差异')
    for ax in axes[1]:
        ax.grid(alpha=.2)
    fig.savefig(output_dir / 'rotation_validation.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/rotation_pilot.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data/rotation_validation')
    parser.add_argument('--angle', type=float, default=20)
    args = parser.parse_args()
    validate_rotation(args.config, args.output_dir, args.angle)


if __name__ == '__main__':
    main()
