"""Reuse or solve real FDTD spectra at fixed dimensions, then show a few positive angles."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from fdtd_pipeline.config import Geometry, PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator, validate_spectrum
from fdtd_pipeline.pipeline import atomic_json, atomic_npz, checksum


BASE_GEOMETRY = Geometry(3200, 900, 1000, 250, 300)


def angle_key(angle):
    return f'angle_{angle:g}'.replace('.', 'p')


def load_result(path, geometry, settings):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    if (not np.array_equal(result['geometry_nm'], geometry.array())
            or float(result['rotation_deg']) != geometry.rotation_deg):
        raise ValueError('Cached rotation spectrum has different dimensions or angle')
    validate_spectrum(result['wavelength_nm'], result['reflectance'], result['transmittance'],
                      settings, int(result['status']))
    return result


def collect_results(config, output_dir, angles):
    geometry_nm = BASE_GEOMETRY.array().tolist()
    settings = config.metadata()['simulation']
    fingerprint = hashlib.sha256(json.dumps({'simulation': settings, 'geometry_nm': geometry_nm},
                                           sort_keys=True).encode()).hexdigest()
    reference_dir = PROJECT_ROOT / 'data/rotation_validation'
    reference_report = reference_dir / 'validation_report.json'
    reference = json.loads(reference_report.read_text(encoding='utf-8')) if reference_report.exists() else {}
    results, pending = {}, []
    output_dir.mkdir(parents=True, exist_ok=True)
    for angle in angles:
        geometry = replace(BASE_GEOMETRY, rotation_deg=angle)
        path = output_dir / (angle_key(angle) + '.npz')
        if path.exists():
            result = load_result(path, geometry, config.simulation)
            if str(result['simulation_config_fingerprint'].item()) != fingerprint:
                raise ValueError('Cached physics settings changed; use a separate output directory')
            results[angle] = result
            print(f'Reuse completed FDTD: {angle:g} deg', flush=True)
            continue
        reference_name = {0: 'zero', 20: 'positive'}.get(angle)
        source = reference_dir / f'{reference_name}.npz'
        if (reference_name and source.exists() and reference.get('simulation_settings') == settings
                and reference.get('geometry_nm') == geometry_nm):
            result = load_result(source, geometry, config.simulation)
            result.update(simulation_config_fingerprint=np.array(fingerprint),
                          source_npz=np.array(str(source)), source_sha256=np.array(checksum(source)),
                          native_objects_json=np.array(json.dumps(reference['native_objects'][reference_name])))
            atomic_npz(path, **result)
            results[angle] = result
            print(f'Reuse verified FDTD: {angle:g} deg, {source}', flush=True)
        else:
            pending.append((angle, geometry, path))
    if pending:
        with FDTDSimulator(config.simulation, config.lumapi_path) as simulator:
            for angle, geometry, path in pending:
                print(f'Real FDTD: {angle:g} deg; remaining five parameters fixed', flush=True)
                result = simulator.run(geometry, output_dir / (angle_key(angle) + '.fsp'))
                objects = []
                for i, center in enumerate(BASE_GEOMETRY.centers()):
                    x = float(np.asarray(simulator.fdtd.getnamed(f'si_{i}', 'x')).item()) * 1e9
                    y = float(np.asarray(simulator.fdtd.getnamed(f'si_{i}', 'y')).item()) * 1e9
                    native_angle = float(np.asarray(simulator.fdtd.getnamed(f'si_{i}', 'rotation 1')).item())
                    if not np.allclose([x, y], center, atol=1e-6) or not np.isclose(native_angle, -angle):
                        raise ValueError('Native square centers or rotation differ from the requested geometry')
                    objects.append({'name': f'si_{i}', 'center_nm': [x, y], 'native_rotation_deg': native_angle})
                result.update(geometry_nm=geometry.array(), rotation_deg=np.float64(angle),
                              simulation_config_fingerprint=np.array(fingerprint),
                              source_npz=np.array(str(path)), native_objects_json=np.array(json.dumps(objects)))
                atomic_npz(path, **result)
                results[angle] = result
    baseline = results[0]['reflectance']
    report = {
        'backend': 'real Lumerical FDTD', 'geometry_nm': geometry_nm,
        'angles_deg': angles, 'simulation_settings': settings,
        'simulation_config_fingerprint': fingerprint,
        'statuses': {str(angle): int(results[angle]['status']) for angle in angles},
        'max_reflectance_change_from_zero': {str(angle): float(np.max(np.abs(results[angle]['reflectance'] - baseline)))
                                            for angle in angles},
        'samples': {str(angle): {'npz': str(output_dir / (angle_key(angle) + '.npz')),
                                'sha256': checksum(output_dir / (angle_key(angle) + '.npz')),
                                'source_npz': str(results[angle]['source_npz'].item()),
                                'native_objects': json.loads(str(results[angle]['native_objects_json'].item()))}
                    for angle in angles},
        'note': 'Only the common self-center rotation changes. Real original power spectra; coarse-mesh engineering check.',
    }
    atomic_json(output_dir / 'comparison_report.json', report)
    return results, report


def draw_structure(ax, geometry, color, title):
    from matplotlib.patches import Polygon, Rectangle
    ax.add_patch(Rectangle((-geometry.p / 2, -geometry.p / 2), geometry.p, geometry.p,
                           facecolor='#edf2f7', edgecolor='#61718a', linestyle='--'))
    for vertices in geometry.vertices():
        ax.add_patch(Polygon(vertices, facecolor=color, edgecolor='#17395f', alpha=.9))
    centers = np.asarray(geometry.centers())
    ax.scatter(centers[:, 0], centers[:, 1], s=14, color='#e9a32a')
    ax.set(xlim=(-geometry.p * .55, geometry.p * .55), ylim=(-geometry.p * .55, geometry.p * .55),
           aspect='equal', xlabel='x (nm)', ylabel='y (nm)', title=title)


def plot_comparison(results, output_dir, show):
    import matplotlib
    matplotlib.use('TkAgg' if show else 'Agg', force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11})
    angles = sorted(results)
    colors = ['#315f9c', '#e58624', '#049b86', '#8b68bd']
    g = BASE_GEOMETRY
    dimensions = f'p = {g.p:g} nm   l = {g.l:g} nm   h = {g.h:g} nm   αx = {g.alpha_x:g} nm   αy = {g.alpha_y:g} nm'
    structure_dir = output_dir / 'structures'
    structure_dir.mkdir(exist_ok=True)
    # 单独输出结构图，方便在 PyCharm 打开或用于报告。
    for angle, color in zip(angles, colors):
        single, ax = plt.subplots(figsize=(5, 5.5), layout='constrained')
        single.get_layout_engine().set(rect=(0, .1, 1, .88))
        label = f'+{angle:g}°' if angle else '0°'
        draw_structure(ax, replace(g, rotation_deg=angle), color, f'各方块绕自身中心旋转 · θ = {label}')
        single.text(.5, .022, f'p={g.p:g}, l={g.l:g}, h={g.h:g} nm\nαx={g.alpha_x:g}, αy={g.alpha_y:g} nm · 黄色点为固定中心',
                    ha='center', fontsize=10, color='#61718a')
        path = structure_dir / (angle_key(angle) + '.png')
        single.savefig(path, dpi=180)
        single.savefig(path.with_suffix('.svg'))
        plt.close(single)
    fig = plt.figure(figsize=(13.5, 9), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .045, 1, .82), hspace=.08)
    grid = fig.add_gridspec(2, len(angles), height_ratios=[1, 1.1])
    fig.canvas.manager.set_window_title('MetaFormer | 0° 与正角度真实 FDTD 对照')
    fig.suptitle('固定尺寸下的旋转对照 · 只显示 0° 和正角度', fontsize=19, fontweight='bold', y=.985, color='#152849')
    fig.text(.5, .925, dimensions, ha='center', color='#50617c', fontsize=11)
    for i, (angle, color) in enumerate(zip(angles, colors)):
        ax = fig.add_subplot(grid[0, i])
        label = f'+{angle:g}°' if angle else '0°'
        draw_structure(ax, replace(g, rotation_deg=angle), color, f'θ = {label}')
    spectrum_ax = fig.add_subplot(grid[1, :])
    for angle, color in zip(angles, colors):
        result = results[angle]
        spectrum_ax.plot(result['wavelength_nm'] / 1000, result['reflectance'], color=color, lw=1.8,
                         label=f'+{angle:g}°' if angle else '0°')
    spectrum_ax.set(xlabel='波长 (μm)', ylabel='反射率 R', title='真实 FDTD 反射谱 · 其余五个参数固定',
                    xlim=(4, 8), ylim=(-.02, 1.02))
    spectrum_ax.grid(alpha=.2)
    spectrum_ax.legend(ncol=len(angles), loc='upper right', fontsize=12)
    fig.text(.03, .018, '黄色点为各方块的固定中心；仅共同旋转角发生变化。所有曲线均由真实 FDTD 求解并收敛。',
             fontsize=10, color='#61718a')
    output = output_dir / 'positive_rotation_comparison.png'
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    print(f'Comparison figure: {output}\nIndividual structures: {structure_dir}', flush=True)
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/rotation_pilot.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data/rotation_positive')
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 20, 30])
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    angles = sorted(set(args.angles))
    if not 1 <= len(angles) <= 4 or 0 not in angles or any(not np.isfinite(a) or a < 0 or a > 45 for a in angles):
        parser.error('Choose at most four angles in [0,45], including 0 degrees')
    for angle in angles:
        replace(BASE_GEOMETRY, rotation_deg=angle)  # 预先拒绝相交/越界角度，不启动无效仿真。
    output_dir = args.output_dir.resolve()
    results, report = collect_results(load_config(args.config), output_dir, angles)
    print(json.dumps({'angles_deg': angles, 'statuses': report['statuses'],
                      'max_reflectance_change_from_zero': report['max_reflectance_change_from_zero']}, indent=2), flush=True)
    plot_comparison(results, output_dir, not args.no_show)


if __name__ == '__main__':
    main()
