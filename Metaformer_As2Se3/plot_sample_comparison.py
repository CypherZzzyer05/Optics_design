"""Compare real rotation-sample spectra and two selected geometries in one figure."""

import argparse
import json
from pathlib import Path

import numpy as np

from fdtd_pipeline.config import Geometry, PROJECT_ROOT, ROTATION_PARAMETER_NAMES
from fdtd_pipeline.pipeline import atomic_json, checksum


def plot_sample_comparison(dataset_path, sample_ids=None, pair=(0, 1), show=True):
    import matplotlib
    matplotlib.use("TkAgg" if show else "Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    dataset_path = Path(dataset_path).resolve()
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if tuple(data['parameter_names']) != ROTATION_PARAMETER_NAMES:
        raise ValueError('This comparison requires the six-parameter rotation dataset')
    if np.any(data['status'] != 2):
        raise ValueError('Comparison requires converged real samples')
    lookup = {int(identifier): row for row, identifier in enumerate(data['sample_ids'])}
    selected = list(lookup)[:6] if sample_ids is None else list(sample_ids)
    pair = tuple(pair)
    if not selected or len(pair) != 2 or pair[0] == pair[1] or len(set(selected)) != len(selected):
        raise ValueError('Select unique sample IDs and two different comparison IDs')
    if any(identifier not in lookup for identifier in [*selected, *pair]):
        raise ValueError('A requested sample ID is missing from the dataset')

    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .035, 1, .88), hspace=.07, wspace=.06)
    fig.canvas.manager.set_window_title('MetaFormer | 真实旋转样本光谱对比')
    fig.suptitle(f'{len(selected)} 个真实 FDTD 样本反射谱 · 样本 {pair[0]:02d} / {pair[1]:02d} 结构对照',
                 fontsize=18, fontweight='bold', color='#152849')
    palette = ['#2667d8', '#e58624', '#049b86', '#cf4d68', '#8b68bd', '#657789']
    colors = {identifier: palette[i % len(palette)] for i, identifier in enumerate(selected)}
    for identifier in pair:
        colors.setdefault(identifier, palette[len(colors) % len(palette)])
    geometries = {identifier: Geometry(*data['geometry_values'][lookup[identifier]]) for identifier in pair}
    extent = max(geometry.p for geometry in geometries.values()) * .55
    for ax, identifier in zip(axes[0], pair):
        geometry = geometries[identifier]
        ax.add_patch(Rectangle((-geometry.p / 2, -geometry.p / 2), geometry.p, geometry.p,
                               facecolor='#edf2f7', edgecolor='#61718a', linestyle='--'))
        for vertices in geometry.vertices():
            ax.add_patch(Polygon(vertices, facecolor=colors[identifier], edgecolor='#17395f', alpha=.9))
        centers = np.asarray(geometry.centers())
        ax.scatter(centers[:, 0], centers[:, 1], s=14, color='#e9a32a')
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect='equal',
               xlabel='x (nm)', ylabel='y (nm)', title=f'样本 {identifier:02d} · θ = {geometry.rotation_deg:+.2f}°')
        ax.text(.5, -.21,
                f'p = {geometry.p:.1f}   l = {geometry.l:.1f}   h = {geometry.h:.1f} nm\n'
                f'αx = {geometry.alpha_x:.1f}   αy = {geometry.alpha_y:.1f} nm',
                transform=ax.transAxes, ha='center', va='top', fontsize=10, color='#334969')

    wavelength_um = data['wavelength_nm'] / 1000
    for identifier in selected:
        row = lookup[identifier]
        angle = float(data['rotation_deg'][row])
        axes[1, 0].plot(wavelength_um, data['spectra'][row], lw=1.5, color=colors[identifier],
                        label=f'{identifier:02d}  θ={angle:+.2f}°')
    axes[1, 0].set(xlabel='波长 (μm)', ylabel='反射率 R', title=f'{len(selected)} 条真实反射谱（全部采样点）',
                   xlim=(wavelength_um[0], wavelength_um[-1]), ylim=(-.02, 1.02))
    axes[1, 0].legend(ncol=2, fontsize=9, loc='upper right')

    first, second = [data['spectra'][lookup[identifier]].astype(np.float64) for identifier in pair]
    difference = second - first
    peak_row = int(np.argmax(np.abs(difference)))
    peak_wavelength = float(wavelength_um[peak_row])
    for identifier in pair:
        axes[1, 1].plot(wavelength_um, data['spectra'][lookup[identifier]], lw=1.8,
                        color=colors[identifier], label=f'样本 {identifier:02d}')
    # 自动围绕差异最大的波长放大，仍直接显示原始光谱，不做平滑或插值。
    zoom_min = max(float(wavelength_um[0]), peak_wavelength - .6)
    zoom_max = min(float(wavelength_um[-1]), peak_wavelength + .6)
    axes[1, 1].set(xlabel='波长 (μm)', ylabel='反射率 R', title=f'{pair[0]:02d} / {pair[1]:02d} 峰区放大',
                   xlim=(zoom_min, zoom_max), ylim=(-.02, 1.02))
    axes[1, 1].vlines(peak_wavelength, first[peak_row], second[peak_row], color='#334969', linestyle='--')
    axes[1, 1].scatter([peak_wavelength] * 2, [first[peak_row], second[peak_row]], s=25,
                      color=[colors[pair[0]], colors[pair[1]]], zorder=4)
    axes[1, 1].text(.98, .76, f'最大 |ΔR| = {abs(difference[peak_row]):.3f}\nλ = {peak_wavelength:.3f} μm',
                    transform=axes[1, 1].transAxes, ha='right', va='top', color='#334969',
                    bbox={'facecolor': 'white', 'edgecolor': '#d9e1ed', 'alpha': .95})
    axes[1, 1].legend(loc='upper right')
    for ax in axes[1]:
        ax.grid(alpha=.2)
    fig.text(.02, .012, '这些样本的五个尺寸和旋转角同时变化；本图比较完整结构，不能单独归因于旋转角。',
             fontsize=10, color='#61718a')
    output = dataset_path.parent / 'sample_spectra_comparison.png'
    fig.savefig(output, dpi=170)
    fig.savefig(output.with_suffix('.svg'))
    report = {
        'dataset': str(dataset_path), 'dataset_sha256': checksum(dataset_path),
        'sample_ids': selected, 'pair_ids': list(pair),
        'parameter_names': data['parameter_names'].tolist(), 'parameter_units': data['parameter_units'].tolist(),
        'geometry_values': {str(identifier): data['geometry_values'][lookup[identifier]].tolist()
                            for identifier in [*selected, *pair]},
        'pair_max_absolute_reflectance_difference': float(abs(difference[peak_row])),
        'pair_max_difference_wavelength_nm': float(data['wavelength_nm'][peak_row]),
        'reflectance_at_max_difference': {str(pair[0]): float(first[peak_row]), str(pair[1]): float(second[peak_row])},
        'note': 'Real original reflection spectra; all six parameters change across samples; not an isolated angle sweep.',
    }
    atomic_json(output.with_suffix('.json'), report)
    print(json.dumps({'output': str(output), 'sample_ids': selected,
                      'pair_max_absolute_reflectance_difference': report['pair_max_absolute_reflectance_difference'],
                      'pair_max_difference_wavelength_nm': report['pair_max_difference_wavelength_nm']}, indent=2), flush=True)
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT_ROOT / 'data/rotation_pilot/dataset.npz')
    parser.add_argument('--samples', nargs='+', type=int, help='Sample IDs; defaults to the first six')
    parser.add_argument('--pair', nargs=2, type=int, default=[0, 1], help='Two sample IDs for structure and peak comparison')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    plot_sample_comparison(args.dataset, args.samples, args.pair, not args.no_show)


if __name__ == '__main__':
    main()
