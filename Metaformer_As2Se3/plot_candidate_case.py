"""Draw a verified strict-mesh candidate and its two measured numerical troughs."""
import argparse
from pathlib import Path
import numpy as np

from compare_double_mesh import read_case
from explore_double_dips import double_pair
from fdtd_pipeline.config import PROJECT_ROOT, Geometry


def plot(directory, case_id, output):
    case = read_case(directory, case_id)
    values = case['entry']['geometry_values']
    geometry = Geometry(*values[:5], theta_odd_deg=values[5], theta_even_deg=values[6])
    w, r = case['data']['wavelength_nm'], case['data']['reflectance']
    pair = double_pair(w, r)['dominant_pair']
    if pair is None:
        raise ValueError('This case does not pass the two dominant trough criteria')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.65))
    ax = fig.add_subplot(grid[:, 0])
    for i, vertices in enumerate(geometry.vertices()):
        odd = i in (0, 3)
        label = f"{'I/III' if odd else 'II/IV'}: {values[5 if odd else 6]:.3f} deg" if i in (0, 1) else None
        ax.add_patch(Polygon(vertices, color='#e29641' if odd else '#417bc4', alpha=.9, label=label))
    ax.set(xlim=(-geometry.p/2, geometry.p/2), ylim=(-geometry.p/2, geometry.p/2), aspect='equal',
           xlabel='x (nm)', ylabel='y (nm)', title='As2Se3 tetramer, top view')
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', fontsize=8)
    limits = (pair['dips'][0]['wavelength_nm']-35, pair['dips'][1]['wavelength_nm']+35)
    for row, xlim in enumerate(((w[0], w[-1]), limits)):
        a = fig.add_subplot(grid[row, 1])
        a.plot(w, r, lw=1.3, color='#28558b')
        for index, dip in enumerate(pair['dips']):
            a.plot(dip['wavelength_nm'], dip['reflectance_min'], 'o', color='#c64232')
            left_label = row == 0 and index == 0
            a.annotate(f"{dip['wavelength_nm']:.1f} nm\nR={dip['reflectance_min']:.3f}",
                       (dip['wavelength_nm'], dip['reflectance_min']), xytext=(-8 if left_label else 8, 12),
                       ha='right' if left_label else 'left', textcoords='offset points', fontsize=9)
        a.set(xlim=xlim, ylim=(-.02, 1.03), xlabel='Wavelength (nm)', ylabel='Reflectance')
        a.grid(alpha=.2)
    fig.suptitle(f"Real FDTD: 350 nm SiO2 / 150 nm {case['entry']['metal']} | maximum {case['settings'].maximum_mesh_nm:g} nm mesh")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=PROJECT_ROOT/'data/double_dip_controls8')
    parser.add_argument('--case-id', default='Au_full_difference')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'research_results/double_dip_controls8/candidate.png')
    args = parser.parse_args()
    plot(args.directory, args.case_id, args.output)
