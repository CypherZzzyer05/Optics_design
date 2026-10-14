"""Display the audited Au4 double-dip candidate without starting a solver."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'research_results/gold_mesh_comparison/raw/cases/Au4'
    sample = source / 'sample.npz'
    state = json.loads((source / 'state.json').read_text(encoding='utf-8'))
    audit = json.loads((source / 'native_audit.json').read_text(encoding='utf-8'))
    report_path = ROOT / 'research_results/gold_mesh_comparison/analysis.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sample_sha = digest(sample)
    if not (sample_sha == state['sample_sha256'] == audit['sample_sha256']
            == report['cases']['Au4']['sample_sha256'] and audit['status'] == 2
            and audit['native_spectra_match_export'] and report['complete_matrix']):
        raise ValueError('The archived Au4 spectrum is not verified')
    with np.load(sample) as z:
        wavelength = z['wavelength_nm'].copy()
        reflection = z['reflectance'].copy()
        transmission = z['transmittance'].copy()
        assert int(z['status']) == 2
    assert len(wavelength) == 1692 and np.isfinite([wavelength, reflection, transmission]).all()
    assert np.all(np.diff(wavelength) > 0)
    dips = [row['second'] for row in report['comparisons']['Au_8nm_vs_4nm']['tracked_spectral_features']]
    assert len(dips) == 2
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams.update({'font.family': font, 'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.subplots_adjust(left=.07, right=.97, top=.82, bottom=.23, hspace=.42, wspace=.23)
    fig.suptitle('当前双谷候选结构：反射与透射光谱（镀金，4 nm 网格）', fontsize=17, y=.97)
    g = state['recipe']['geometry_values']
    fig.text(.5, .91, f'P={g[0]:.2f}，L={g[1]:.2f}，H={g[2]:.2f}，dX={g[3]:.2f}，dY={g[4]:.2f} nm；奇/偶角={g[5]:.2f}° / {g[6]:.2f}°',
             ha='center', fontsize=11)
    fig.text(.5, .872, 'As2Se3 / 350 nm SiO2 / 150 nm Au；法向入射，x 偏振', ha='center', fontsize=11)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel('波长（nm）')
        ax.set_ylabel('功率比例（%）')
        ax.set_ylim(-2, 102)
    axes[0, 0].plot(wavelength, reflection * 100, color='#b57113', lw=1.6)
    axes[0, 0].set_title('反射 R：完整 700-1700 nm 波段')
    axes[0, 0].set_xlim(700, 1700)
    axes[1, 0].plot(wavelength, transmission * 100, color='#167c9c', lw=1.8)
    axes[1, 0].set_title('透射 T：完整波段，数值接近 0')
    axes[1, 0].set_xlim(700, 1700)
    axes[1, 0].text(.04, .55, '金属背板下，透射在本次计算精度内近零', transform=axes[1, 0].transAxes,
                    fontsize=11, color='#167c9c')
    for ax, dip in zip(axes[:, 1], dips):
        center = dip['wavelength_nm']
        i = int(np.argmin(abs(wavelength - center)))
        assert abs(wavelength[i] - center) < 1e-6
        assert abs(reflection[i] - dip['reflectance_min']) < 1e-12
        ax.plot(wavelength, reflection * 100, color='#b57113', lw=1.8)
        ax.scatter([center], [100 * reflection[i]], color='#ba3e36', s=32, zorder=5)
        ax.set_xlim(center - 12, center + 12)
        ax.set_title(f'反射谷局部放大：{center:.2f} nm')
        ax.annotate(f'{center:.2f} nm\nR = {100 * reflection[i]:.2f}%',
                    xy=(center, 100 * reflection[i]), xytext=(center + 4.5, 18),
                    arrowprops={'arrowstyle': '->', 'color': '#ba3e36'}, fontsize=11, color='#ba3e36')
        axes[0, 0].axvline(center, color='#ba3e36', lw=.7, ls='--', alpha=.55)
    fig.text(.07, .12, '材料限制：当前 Au 拟合几乎漏掉损耗；图示仅代表保存模型，修正后的器件性能尚未验证。',
             color='#9c332b', fontsize=11)
    fig.text(.07, .075, f'T 原始数值范围：{100*transmission.min():.6f}% 至 {100*transmission.max():.6f}%；微小负值为数值残差，绘图未截断。', fontsize=10)
    fig.text(.07, .03, '1692 个已核验原生频率点；直接连接实算点，未平滑或生成额外谱线数据。', color='#56616b', fontsize=10)
    destination = ROOT / 'output/figures'
    destination.mkdir(parents=True, exist_ok=True)
    plot = destination / 'Candidate_Au4_RT.png'
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    provenance = {'case': 'Au4', 'sample_sha256': sample_sha, 'analysis_sha256': digest(report_path),
                  'native_status': 2, 'spectral_points': len(wavelength), 'geometry_values': g,
                  'reflectance_dips': dips, 'raw_transmittance_min': float(transmission.min()),
                  'raw_transmittance_max': float(transmission.max()), 'clipped_or_smoothed': False,
                  'material_interpretation': report['material_interpretation'], 'image_sha256': digest(plot)}
    (destination / 'Candidate_Au4_RT_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files = {path.name: digest(path) for path in destination.iterdir() if path.is_file() and path.name != 'checksums.json'}
    (destination / 'checksums.json').write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
    print(plot)


if __name__ == '__main__':
    main()
