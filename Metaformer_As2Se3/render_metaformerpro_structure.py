"""Render an inverse candidate and its input spectrum without an FDTD run."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from fdtd_pipeline.config import Geometry


def prism(vertices, bottom, top):
    lower = np.column_stack((vertices, np.full(4, bottom)))
    upper = np.column_stack((vertices, np.full(4, top)))
    return [lower, upper] + [np.array([lower[i], lower[(i+1) % 4],
                                       upper[(i+1) % 4], upper[i]]) for i in range(4)]


def render_structure(prediction, wavelength_nm, reflectance, output, source_label):
    """Draw the checkpoint's stack; untagged historical weights use the v1 stack."""
    values = prediction['parameters']
    geometry = Geometry(*(values[name] for name in ('p', 'l', 'h', 'alpha_x', 'alpha_y')),
                        theta_odd_deg=values['theta_odd_deg'],
                        theta_even_deg=values['theta_even_deg'])
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig = plt.figure(figsize=(14, 9), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.6, 1),
                           left=.075, right=.96, bottom=.19, top=.88,
                           hspace=.36, wspace=.23)
    fig.suptitle('MetaformerPro · 固定光谱 → 结构候选', fontsize=18, weight='bold', y=.975)
    fig.text(.5, .932, f'输入：{source_label}    |    选中模型 v{prediction["version"]}',
             ha='center', fontsize=11, color='#42556a')
    top = fig.add_subplot(grid[0, 0])
    colors = ['#c87642' if i in (0, 3) else '#287fa5' for i in range(4)]
    quadrants = ['III', 'II', 'IV', 'I']
    for index, (vertices, center, angle, color) in enumerate(zip(
            geometry.vertices(), geometry.centers(), geometry.angles(), colors)):
        top.add_patch(Polygon(vertices, facecolor=color, edgecolor='#253c50', linewidth=.9))
        top.text(*center, f'{quadrants[index]}\n{angle:.2f}°', color='white',
                 ha='center', va='center', fontsize=10, weight='bold')
    p = geometry.p
    top.add_patch(Rectangle((-p/2, -p/2), p, p, fill=False,
                            linestyle='--', edgecolor='#66788a', linewidth=1.2))
    top.axhline(0, color='#66788a', lw=.7, alpha=.4)
    top.axvline(0, color='#66788a', lw=.7, alpha=.4)
    top.set(xlim=(-p*.55, p*.55), ylim=(-p*.55, p*.55), aspect='equal',
            xlabel='x (nm)', ylabel='y (nm)', title='俯视图：I / III 同角，II / IV 同角')
    top.tick_params(labelsize=9)

    view = fig.add_subplot(grid[0, 1], projection='3d')
    footprint = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * p/2
    settings = prediction.get('simulation_settings')
    legacy = settings is None
    settings = settings or {'metal_position': 'below_substrate',
                            'substrate_termination': 'finite',
                            'metal_thickness_nm': 100, 'substrate_depth_nm': 350}
    metal_depth = float(settings['metal_thickness_nm'])
    thick_substrate = settings.get('substrate_termination', 'finite') == 'pml'
    # The displayed portion of a thick substrate is a cutaway, not its physical thickness.
    substrate_depth = 350. if thick_substrate else float(settings['substrate_depth_nm'])
    stack_bottom = -metal_depth - substrate_depth
    if settings.get('metal_position', 'below_substrate') == 'above_substrate':
        layers = [(footprint, stack_bottom, -metal_depth, '#b6ccdd'),
                  (footprint, -metal_depth, 0, '#d9b34c')]
        layer_labels = [(f'Au {metal_depth:g} nm', '#98701c'),
                        ('厚 SiO$_2$（截示）\n↓ 延伸至下方 PML' if thick_substrate
                         else f'SiO$_2$ {substrate_depth:g} nm', '#627f95')]
        stack_title = '三维图：SiO$_2$ / Au / 四聚体（从下到上）'
    else:
        layers = [(footprint, stack_bottom, -substrate_depth, '#d9b34c'),
                  (footprint, -substrate_depth, 0, '#b6ccdd')]
        layer_labels = [(f'SiO$_2$ {substrate_depth:g} nm', '#627f95'),
                        (f'Au {metal_depth:g} nm', '#98701c')]
        stack_title = '三维图：Au / SiO$_2$ / 四聚体（从下到上）'
    if legacy:
        stack_title = '三维图：旧版 Au / SiO$_2$ / 四聚体'
    layers += [(vertices, 0, geometry.h, color) for vertices, color in zip(geometry.vertices(), colors)]
    faces, facecolors = [], []
    for vertices, bottom, upper, color in layers:
        current = prism(vertices, bottom, upper)
        faces.extend(current)
        facecolors.extend([color] * len(current))
    # Sort all faces together so the substrate cannot hide a pillar above it.
    view.add_collection3d(Poly3DCollection(faces, facecolor=facecolors, zsort='average',
                                         edgecolor='#3e5264', linewidth=.5))
    view.set(xlim=(-p/2, p/2), ylim=(-p/2, p/2), zlim=(stack_bottom, geometry.h),
             xlabel='x (nm)', ylabel='y (nm)', zlabel='z (nm)')
    view.set_title(stack_title, pad=8)
    view.set_box_aspect((1, 1, (geometry.h-stack_bottom)/p))
    view.view_init(elev=23, azim=-55)
    view.tick_params(labelsize=8)
    view.text2D(-.25, .88, r'As$_2$Se$_3$ 四聚体', transform=view.transAxes,
                color='#287fa5', fontsize=9, zorder=100)
    for position, (label, color) in zip((.43, .22), layer_labels):
        view.text2D(-.25, position, label, transform=view.transAxes,
                    color=color, fontsize=9, zorder=100)
    if thick_substrate:
        view.plot([-p/2, p/2], [-p/2, -p/2], [stack_bottom, stack_bottom],
                  color='#385a72', linestyle='--', linewidth=1.5)

    spectrum = fig.add_subplot(grid[1, :])
    spectrum.plot(wavelength_nm, reflectance, color='#2767a3', lw=1.4,
                  label='输入目标反射谱')
    spectrum.set(xlim=(wavelength_nm[0], wavelength_nm[-1]),
                 xlabel='波长 (nm)', ylabel='反射率 R', title='输入光谱（该候选的光谱尚待回仿真）')
    spectrum.grid(alpha=.18)
    spectrum.legend(loc='lower right', fontsize=9)
    stack_note = ('基板绘图区底面是截示边界；SiO$_2$ 在仿真中继续向下延伸至 PML。'
                  if thick_substrate else
                  '旧版权重：Au 100 nm / SiO$_2$ 350 nm / 四聚体。' if legacy else
                  '基板按训练数据中的有限厚度绘制。')
    fig.text(.075, .144, stack_note, fontsize=9, color='#5b6978')
    fig.text(.075, .105,
             f'P = {geometry.p:.2f} nm    L = {geometry.l:.2f} nm    H = {geometry.h:.2f} nm'
             f'    dX = {geometry.alpha_x:.2f} nm    dY = {geometry.alpha_y:.2f} nm', fontsize=11)
    fig.text(.075, .066,
             f'θ奇 = {values["theta_odd_deg"]:.2f}°    θ偶 = {values["theta_even_deg"]:.2f}°'
             '    |    dX / dY 为旋转前边缘间距；正角为从 +z 看向平面的逆时针旋转。', fontsize=10)
    fig.text(.075, .026, r'候选已通过几何检查；未确认物理最优。两组颜色均代表 As$_2$Se$_3$，仅用于区分奇偶象限。',
             fontsize=9, color='#5b6978')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output.resolve()
