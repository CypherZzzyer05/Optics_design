"""Run in PyCharm: browse real spectra/geometry and verify the Metaformer input."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from fdtd_pipeline.config import Geometry, PARAMETER_NAMES, PROJECT_ROOT


def verify_model_interface(dataset_path):
    import torch
    from torch.utils.data import DataLoader
    from metaformer import InverseMetaformer, MetaformerConfig, geometry_mse
    from spectrum_dataset import SpectrumGeometryDataset

    dataset = SpectrumGeometryDataset(dataset_path, split="all", allow_validation_data=True)
    if not len(dataset):
        raise ValueError("No successful simulations to check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    rotated = len(dataset.parameter_names) == 6
    model = InverseMetaformer(MetaformerConfig(
        geometry_dim=len(dataset.parameter_names),
        spectral_frontend="resnet1d" if len(dataset.parameter_names) > 5 else "linear",
        angle_symmetry="square_periodic" if len(dataset.parameter_names) == 7 else ("square_mirror" if rotated else "none"))).to(device)
    spectra, target = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)))
    prediction = model(spectra.to(device))
    loss = geometry_mse(prediction, target.to(device), model.config.angle_symmetry)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Model loss/gradients are not finite")
    report = {"device": str(device), "samples": len(dataset),
              "input_shape": list(spectra.shape), "label_shape": list(target.shape),
              "prediction_shape": list(prediction.shape), "initial_untrained_mse": loss.item(),
              "finite_gradients": True, "training_performed": False}
    print(json.dumps(report, indent=2))
    (Path(dataset_path).parent / "model_interface_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_dataset_figure(dataset_path):
    """Build the browser and retain its widgets until the window closes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib.widgets import Slider, Button

    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    n = len(data["spectra"])
    if not n:
        raise ValueError("No successful samples; check manifest.json/errors.jsonl")
    if len(data["parameter_names"]) == 7:
        return build_quadrant_figure(data)
    metadata = json.loads(str(data["metadata_json"].item()))
    names = tuple(str(name) for name in data["parameter_names"])
    units = tuple(str(unit) for unit in data["parameter_units"]) if len(names) == 6 else ("nm",) * 5
    values = data.get("geometry_values", data["geometry_nm"])
    rotated = len(names) == 6
    state = {"index": 0, "angle_deg": float(data["rotation_deg"][0]) if rotated else 0.0,
             "geometry_valid": True, "preview_only": False}
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig = plt.figure(figsize=(13.5, 8), facecolor="#f5f7fb")
    fig.canvas.manager.set_window_title("MetaFormer | Lumerical FDTD 数据集")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.65, 1), height_ratios=(1.25, 1),
                           left=.07, right=.96, top=.86, bottom=.25 if rotated else .17,
                           hspace=.45, wspace=.32)
    spectral_ax = fig.add_subplot(grid[0, 0])
    cell_ax = fig.add_subplot(grid[0, 1])
    label_ax = fig.add_subplot(grid[1, 0])
    info_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle("Python → Lumerical FDTD → 光谱数据集 → MetaFormer", x=.07, ha="left", y=.965,
                 fontsize=18, fontweight="bold", color="#152849")
    level = metadata["config"]["quality_level"]
    fig.text(.07, .9, f"真实三维 FDTD · {n} 个有效样本 · 1000 点统一波长网格 · {level}", color="#50617c")
    curves = [spectral_ax.plot(data["wavelength_nm"]/1000, values[0], lw=1.7, color=color, label=label)[0]
              for values, color, label in ((data["spectra"], "#2667d8", "反射 R"),
                                           (data["transmittance"], "#049b86", "透射 T"),
                                           (data["absorptance"], "#d58b23", "1 − R − T"))]
    spectral_ax.set(xlabel="波长 (μm)", ylabel="归一化功率", ylim=(-.05, 1.05))
    spectral_ax.grid(alpha=.2)
    spectral_ax.legend(loc="upper right", ncol=3, fontsize=9)
    bars = label_ax.bar(names, data["geometry_01"][0], color="#748bc1", width=.5)
    label_ax.set(ylim=(0, 1.15), ylabel="标签 [0, 1]", title="该样本的真实训练标签及物理尺寸")
    label_ax.grid(axis="y", alpha=.2)
    physical_labels = [label_ax.text(i, .02, "", ha="center", fontsize=9) for i in range(len(names))]
    info_ax.axis("off")
    counts = {s: len(data[s+"_indices"]) for s in ("train", "val", "test")}
    info = (f"数据集接口\n\n光谱: ({n}, 1000)   float32\n标签: ({n}, {len(names)})   float32\n"
            f"训练 / 验证 / 测试: {counts['train']} / {counts['val']} / {counts['test']}\n"
            f"标签: p, l, h, αx, αy" + (", θ (deg)\n各方块绕自身中心共同旋转\n" if len(names) == 6 else "\n") +
            "\n参数范围: 工程示例，Table S3 尚未核对\n"
            "材料: Si / SiO2，Lumerical Palik 库\n"
            "采样位置的 1−R−T 包含材料吸收。\n\n"
            "拖动下方“样本”切换已有仿真。\n" +
            ("“旋转角”只预览结构，不重新计算光谱。\n" if rotated else "") +
            ("旋转试运行结果请运行 16 查看。" if len(names) == 6 else "训练结果请运行 07 查看。"))
    info_ax.text(0, 1, info, va="top", color="#334969", linespacing=1.3 if rotated else 1.5, fontsize=9.5)
    status = fig.text(.07, .025, "", fontsize=9, color="#61718a")

    def render():
        i = state["index"]
        sample_angle = float(data["rotation_deg"][i]) if rotated else 0.0
        angle = state["angle_deg"]
        preview_only = rotated and not np.isclose(angle, sample_angle, rtol=0, atol=1e-8)
        base = Geometry(*data["geometry_nm"][i])
        try:
            geometry = replace(base, rotation_deg=angle)
            vertices = geometry.vertices()
            valid = True
        except ValueError:
            # 不合法角度也可以画出来，帮助观察相交；它不能作为采样或仿真结构。
            radians = np.deg2rad(angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
            vertices = [(corners - center) @ rotation.T + center
                        for corners, center in zip(base.vertices(), base.centers())]
            valid = False
        state.update(geometry_valid=valid, preview_only=preview_only)
        for line, key in zip(curves, ("spectra", "transmittance", "absorptance")):
            line.set_ydata(data[key][i])
        spectral_ax.set_title(f"真实光谱 · 样本 {int(data['sample_ids'][i]):06d}  |  {data['split_names'][i]}"
                             + (f"  |  采样 θ = {sample_angle:.2f}°" if rotated else f"  |  收敛状态 {int(data['status'][i])}"),
                             fontsize=11)
        cell_ax.clear()
        cell_ax.add_patch(Rectangle((-base.p/2, -base.p/2), base.p, base.p, facecolor="#e7edf6", edgecolor="#607597", ls="--"))
        for corners in vertices:
            cell_ax.add_patch(Polygon(corners, facecolor="#315f9c" if valid else "#d96779", edgecolor="#17395f"))
        if rotated:
            centers = np.asarray(base.centers())
            cell_ax.scatter(centers[:, 0], centers[:, 1], color="#e9a32a", s=12)
        cell_ax.set(xlim=(-base.p*.55, base.p*.55), ylim=(-base.p*.55, base.p*.55),
                    xlabel="x (nm)", ylabel="y (nm)", aspect="equal")
        cell_ax.set_title((f"结构预览 | θ = {angle:.2f}°" if rotated else f"单元顶视图 | h = {base.h:.1f} nm")
                          + ("（相交或越界）" if not valid else "（未仿真）" if preview_only else ""), fontsize=11)
        for bar, label, value01, value, unit in zip(bars, physical_labels, data["geometry_01"][i], values[i], units):
            bar.set_height(value01)
            label.set_position((bar.get_x()+bar.get_width()/2, value01+.045))
            label.set_text(f"{value:.1f} {'°' if unit == 'deg' else unit}")
        if preview_only:
            status.set_text(f"旋转预览 {angle:.2f}°：" + ("方块相交或越界，不能采样。" if not valid else "新角度尚未仿真。")
                            + f" 光谱和训练标签仍对应采样角 {sample_angle:.2f}°；点击“还原角度”恢复该样本。")
            status.set_color("#a84252" if not valid else "#9d631c")
        else:
            status.set_text("拖动“样本”切换真实仿真；“旋转角”预览四块共同旋转，黄色点为固定中心。"
                            if rotated else "拖动“样本”切换真实光谱和结构。")
            status.set_color("#61718a")
        fig.canvas.draw_idle()

    def update_sample(index):
        state["index"] = int(index)
        state["angle_deg"] = float(data["rotation_deg"][int(index)]) if rotated else 0.0
        if rotated:
            # 换样本时还原其真实角度；暂停回调以免渲染旧样本和新角度的组合。
            angle_slider.eventson = False
            angle_slider.set_val(state["angle_deg"])
            angle_slider.eventson = True
        render()

    def update_angle(angle):
        state["angle_deg"] = float(angle)
        render()

    render()
    if n > 1:
        slider_ax = fig.add_axes((.16, .14 if rotated else .065, .62 if rotated else .7, .025), facecolor="#e7edf6")
        slider = Slider(slider_ax, "样本", 0, n-1, valinit=0, valstep=1)
        slider.on_changed(update_sample)
        fig._sample_slider = slider
    if rotated:
        angle_ax = fig.add_axes((.16, .09, .62, .025), facecolor="#e7edf6")
        angle_slider = Slider(angle_ax, "旋转角 (°)", -45, 45, valinit=state["angle_deg"], valfmt="%1.2f°")
        angle_slider.on_changed(update_angle)
        button = Button(fig.add_axes((.84, .085, .11, .04)), "还原角度", color="#e7edf6", hovercolor="#d7e4f7")
        button.on_clicked(lambda event: update_sample(state["index"]))
        fig._angle_slider, fig._reset_button = angle_slider, button
    fig._preview_state = state
    return fig


def build_quadrant_figure(data):
    """Display the actual two-angle geometry and spectra of each saved sample."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.widgets import Slider
    values, names = data['geometry_values'], list(data['parameter_names'])
    metadata=json.loads(str(data['metadata_json'].item()))
    settings=metadata['config']['simulation']
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    fig.subplots_adjust(bottom=.17,hspace=.4,wspace=.3)
    spectral,cell,labels,info=axes.flat
    state={'index':0}
    curves=[spectral.plot(data['wavelength_nm'],data[key][0],label=name)[0]
            for key,name in (('spectra','R'),('transmittance','T'),('absorptance','A'))]
    spectral.set(xlabel='Wavelength (nm)',ylabel='Power',ylim=(-.05,1.05))
    spectral.legend();spectral.grid(alpha=.2)
    bars=labels.bar(['P','L','H','dX','dY','odd','even'],data['geometry_01'][0])
    labels.set(ylim=(0,1.1),ylabel='Normalized parameter')
    info.axis('off')
    information=info.text(0,1,'',va='top',linespacing=1.6)
    def render(index):
        i=int(index);state['index']=i
        g=Geometry(*values[i,:5],theta_odd_deg=values[i,5],theta_even_deg=values[i,6])
        for curve,key in zip(curves,('spectra','transmittance','absorptance')):
            curve.set_ydata(data[key][i])
        spectral.set_title(f"Real FDTD sample {int(data['sample_ids'][i])}")
        cell.clear()
        cell.add_patch(Rectangle((-g.p/2,-g.p/2),g.p,g.p,fill=False,ls='--'))
        for j,polygon in enumerate(g.vertices()):
            cell.add_patch(Polygon(polygon,facecolor='#7359aa' if j in (0,3) else '#278794'))
        cell.set(xlim=(-g.p*.55,g.p*.55),ylim=(-g.p*.55,g.p*.55),aspect='equal',xlabel='x (nm)',ylabel='y (nm)',
                 title=f'Odd {g.theta_odd_deg:+.1f} deg / even {g.theta_even_deg:+.1f} deg')
        for bar,height in zip(bars,data['geometry_01'][i]):bar.set_height(height)
        text='\n'.join(f'{name}: {value:.2f} {"nm" if j<5 else "deg"}' for j,(name,value) in enumerate(zip(names,values[i])))
        information.set_text(f"{settings.get('resonator_material','As2Se3')}\n{settings.get('metal_material')}\n\n{text}")
        fig.canvas.draw_idle()
    render(0)
    if len(values)>1:
        slider=Slider(fig.add_axes((.15,.065,.7,.025)),'Sample',0,len(values)-1,valinit=0,valstep=1)
        slider.on_changed(render);fig._sample_slider=slider
    fig._preview_state=state
    return fig


def plot_dataset(dataset_path, show=True):
    import matplotlib
    # PyCharm 的图片编辑器/内嵌静态图不运行控件。使用独立 Tk 窗口和阻塞事件循环。
    matplotlib.use("TkAgg" if show else "Agg", force=True)
    import matplotlib.pyplot as plt

    fig = build_dataset_figure(dataset_path)
    output = Path(dataset_path).parent / "dataset_preview.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    print(f"Dataset preview: {output}")
    if show:
        print("Interactive window: drag the sample slider or the rotation-angle slider; PNG files are static.", flush=True)
        plt.show(block=True)
    else:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/as2se3_au_pilot/dataset.npz")
    parser.add_argument("--no-show", action="store_true", help="Save preview without opening a window")
    parser.add_argument("--check-model", action="store_true", help="Verify real spectra through forward/backward; no training")
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"Dataset missing: {args.dataset}. Run generate_dataset.py generate first.")
    if args.check_model:
        verify_model_interface(args.dataset)
    plot_dataset(args.dataset, show=not args.no_show)


if __name__ == "__main__":
    main()
