"""Load the local 2024 R1 lumapi and control real 3D FDTD sessions."""
# Python 控制真实求解器：连接会话 → 建立周期单元 → 求解 → 读取 R/T → 检查光谱。
# 本模块负责电磁仿真，不包含神经网络训练或人为生成的替代光谱。

from collections import OrderedDict
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

from .config import Geometry, SimulationSettings

LOG = logging.getLogger(__name__)


def mesh_grid_summary(grids, maximum_nm):
    """Check solver-generated coordinates, including every returned grid cell."""
    summary = {}
    for axis in ('x', 'y', 'z'):
        positions = np.asarray(grids[axis], dtype=float).reshape(-1)
        steps_nm = np.diff(positions) * 1e9
        if len(steps_nm) < 1 or not np.isfinite(positions).all() or np.any(steps_nm <= 0):
            raise ValueError(f'Invalid solver-generated {axis} mesh')
        largest = float(steps_nm.max())
        if largest > maximum_nm + 1e-6 or largest >= 10:
            raise ValueError(f'Actual {axis} mesh exceeds the strict limit: {largest:.9f} nm')
        summary[axis] = {'points': len(positions), 'maximum_step_nm': largest,
                         'minimum_step_nm': float(steps_nm.min())}
    return summary


def validate_power_integration(wavelength, reflection, transmission, check_wavelength, check_r, check_t):
    """Reject reduced spatial storage when full-spatial integrated powers disagree."""
    check_wavelength = np.asarray(check_wavelength).reshape(-1)
    check_r, check_t = np.asarray(check_r).reshape(-1), np.asarray(check_t).reshape(-1)
    if (len(check_wavelength) < 3 or check_r.shape != check_wavelength.shape or check_t.shape != check_wavelength.shape
            or not np.isfinite(np.column_stack([check_wavelength, check_r, check_t])).all()):
        raise ValueError('Invalid full-spatial power-check data')
    order = np.argsort(check_wavelength)
    check_wavelength, check_r, check_t = check_wavelength[order], check_r[order], check_t[order]
    if np.any(np.diff(check_wavelength) <= 0) or check_wavelength[0] < wavelength[0]-1e-5 or check_wavelength[-1] > wavelength[-1]+1e-5:
        raise ValueError('Power-check wavelengths are outside the main spectrum or duplicated')
    errors = {'maximum_R_error': float(np.max(np.abs(np.interp(check_wavelength, wavelength, reflection)-check_r))),
              'maximum_T_error': float(np.max(np.abs(np.interp(check_wavelength, wavelength, transmission)-check_t))),
              'check_points': len(check_wavelength), 'maximum_allowed_R_error': .002, 'maximum_allowed_T_error': 1e-5}
    if errors['maximum_R_error'] > .002 or errors['maximum_T_error'] > 1e-5:
        raise ValueError(f'Reduced monitor storage failed full-spatial power validation: {errors}')
    return errors, check_wavelength, check_r, check_t


def locate_lumapi(explicit=None):
    """Prefer the installed v241 API; never silently use a newer PyLumerical package."""
    # 优先使用配置/环境变量明确指定的本机 API，保持与已安装 2024 R1 的版本一致。
    requested = explicit or os.environ.get("LUMERICAL_LUMAPI_PATH")
    if requested:
        path = Path(requested).expanduser()
        if path.is_dir():
            path = path / "lumapi.py"
        if not path.is_file():
            raise FileNotFoundError(f"Configured lumapi.py does not exist: {path}")
        return path.resolve()
    roots = [Path(r"D:\Program Files\Lumerical\v241"),
             Path(r"C:\Program Files\Lumerical\v241"),
             Path(r"C:\Program Files\ANSYS Inc\v241\Lumerical"),
             Path(r"D:\Program Files\ANSYS Inc\v241\Lumerical")]
    for root in roots:
        path = root / "api/python/lumapi.py"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Lumerical v241 API not found. Set lumapi_path or LUMERICAL_LUMAPI_PATH.")


def load_lumapi(explicit=None):
    path = locate_lumapi(explicit)
    existing = sys.modules.get("lumapi")
    if existing is not None:
        # 同一个 Python 进程已加载另一套 API 时拒绝混用，避免版本与 DLL 不对应。
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("A different lumapi is already loaded; restart Python with the configured API")
        return existing
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lumapi", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lumapi"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("lumapi", None)
        raise
    return module


def validate_spectrum(wavelength_nm, reflectance, transmittance, settings, status):
    """Reject bad physics; do not hide sign errors or energy errors by clipping."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64).reshape(-1)
    reflectance = np.asarray(reflectance, dtype=np.float64).reshape(-1)
    transmittance = np.asarray(transmittance, dtype=np.float64).reshape(-1)
    grid = settings.wavelength_grid()
    if any(x.shape != grid.shape for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Monitor must return the configured number of wavelength/R/T points")
    if not all(np.isfinite(x).all() for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Nonfinite monitor values")
    # 按波长统一升序，同步排列 R 和 T，使网络第 i 个输入始终对应同一波长。
    order = np.argsort(wavelength_nm)
    wavelength_nm, reflectance, transmittance = [x[order] for x in (wavelength_nm, reflectance, transmittance)]
    if not np.allclose(wavelength_nm, grid, atol=1e-5, rtol=1e-8):
        raise ValueError("Monitor wavelength grid does not match the configured grid")
    if status not in (1, 2):
        raise ValueError(f"FDTD failed or diverged (status={status})")
    # status=1 表示到达时间上限；status=2 表示自动关断，严格采样要求后者。
    if settings.require_auto_shutoff and status != 2:
        raise ValueError("FDTD reached its time limit before auto-shutoff; increase simulation_time_ps")
    tol = settings.quality_tolerance
    # 保留真实数值并检查功率/能量范围；数值容差允许少量离散误差。
    # 这里不靠裁剪或逐条光谱 min-max 缩放来掩盖符号、归一化或收敛错误。
    if min(reflectance.min(), transmittance.min()) < -tol:
        raise ValueError("Negative R/T beyond tolerance; check propagation/monitor directions")
    if max(reflectance.max(), transmittance.max()) > 1 + tol:
        raise ValueError("R/T exceeds unity beyond tolerance; check convergence/source normalization")
    if np.max(reflectance + transmittance) > 1 + tol:
        raise ValueError("R+T exceeds unity beyond tolerance; check convergence and PML")
    # R、T 由源功率归一化；1-R-T 包含传播到 T 监视器之前的材料吸收损耗。
    return {"wavelength_nm": grid, "reflectance": reflectance,
            "transmittance": transmittance, "absorptance": 1 - reflectance - transmittance}


class FDTDSimulator:
    def __init__(self, settings: SimulationSettings, lumapi_path=None):
        self.settings = settings
        self.lumapi = load_lumapi(lumapi_path)
        self.fdtd = None

    def __enter__(self):
        # with 语句进入时创建 FDTD 会话，设置 CPU 资源并检查所需材料。
        self.fdtd = self.lumapi.FDTD(hide=self.settings.hide_fdtd)
        try:
            self.version = str(self.fdtd.version())
            self.fdtd.setresource("FDTD", 1, "processes", str(self.settings.processes))
            self.fdtd.setresource("FDTD", 1, "threads", str(self.settings.threads))
            s = self.settings
            if s.resonator_nk_csv:
                data = np.loadtxt(s.resonator_nk_csv, delimiter=",", skiprows=1)
                if (data.ndim != 2 or data.shape[1] != 3 or len(data) < 3
                        or not np.isfinite(data).all() or np.any(data[:, :2] <= 0)
                        or np.any(data[:, 2] < 0) or np.any(np.diff(data[:, 0]) <= 0)):
                    raise ValueError("nk CSV must contain increasing wavelength_nm,n,k; n>0,k>=0")
                extension = max(data[0, 0] - s.wavelength_start_nm,
                                s.wavelength_stop_nm - data[-1, 0], 0)
                if extension > s.material_max_extension_nm + 1e-8:
                    raise ValueError("Measured nk data does not cover the wavelength band")
                if extension:
                    LOG.warning("EXPLORATORY MATERIAL EXTENSION: holding endpoint nk over %.3f nm", extension)
                    if s.wavelength_stop_nm > data[-1, 0]:
                        data = np.vstack([data, [s.wavelength_stop_nm, *data[-1, 1:]]])
                    if s.wavelength_start_nm < data[0, 0]:
                        data = np.vstack([[s.wavelength_start_nm, *data[0, 1:]], data])
                # Use only the fitted source band plus neighboring measured knots.
                left = max(0, np.searchsorted(data[:, 0], s.wavelength_start_nm, side='left')-1)
                right = min(len(data), np.searchsorted(data[:, 0], s.wavelength_stop_nm, side='right')+1)
                data = data[left:right]
                name = s.resonator_material
                if not self.fdtd.materialexists(name):
                    material = self.fdtd.addmaterial("Sampled data")
                    self.fdtd.setmaterial(material, "name", name)
                sampled = np.column_stack([299792458 / (data[:, 0]*1e-9), (data[:, 1]+1j*data[:, 2])**2])
                self.fdtd.setmaterial(name, "max coefficients", 12)
                if s.material_fit_tolerance is not None:
                    self.fdtd.setmaterial(name, "tolerance", s.material_fit_tolerance)
                    self.fdtd.setmaterial(name, "imaginary weight", s.material_imaginary_weight)
                self.fdtd.setmaterial(name, "sampled data", sampled[::-1].copy())
                self.material_data = data
            for name in filter(None, (s.resonator_material or s.silicon_material, s.substrate_material, s.metal_material)):
                if not self.fdtd.materialexists(name):
                    raise ValueError(f"Material missing from installed FDTD library: {name}")
        except BaseException:
            self.fdtd.close()
            self.fdtd = None
            raise
        return self

    def __exit__(self, *args):
        # 正常结束或抛出异常都关闭会话，释放该 Python 进程持有的求解器资源。
        if self.fdtd is not None:
            self.fdtd.close()
            self.fdtd = None

    def build(self, geometry: Geometry, include_silicon=True, substrate_index=None):
        """Create four configured-material squares, with an optional finite spacer/metal stack."""
        f, s = self.fdtd, self.settings
        # 项目和标签采用 nm；Lumerical 的长度属性使用 SI 单位 m。
        nm = 1e-9
        p, h = geometry.p * nm, geometry.h * nm
        top, bottom = h + s.air_padding_nm * nm, -s.substrate_depth_nm * nm
        metal_enabled = s.metal_material is not None and substrate_index is None
        substrate_bottom = -s.substrate_depth_nm * nm
        metal_bottom = substrate_bottom - s.metal_thickness_nm * nm
        if metal_enabled:
            bottom = metal_bottom - s.bottom_padding_nm * nm
        # 每个样本重新构造几何，防止上一组结构或求解结果遗留到当前样本。
        f.switchtolayout()
        f.selectall()
        f.delete()
        # x/y 周期边界模拟无限阵列，z 方向 PML 吸收上下传播的出射波。
        f.addfdtd(properties=OrderedDict([
            ("dimension", "3D"), ("x", 0.), ("y", 0.),
            ("x span", p), ("y span", p), ("z min", bottom), ("z max", top),
            ("x min bc", "Periodic"), ("x max bc", "Periodic"),
            ("y min bc", "Periodic"), ("y max bc", "Periodic"),
            ("z min bc", "PML"), ("z max bc", "PML"),
            ("index", s.background_index), ("mesh accuracy", s.mesh_accuracy),
            ("simulation time", s.simulation_time_ps * 1e-12),
            ("auto shutoff min", s.auto_shutoff_min), ("pml profile", 3)
        ]))
        if s.maximum_mesh_nm is not None:
            # Cover the complete simulation volume, including substrate, metal,
            # air and PML boundaries. A local pillar override alone is insufficient.
            f.addmesh(name='global_mesh_limit', x=0., y=0., x_span=p, y_span=p,
                      z_min=bottom, z_max=top, override_x_mesh=True,
                      override_y_mesh=True, override_z_mesh=True,
                      dx=s.maximum_mesh_nm*nm, dy=s.maximum_mesh_nm*nm, dz=s.maximum_mesh_nm*nm)
        # 金属背板模式使用有限厚衬底；无背板时衬底延伸进入底部 PML。
        material = s.substrate_material if substrate_index is None else "<Object defined dielectric>"
        f.addrect(name="substrate", x=0., y=0., x_span=3*p, y_span=3*p,
                  z_min=substrate_bottom if metal_enabled else bottom - p, z_max=0., material=material)
        if substrate_index is not None:
            f.set("index", float(substrate_index))
        if metal_enabled:
            f.addrect(name="back_metal", x=0., y=0., x_span=3*p, y_span=3*p,
                      z_min=metal_bottom, z_max=substrate_bottom, material=s.metal_material)
            f.addmesh(name="metal_mesh", x=0., y=0., x_span=p, y_span=p,
                      z_min=metal_bottom-s.metal_mesh_nm*nm,
                      z_max=substrate_bottom+s.metal_mesh_nm*nm,
                      override_x_mesh=False, override_y_mesh=False, override_z_mesh=True,
                      dz=s.metal_mesh_nm*nm)
        if include_silicon:
            # 四个材料方柱共用边长 l 和高度 h；中心位置由名义间隙确定。
            for i, ((x, y), angle) in enumerate(zip(geometry.centers(), geometry.angles())):
                f.addrect(name=f"si_{i}", x=x*nm, y=y*nm,
                          x_span=geometry.l*nm, y_span=geometry.l*nm,
                          z_min=0., z_max=h, material=s.resonator_material or s.silicon_material)
                if angle != 0:
                    # Lumerical 的 rotation 1 采用顺时针度数；项目约定正角逆时针。
                    # 只旋转刚建立的方块，自身中心不移动，光源偏振和周期单元不旋转。
                    f.set("first axis", "z")
                    f.set("rotation 1", -float(angle))
        # 在结构及其上下邻近区域设置局部网格，控制几何和共振的离散精度。
        f.addmesh(name="structure_mesh", x=0., y=0., x_span=p, y_span=p,
                  z_min=-s.mesh_nm*nm, z_max=h + s.mesh_nm*nm,
                  dx=s.mesh_nm*nm, dy=s.mesh_nm*nm, dz=s.mesh_nm*nm)
        # 平面波由空气侧沿 -z 垂直入射，偏振角 0 对应本模型的 x 偏振。
        f.addplane(properties=OrderedDict([
            ("name", "source"), ("injection axis", "z"), ("direction", "Backward"),
            ("x", 0.), ("y", 0.), ("x span", p), ("y span", p),
            ("z", h + 0.5*s.air_padding_nm*nm), ("angle theta", 0.),
            ("polarization angle", s.polarization_deg), ("override global source settings", True),
            ("wavelength start", s.wavelength_start_nm*nm),
            ("wavelength stop", s.wavelength_stop_nm*nm)
        ]))
        # 用波长等间隔采样 1000 点，与训练所用 wavelength_grid() 对齐。
        f.setglobalmonitor("use source limits", True)
        f.setglobalmonitor("use wavelength spacing", True)
        f.setglobalmonitor("frequency points", s.spectrum_points)
        # R 位于光源上方；金属背板模式的 T 位于金属下方空气中。
        for name, z in (("R", h + 0.75*s.air_padding_nm*nm),
                        ("T", metal_bottom-0.5*s.bottom_padding_nm*nm if metal_enabled else -0.5*s.substrate_depth_nm*nm)):
            f.addpower(name=name, monitor_type="2D Z-normal", x=0., y=0.,
                       x_span=p, y_span=p, z=z)
            if s.research_wavelengths_nm:
                f.set('override global monitor settings', True)
                f.set('sample spacing', 'custom')
                f.set('custom frequency samples', (299792458/(s.wavelength_grid()*nm))[::-1])
            if s.monitor_downsample_xy > 1:
                f.set('down sample x', s.monitor_downsample_xy)
                f.set('down sample y', s.monitor_downsample_xy)
            # Only accumulated power is needed. Full 1000-frequency E/H fields are large.
            for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"):
                f.set("output " + field, False)
            f.set("output power", True)
            if s.power_validation_wavelengths_nm:
                f.addpower(name=name+'_powercheck', monitor_type='2D Z-normal', x=0., y=0., x_span=p, y_span=p, z=z)
                f.set('down sample x', 1)
                f.set('down sample y', 1)
                f.set('override global monitor settings', True)
                f.set('sample spacing', 'custom')
                f.set('custom frequency samples', (299792458/(np.asarray(s.power_validation_wavelengths_nm)*nm))[::-1])
                for field in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Px', 'Py', 'Pz'):
                    f.set('output '+field, False)
                f.set('output power', True)

    def run(self, geometry, project_path, include_silicon=True, substrate_index=None):
        self.build(geometry, include_silicon, substrate_index)
        project_path = Path(project_path).resolve()
        project_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_grids = {}
        mesh_summary = None
        if self.settings.maximum_mesh_nm is not None:
            mesh_grids = {axis: np.asarray(self.fdtd.getresult('FDTD', axis)).reshape(-1)
                          for axis in ('x', 'y', 'z')}
            mesh_summary = mesh_grid_summary(mesh_grids, self.settings.maximum_mesh_nm)
            if self.settings.monitor_downsample_xy > 1:
                bound = max(mesh_summary[axis]['maximum_step_nm'] for axis in ('x', 'y'))*self.settings.monitor_downsample_xy
                if bound >= 10:
                    raise ValueError('Monitor recording spacing must also remain strictly below 10 nm')
            project_path.with_suffix('.mesh.json').write_text(json.dumps({
                'maximum_mesh_nm': self.settings.maximum_mesh_nm, 'actual_grid': mesh_summary,
                'all_returned_cells_below_10_nm': True}, indent=2), encoding='utf-8')
        attempts = []
        elapsed = 0.0
        # 未自动关断时仅延长时间上限，最多四次；几何、网格与波长设置保持相同。
        for attempt in range(4):
            time_limit_ps = self.settings.simulation_time_ps * (2 ** attempt)
            if attempt:
                LOG.warning("Auto-shutoff not reached; retrying %s with %.1f ps time limit",
                            project_path.name, time_limit_ps)
                self.fdtd.switchtolayout()
                self.fdtd.setnamed("FDTD", "simulation time", time_limit_ps * 1e-12)
            self.fdtd.save(str(project_path))
            start = time.monotonic()
            self.fdtd.run()
            wall_seconds = time.monotonic() - start
            elapsed += wall_seconds
            status = int(np.asarray(self.fdtd.getresult("FDTD", "status")).item())
            attempts.append({"time_limit_ps": time_limit_ps, "status": status,
                             "elapsed_seconds": wall_seconds})
            if self.settings.save_projects:
                self.fdtd.save(str(project_path))
            if status != 1 or not self.settings.require_auto_shutoff:
                break
        # 两个监视器必须使用同一波长网格；读取的 "T" 字段代表归一化法向功率。
        r = self.fdtd.getresult("R", "T")
        t = self.fdtd.getresult("T", "T")
        if not np.allclose(np.asarray(r["lambda"]), np.asarray(t["lambda"]), rtol=1e-10, atol=1e-15):
            raise ValueError("R and T monitors returned different wavelength grids")
        # Downward incidence: reflected power flows +z; transmitted power flows -z.
        # 法向参考为 +z：反射功率为正，向 -z 传播的透射原始值为负，故只对 T 取负。
        result = validate_spectrum(np.asarray(r["lambda"])*1e9, r["T"], -np.asarray(t["T"]), self.settings, status)
        if self.settings.power_validation_wavelengths_nm:
            full_r, full_t = self.fdtd.getresult('R_powercheck', 'T'), self.fdtd.getresult('T_powercheck', 'T')
            if not np.allclose(full_r['lambda'], full_t['lambda'], rtol=1e-10, atol=1e-15):
                raise ValueError('Full-spatial R/T validation grids differ')
            errors, check_w, check_r, check_t = validate_power_integration(result['wavelength_nm'], result['reflectance'],
                result['transmittance'], np.asarray(full_r['lambda'])*1e9, full_r['T'], -np.asarray(full_t['T']))
            result.update(power_integration_validation_json=json.dumps(errors), power_check_wavelength_nm=check_w,
                          power_check_reflectance=check_r, power_check_transmittance=check_t)
        # 将最终时间上限、尝试次数和求解器版本一起保存，保留每条光谱的来源记录。
        result.update(status=status, elapsed_seconds=elapsed, fdtd_version=self.version,
                      simulation_time_ps=time_limit_ps, run_attempts=len(attempts),
                      attempts_json=json.dumps(attempts))
        if mesh_summary is not None:
            result.update({f'mesh_{axis}_nm': positions * 1e9 for axis, positions in mesh_grids.items()})
            result['mesh_summary_json'] = json.dumps(mesh_summary)
        return result
