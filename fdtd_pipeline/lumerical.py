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
        raise ValueError("Monitor must return exactly 1000 wavelength/R/T points")
    if not all(np.isfinite(x).all() for x in (wavelength_nm, reflectance, transmittance)):
        raise ValueError("Nonfinite monitor values")
    # 按波长统一升序，同步排列 R 和 T，使网络第 i 个输入始终对应同一波长。
    order = np.argsort(wavelength_nm)
    wavelength_nm, reflectance, transmittance = [x[order] for x in (wavelength_nm, reflectance, transmittance)]
    if not np.allclose(wavelength_nm, grid, atol=1e-5, rtol=1e-8):
        raise ValueError("Monitor wavelength grid does not match the configured uniform grid")
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
            for name in (self.settings.silicon_material, self.settings.substrate_material):
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
        """Create four Si squares on a semi-infinite substrate, with air-side incidence."""
        f, s = self.fdtd, self.settings
        # 项目和标签采用 nm；Lumerical 的长度属性使用 SI 单位 m。
        nm = 1e-9
        p, h = geometry.p * nm, geometry.h * nm
        top, bottom = h + s.air_padding_nm * nm, -s.substrate_depth_nm * nm
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
        # 衬底向下延伸进入底部 PML；Fresnel 对照时可换成指定折射率介质。
        material = s.substrate_material if substrate_index is None else "<Object defined dielectric>"
        f.addrect(name="substrate", x=0., y=0., x_span=3*p, y_span=3*p,
                  z_min=bottom - p, z_max=0., material=material)
        if substrate_index is not None:
            f.set("index", float(substrate_index))
        if include_silicon:
            # 四个硅方块共用边长 l 和高度 h；中心位置由 alpha_x/alpha_y 间隙决定。
            for i, (x, y) in enumerate(geometry.centers()):
                f.addrect(name=f"si_{i}", x=x*nm, y=y*nm,
                          x_span=geometry.l*nm, y_span=geometry.l*nm,
                          z_min=0., z_max=h, material=s.silicon_material)
                if geometry.rotation_deg != 0:
                    # Lumerical 的 rotation 1 采用顺时针度数；项目约定正角逆时针。
                    # 只旋转刚建立的方块，自身中心不移动，光源偏振和周期单元不旋转。
                    f.set("first axis", "z")
                    f.set("rotation 1", -float(geometry.rotation_deg))
        # 在结构及其上下邻近区域设置局部网格，控制几何和共振的离散精度。
        f.addmesh(name="structure_mesh", x=0., y=0., x_span=p, y_span=p,
                  z_min=-s.mesh_nm*nm, z_max=h + s.mesh_nm*nm,
                  dx=s.mesh_nm*nm, dy=s.mesh_nm*nm, dz=s.mesh_nm*nm)
        # 平面波由空气侧沿 -z 垂直入射，偏振角 0 对应本模型的 x 偏振。
        f.addplane(properties=OrderedDict([
            ("name", "source"), ("injection axis", "z"), ("direction", "Backward"),
            ("x", 0.), ("y", 0.), ("x span", p), ("y span", p),
            ("z", h + 0.5*s.air_padding_nm*nm), ("angle theta", 0.),
            ("polarization angle", 0.), ("override global source settings", True),
            ("wavelength start", s.wavelength_start_nm*nm),
            ("wavelength stop", s.wavelength_stop_nm*nm)
        ]))
        # 用波长等间隔采样 1000 点，与训练所用 wavelength_grid() 对齐。
        f.setglobalmonitor("use source limits", True)
        f.setglobalmonitor("use wavelength spacing", True)
        f.setglobalmonitor("frequency points", s.spectrum_points)
        # R 在光源上方接收反射波；T 在衬底内部记录向下传播的透射功率。
        for name, z in (("R", h + 0.75*s.air_padding_nm*nm),
                        ("T", -0.5*s.substrate_depth_nm*nm)):
            f.addpower(name=name, monitor_type="2D Z-normal", x=0., y=0.,
                       x_span=p, y_span=p, z=z)
            # Only accumulated power is needed. Full 1000-frequency E/H fields are large.
            for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"):
                f.set("output " + field, False)
            f.set("output power", True)

    def run(self, geometry, project_path, include_silicon=True, substrate_index=None):
        self.build(geometry, include_silicon, substrate_index)
        project_path = Path(project_path).resolve()
        project_path.parent.mkdir(parents=True, exist_ok=True)
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
        # 将最终时间上限、尝试次数和求解器版本一起保存，保留每条光谱的来源记录。
        result.update(status=status, elapsed_seconds=elapsed, fdtd_version=self.version,
                      simulation_time_ps=time_limit_ps, run_attempts=len(attempts),
                      attempts_json=json.dumps(attempts))
        return result
