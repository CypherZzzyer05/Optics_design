"""Validated settings: lengths/wavelengths in nm, optional common rotation in degrees."""
# 仿真配置与几何定义：长度用 nm，在 Lumerical 接口处转换成 m；角度用 deg。
# 五个长度参数顺序固定，旋转任务在末尾追加第六个角度参数。

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np

PARAMETER_NAMES = ("p", "l", "h", "alpha_x", "alpha_y")
ROTATION_PARAMETER_NAMES = (*PARAMETER_NAMES, "rotation_deg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Geometry:
    p: float
    l: float
    h: float
    alpha_x: float
    alpha_y: float
    rotation_deg: float = 0.0  # 各方块绕自身中心、共同绕 z 轴旋转；项目正角采用逆时针。

    def __post_init__(self):
        values = self.array()
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("All geometry dimensions must be finite and positive (nm)")
        # 一个周期单元内沿每个方向放两个正方形；边长与间隙之和不能超出周期。
        if 2 * self.l + max(self.alpha_x, self.alpha_y) >= self.p:
            raise ValueError("Four squares must fit inside a cell: 2*l + gap < p")
        if not np.isfinite(self.rotation_deg):
            raise ValueError("rotation_deg must be finite")
        vertices = self.vertices()
        if any(np.max(np.abs(v)) >= self.p / 2 for v in vertices):
            raise ValueError("Rotated squares must stay inside the periodic cell")
        # 所有方块使用同一角度，变换回共同局部坐标后用分离轴检查相交/接触。
        centers = np.asarray(self.centers()) @ self.rotation_matrix()
        for i in range(len(centers)):
            for j in range(i):
                if np.all(np.abs(centers[i] - centers[j]) <= self.l + 1e-8):
                    raise ValueError("Rotated squares overlap or touch; change gaps or rotation")

    def array(self, include_rotation=False):
        names = ROTATION_PARAMETER_NAMES if include_rotation else PARAMETER_NAMES
        return np.array([getattr(self, name) for name in names], dtype=np.float64)

    def rotation_matrix(self):
        angle = np.deg2rad(self.rotation_deg)
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def vertices(self):
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.l / 2
        rotated = corners @ self.rotation_matrix().T
        return [rotated + center for center in self.centers()]

    def centers(self):
        """The alpha parameters denote edge-to-edge gaps, not center distances."""
        # 启用旋转后 alpha 仍描述未旋转排布的边缘间隙；中心保持不变，实际最小间隙会变化。
        # 相邻方块中心距离为 l + alpha；以原点为中心对称放置四个方块。
        x = (self.l + self.alpha_x) / 2
        y = (self.l + self.alpha_y) / 2
        return [(-x, -y), (-x, y), (x, -y), (x, y)]


@dataclass(frozen=True)
class SimulationSettings:
    # 以下为类默认值；实际采样时由 configs/smoke.json 等 JSON 文件覆盖。
    wavelength_start_nm: float = 4000
    wavelength_stop_nm: float = 8000
    spectrum_points: int = 1000
    mesh_accuracy: int = 2
    mesh_nm: float = 80
    simulation_time_ps: float = 3
    auto_shutoff_min: float = 1e-4
    air_padding_nm: float = 4000
    substrate_depth_nm: float = 3000
    background_index: float = 1
    silicon_material: str = "Si (Silicon) - Palik"
    substrate_material: str = "SiO2 (Glass) - Palik"
    processes: int = 1
    threads: int = 4
    hide_fdtd: bool = True
    save_projects: bool = True
    quality_tolerance: float = 0.03
    require_auto_shutoff: bool = True

    def __post_init__(self):
        positive = (self.wavelength_start_nm, self.wavelength_stop_nm, self.mesh_nm,
                    self.simulation_time_ps, self.air_padding_nm,
                    self.substrate_depth_nm, self.background_index)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Simulation lengths, time and refractive index must be positive")
        if self.wavelength_stop_nm <= self.wavelength_start_nm:
            raise ValueError("wavelength_stop_nm must exceed wavelength_start_nm")
        if self.spectrum_points != 1000:
            raise ValueError("The existing Metaformer requires exactly 1000 spectral points")
        if not 1 <= self.mesh_accuracy <= 8 or int(self.mesh_accuracy) != self.mesh_accuracy:
            raise ValueError("mesh_accuracy must be an integer in [1, 8]")
        if not 0 < self.auto_shutoff_min < 1 or not 0 < self.quality_tolerance < 0.2:
            raise ValueError("Invalid shutoff threshold or quality tolerance")
        for v in (self.processes, self.threads):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError("processes and threads must be positive integers")
        for v in (self.hide_fdtd, self.save_projects, self.require_auto_shutoff):
            if not isinstance(v, bool):
                raise ValueError("Boolean settings must use JSON true/false")

    def wavelength_grid(self):
        # 1000 点按波长均匀排列，包含两端；神经网络的位置编码依赖固定采样顺序。
        return np.linspace(self.wavelength_start_nm, self.wavelength_stop_nm, self.spectrum_points)


@dataclass(frozen=True)
class PipelineConfig:
    simulation: SimulationSettings
    bounds_nm: dict
    sample_count: int
    seed: int
    output_dir: Path
    lumapi_path: str | None = None
    parameter_range_source: str = "Engineering example; not verified against Table S3"
    quality_level: str = "pipeline_validation"
    rotation_bounds_deg: tuple[float, float] | None = None

    def __post_init__(self):
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if set(self.bounds_nm) != set(PARAMETER_NAMES):
            raise ValueError(f"bounds_nm must contain exactly {PARAMETER_NAMES}")
        for name, pair in self.bounds_nm.items():
            if len(pair) != 2 or not np.isfinite(pair).all() or not 0 < pair[0] < pair[1]:
                raise ValueError(f"Invalid lower/upper bounds for {name}")
        if self.rotation_bounds_deg is not None:
            pair = self.rotation_bounds_deg
            if len(pair) != 2 or not np.isfinite(pair).all() or list(pair) != [-45, 45]:
                raise ValueError("Common square rotation currently uses [-45,45] degrees")

    @property
    def parameter_names(self):
        return ROTATION_PARAMETER_NAMES if self.rotation_bounds_deg is not None else PARAMETER_NAMES

    @property
    def parameter_units(self):
        return ("nm",) * 5 + (("deg",) if self.rotation_bounds_deg is not None else ())

    def geometry_values(self, geometry):
        return geometry.array(include_rotation=self.rotation_bounds_deg is not None)

    @property
    def lower(self):
        # 按 PARAMETER_NAMES 排序，不能依赖 JSON 字典键的偶然排列顺序。
        values = [self.bounds_nm[k][0] for k in PARAMETER_NAMES]
        if self.rotation_bounds_deg is not None:
            values.append(self.rotation_bounds_deg[0])
        return np.array(values, dtype=np.float64)

    @property
    def upper(self):
        values = [self.bounds_nm[k][1] for k in PARAMETER_NAMES]
        if self.rotation_bounds_deg is not None:
            values.append(self.rotation_bounds_deg[1])
        return np.array(values, dtype=np.float64)

    def normalize(self, geometry_nm):
        # 长度用 nm，启用旋转时第六列用 deg；各列使用自己的范围归一化。
        return (np.asarray(geometry_nm) - self.lower) / (self.upper - self.lower)

    def sample_geometries(self):
        """Rejection sampling with a fixed seed; one plan is persisted before running."""
        # 在参数范围内均匀采样，丢弃放不进周期单元的组合；固定种子便于复查计划。
        rng = np.random.default_rng(self.seed)
        result = []
        for _ in range(max(10000, self.sample_count * 1000)):
            try:
                geometry = Geometry(*rng.uniform(self.lower, self.upper))
            except ValueError:
                continue
            result.append(geometry)
            if len(result) == self.sample_count:
                return result
        raise ValueError("Could not sample valid cells; check gaps, cell size and rotated-square overlap")

    def metadata(self):
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        # 未启用旋转时保持旧配置序列化形式，已有五参数 manifest 指纹仍可复用。
        if self.rotation_bounds_deg is None:
            data.pop("rotation_bounds_deg")
        return data


def load_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    known = {f.name for f in fields(PipelineConfig)}
    if set(data) - known:
        raise ValueError(f"Unknown config keys: {sorted(set(data) - known)}")
    # 将仿真子配置转换为经过校验的对象，输出路径相对于 JSON 所在目录解析。
    data["simulation"] = SimulationSettings(**data.get("simulation", {}))
    data["output_dir"] = (path.parent / data.get("output_dir", "../data/fdtd")).resolve()
    return PipelineConfig(**data)
