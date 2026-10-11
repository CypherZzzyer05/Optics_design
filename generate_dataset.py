"""PyCharm/CLI entry point: check installation, inspect a plan, run and resume FDTD."""
# 仿真入口提供四种任务：doctor 检查连接，plan 查看参数计划，
# reference 做已知 Fresnel 对照，generate 按固定计划采样并生成数据集。
# 此脚本生成数据；模型训练另由 train_metaformer.py 启动。

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys

import numpy as np

from fdtd_pipeline.config import PROJECT_ROOT, load_config
from fdtd_pipeline.lumerical import FDTDSimulator, locate_lumapi
from fdtd_pipeline.pipeline import atomic_json, atomic_npz, generate_dataset


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])


def check_installation(config):
    # 这里只连接 API、设置资源和访问材料；真正的求解许可需通过 reference/generate 检查。
    api = locate_lumapi(config.lumapi_path)
    executable = api.parents[2] / "bin/fdtd-solutions.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"FDTD executable missing: {executable}")
    print(f"Python: {sys.executable}\nLocal lumapi: {api}\nFDTD executable: {executable}", flush=True)
    with FDTDSimulator(config.simulation, config.lumapi_path) as simulator:
        print(f"FDTD API connection OK; solver version: {simulator.version}")
        print(f"CPU resource: {config.simulation.processes} process(es), {config.simulation.threads} threads")
        print(f"Materials: {config.simulation.resonator_material or config.simulation.silicon_material}; {config.simulation.substrate_material}; metal={config.simulation.metal_material}")
    print("Session creation and material access verified. Run reference/generate to verify the solving license.")


def run_reference(config):
    """Lossless air/dielectric Fresnel test checks both power signs and normalization."""
    # 用无硅结构的无损介质界面做已知答案对照，检查 R/T 功率符号和归一化。
    # 对照采用较细网格和更严格关断；它不替代实际共振结构的网格收敛检查。
    settings = replace(config.simulation, save_projects=True, require_auto_shutoff=True,
                       mesh_accuracy=max(4, config.simulation.mesh_accuracy),
                       mesh_nm=min(40, config.simulation.mesh_nm),
                       auto_shutoff_min=min(1e-6, config.simulation.auto_shutoff_min),
                       simulation_time_ps=max(20, config.simulation.simulation_time_ps))
    geometry = config.sample_geometries()[0]
    directory = PROJECT_ROOT / "data/reference"
    directory.mkdir(parents=True, exist_ok=True)
    n0, n1 = settings.background_index, 1.45
    # 垂直入射时 Fresnel 反射率；该无损对照的透射率为 1-R。
    expected_r = ((n0 - n1) / (n0 + n1)) ** 2
    with FDTDSimulator(settings, config.lumapi_path) as simulator:
        result = simulator.run(geometry, directory / "fresnel.fsp", include_silicon=False, substrate_index=n1)
    errors = {
        "max_reflectance_error": float(np.max(np.abs(result["reflectance"] - expected_r))),
        "max_transmittance_error": float(np.max(np.abs(result["transmittance"] - (1-expected_r)))),
        "max_energy_error": float(np.max(np.abs(result["reflectance"] + result["transmittance"] - 1))),
    }
    passed = max(errors.values()) < 0.01
    # 保存数值光谱及对照误差报告；超出容差时抛出异常，便于在采样前排查。
    summary = {"passed": passed, "expected_reflectance": expected_r,
               "expected_transmittance": 1-expected_r, "tolerance": 0.01,
               "status": result["status"], "fdtd_version": result["fdtd_version"], **errors}
    summary["simulation_settings"] = {"mesh_accuracy": settings.mesh_accuracy,
                                      "mesh_nm": settings.mesh_nm,
                                      "air_padding_nm": settings.air_padding_nm,
                                      "substrate_depth_nm": settings.substrate_depth_nm,
                                      "auto_shutoff_min": settings.auto_shutoff_min}
    atomic_npz(directory / "fresnel.npz", **result)
    atomic_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Fresnel validation failed; inspect reference project before collecting data")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "plan", "reference", "generate"), nargs="?", default="generate")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/as2se3_au_pilot.json")
    parser.add_argument("--limit", type=int, help="Run at most this many missing/failed samples; resumes the same plan")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    # 默认读取 smoke.json；sample_count、几何范围、波长网格及求解参数均来自 JSON。
    config = load_config(args.config)
    # Set up disk logging after the manifest has been created (empty-directory guard).
    setup_logging()
    if args.command == "doctor":
        check_installation(config)
    elif args.command == "plan":
        print(json.dumps(config.metadata(), indent=2, ensure_ascii=False))
        print(f"First cells: {config.parameter_names}; units: {config.parameter_units}")
        for g in config.sample_geometries()[:5]:
            print(np.round(config.geometry_values(g), 3))
    elif args.command == "reference":
        run_reference(config)
    else:
        # 同一配置重复运行会校验并跳过成功样本，只补算缺失、失败或损坏的项目。
        summary = generate_dataset(config, limit=args.limit, fail_fast=args.fail_fast)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if summary["failed"]:
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # 用户中断时保留已完成结果；下次用相同配置即可继续固定采样计划。
        print("Interrupted. Completed samples are preserved; run the same configuration to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        logging.exception("FDTD dataset task failed: %s", error)
        raise SystemExit(1)
