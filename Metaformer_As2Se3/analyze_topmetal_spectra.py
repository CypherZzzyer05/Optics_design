"""Describe existing top-metal FDTD spectra without fitting or running a solver."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from explore_double_dips import double_pair


ROOT = Path(__file__).resolve().parent
CRITERIA = {
    "maximum_dip_reflectance": 0.2,
    "minimum_prominence": 0.4,
    "minimum_width_samples": 5,
    "minimum_separation_nm": 30,
    "band_edge_exclusion_nm": 30,
    "minimum_intervening_contrast": 0.2,
}


def all_strict_pairs(wavelength, reflectance, dips):
    """Enumerate every pair using the unchanged double_pair acceptance rules."""
    eligible = [d for d in dips if d["resolved"] and d["reflectance_min"] <= 0.2
                and d["prominence"] >= 0.4
                and wavelength[0] + 30 < d["wavelength_nm"] < wavelength[-1] - 30]
    pairs = []
    for a, b in itertools.combinations(eligible, 2):
        left, right = sorted((a, b), key=lambda d: d["wavelength_nm"])
        separation = right["wavelength_nm"] - left["wavelength_nm"]
        if separation < 30:
            continue
        between = (wavelength > left["wavelength_nm"]) & (wavelength < right["wavelength_nm"])
        ridge = float(np.max(reflectance[between]))
        if ridge - max(left["reflectance_min"], right["reflectance_min"]) < 0.2:
            continue
        pairs.append({"dips": [left, right], "separation_nm": separation,
                      "intervening_max_reflectance": ridge,
                      "score": min(left["prominence"], right["prominence"])})
    return pairs


def coverage(rows):
    ordinary = [dip for row in rows for dip in row["ordinary_resolved_dips"]]
    strict = {(row["sample_id"], dip["wavelength_nm"]): dip
              for row in rows for pair in row["strict_pairs"] for dip in pair["dips"]}
    bins = [700, 800, 1000, 1300, 1700]
    return {
        "samples": len(rows),
        "samples_with_strict_pair": sum(bool(row["strict_pairs"]) for row in rows),
        "samples_with_strict_pair_at_least_one_dip_above_1000_nm": sum(
            any(any(d["wavelength_nm"] > 1000 for d in p["dips"]) for p in row["strict_pairs"])
            for row in rows),
        "samples_with_strict_pair_both_dips_above_1000_nm": sum(
            any(all(d["wavelength_nm"] > 1000 for d in p["dips"]) for p in row["strict_pairs"])
            for row in rows),
        "total_qualifying_strict_pairs": sum(len(row["strict_pairs"]) for row in rows),
        "unique_dips_participating_in_strict_pairs": len(strict),
        "ordinary_resolved_dip_count": len(ordinary),
        "samples_with_any_ordinary_resolved_dip": sum(bool(row["ordinary_resolved_dips"]) for row in rows),
        "samples_with_ordinary_resolved_dip_above_1000_nm": sum(
            any(d["wavelength_nm"] > 1000 for d in row["ordinary_resolved_dips"]) for row in rows),
        "ordinary_resolved_dips_above_1000_nm": sum(d["wavelength_nm"] > 1000 for d in ordinary),
        "wavelength_bin_edges_nm": bins,
        "ordinary_resolved_dip_histogram": np.histogram([d["wavelength_nm"] for d in ordinary], bins=bins)[0].tolist(),
        "unique_strict_pair_dip_histogram": np.histogram([d["wavelength_nm"] for d in strict.values()], bins=bins)[0].tolist(),
    }


def plot_overview(wavelength, reflection, transmission, absorption, rows, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    example = next(row for row in rows if row["sample_id"] == 207)
    idx = example["row"]
    ax = axes[0, 0]
    ax.plot(wavelength, reflection[idx], color="#185E91", lw=1.8, label="R (FDTD)")
    ax.plot(wavelength, absorption[idx], color="#C96038", lw=1.1, alpha=0.8, label="A = 1 - R - T")
    ax.plot(wavelength, transmission[idx], color="#23866D", lw=1.2, label="T (FDTD)")
    for n, dip in enumerate(example["ordinary_resolved_dips"]):
        x, y = dip["wavelength_nm"], dip["reflectance_min"]
        ax.scatter(x, y, color="#185E91", s=20, zorder=4)
        ax.annotate(f"{x:.0f} nm", (x, y), xytext=(0, 20 + 12 * (n % 2)),
                    textcoords="offset points", ha="center", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": "#185E91", "lw": 0.7})
    ax.set(title=f"(a) Fixed example: sample 207 ({example['split']})",
           xlabel="Wavelength (nm)", ylabel="Power fraction", xlim=(700, 1700), ylim=(-0.02, 1.05))
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    low, median, high = np.percentile(reflection, [10, 50, 90], axis=0)
    ax.fill_between(wavelength, low, high, color="#94BDDA", alpha=0.6, label="10th-90th percentiles")
    ax.plot(wavelength, median, color="#185E91", lw=1.7, label="Pointwise median R")
    ax.set(title="(b) All 1,000 stored FDTD spectra", xlabel="Wavelength (nm)",
           ylabel="Reflectance", xlim=(700, 1700), ylim=(-0.02, 1.05))
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ordinary = [d["wavelength_nm"] for row in rows for d in row["ordinary_resolved_dips"]]
    strict = {(row["sample_id"], d["wavelength_nm"]) for row in rows
              for pair in row["strict_pairs"] for d in pair["dips"]}
    bins = np.linspace(700, 1700, 41)
    ax.hist(ordinary, bins=bins, color="#9ABCD4", label="Ordinary dips: prominence >= 0.1, width >= 5 bins")
    ax.hist([x[1] for x in strict], bins=bins, color="#C96038", alpha=0.8,
            label="Dips in qualifying strict pairs")
    ax.axvline(1000, color="#444444", ls="--", lw=1)
    ax.set(title="(c) Dip positions: ordinary vs. strict-pair members",
           xlabel="Dip wavelength (nm)", ylabel="Number of dips", xlim=(700, 1700))
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    keys = ["samples_with_strict_pair", "samples_with_strict_pair_at_least_one_dip_above_1000_nm",
            "samples_with_strict_pair_both_dips_above_1000_nm"]
    labels = ["Strict pair\nanywhere", "Strict pair:\nat least one >1000", "Strict pair:\nboth >1000"]
    positions = np.arange(3)
    max_percent = max(100 * report["by_split"][split][key] / report["by_split"][split]["samples"]
                      for split in ["train", "val", "test"] for key in keys)
    y_max = max(3, max_percent * 1.4)
    for offset, split, color in zip([-0.24, 0, 0.24], ["train", "val", "test"], ["#185E91", "#23866D", "#C96038"]):
        data = report["by_split"][split]
        values = [100 * data[k] / data["samples"] for k in keys]
        bars = ax.bar(positions + offset, values, width=0.23, color=color,
                      label=f"{split} (n={data['samples']})")
        for bar, key in zip(bars, keys):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.018,
                    str(data[key]), ha="center", va="bottom", fontsize=8)
    ax.set(xticks=positions, xticklabels=labels, ylabel="Samples (%)", ylim=(0, y_max),
           title="(d) Strict-pair coverage (counts above bars)")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
    for ax in axes[0]:
        ax.axvline(1000, color="#555555", ls="--", lw=0.8, alpha=0.65)
    fig.suptitle("Top-metal dataset | bottom to top: SiO2 / Au 100 nm / As2Se3 tetramer", fontsize=14, y=0.99)
    fig.text(0.5, 0.015, "Existing dataset FDTD spectra; no new prediction round-trip simulation. Metal-only 8 nm: mesh convergence not established.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=2.3, w_pad=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "examples/jc220_topmetal_1000/dataset.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/topmetal1000")
    parser.add_argument("--target-csv", type=Path, default=ROOT / "inputs/target_topmetal.csv")
    args = parser.parse_args()
    cache = ROOT / "tmp/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    with np.load(args.dataset, allow_pickle=False) as data:
        wavelength = data["wavelength_nm"].copy()
        reflection = data["spectra"].copy()
        transmission = data["transmittance"].copy()
        absorption = data["absorptance"].copy()
        sample_ids, splits = data["sample_ids"].copy(), data["split_names"].copy()
    rows = []
    for index, (sample_id, split, spectrum) in enumerate(zip(sample_ids, splits, reflection)):
        result = double_pair(wavelength, spectrum)
        pairs = all_strict_pairs(wavelength, spectrum, result["dips"])
        if (max(pairs, key=lambda p: p["score"]) if pairs else None) != result["dominant_pair"]:
            raise RuntimeError("All-pair enumeration disagrees with the existing double_pair selector")
        rows.append({"row": index, "sample_id": int(sample_id), "split": str(split),
                     "ordinary_resolved_dips": [d for d in result["dips"] if d["resolved"]],
                     "strict_pairs": pairs, "dominant_pair": result["dominant_pair"]})
    example = next(row for row in rows if row["sample_id"] == 207)
    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "spectrum_source": "stored dataset FDTD reflectance; no new prediction round-trip simulation",
        "new_fdtd_run_performed": False,
        "quality_level": "pipeline_validation",
        "mesh_convergence_demonstrated": False,
        "strict_pair_criteria": CRITERIA,
        "ordinary_dip_criteria": {"minimum_prominence": 0.1, "minimum_width_samples": 5},
        "criteria_source": "explore_double_dips.double_pair and explore_spectra.features",
        "above_1000_rule": "strictly greater than 1000 nm; strict-pair coverage considers ALL qualifying pairs, not only the strongest pair",
        "width_interpretation": "Half-prominence widths are descriptive spectral widths, not fitted modal linewidths or certified Q factors.",
        "ordinary_and_strict_are_distinct": True,
        "all": coverage(rows),
        "by_split": {split: coverage([row for row in rows if row["split"] == split]) for split in ["train", "val", "test"]},
        "fixed_example_207": {**example, "selection_reason": "Fixed sample ID 207 requested before model evaluation; not selected by prediction error.",
                              "target_csv": str(args.target_csv.resolve())},
        "samples": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "spectral_coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.target_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.target_csv, np.column_stack((wavelength, reflection[example["row"]])), delimiter=",",
               header="wavelength_nm,reflectance", comments="", fmt="%.12g")
    plot_overview(wavelength, reflection, transmission, absorption, rows, report, args.output_dir / "dataset_overview.png")
    print(json.dumps({"all": report["all"], "by_split": report["by_split"], "example_207": example}, indent=2))


if __name__ == "__main__":
    main()
