"""Data-integrity tests use a fake solver; delivered datasets only use real lumapi."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fdtd_pipeline.config import Geometry, load_config, PROJECT_ROOT, SimulationSettings
from fdtd_pipeline.lumerical import FDTDSimulator, validate_spectrum
from fdtd_pipeline.pipeline import generate_dataset, split_assignments, directory_lock
from spectrum_dataset import SpectrumGeometryDataset


class FakeSimulator:
    calls = 0
    fail_first = False
    def __init__(self, settings, lumapi_path):
        self.settings = settings
        self.version = "test-only"
        self.lumapi = SimpleNamespace(__file__="test-only")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, geometry, project_path):
        type(self).calls += 1
        if self.fail_first and type(self).calls == 1:
            raise RuntimeError("deliberate test failure")
        return {"wavelength_nm": self.settings.wavelength_grid(),
                "reflectance": np.full(1000, .2), "transmittance": np.full(1000, .7),
                "absorptance": np.full(1000, .1), "status": 2,
                "elapsed_seconds": .1, "fdtd_version": "test-only"}


class ConvergenceRetryTests(unittest.TestCase):
    def make_simulator(self, statuses):
        simulator = object.__new__(FDTDSimulator)
        simulator.settings = SimulationSettings(simulation_time_ps=10)
        simulator.version = "test-only"
        simulator.fdtd = MagicMock()
        simulator.build = MagicMock()
        states = iter(statuses)
        def result(name, key):
            if name == "FDTD":
                return np.float64(next(states))
            return {"lambda": simulator.settings.wavelength_grid()*1e-9,
                    "T": np.full(1000, .2 if name == "R" else -.7)}
        simulator.fdtd.getresult.side_effect = result
        return simulator

    def test_retry_reaches_convergence_and_records_actual_time_limit(self):
        simulator = self.make_simulator([1, 2])
        with TemporaryDirectory() as directory:
            result = simulator.run(Geometry(3200, 900, 1000, 200, 300), Path(directory)/"sample.fsp")
        self.assertEqual(result["status"], 2)
        self.assertEqual(result["run_attempts"], 2)
        self.assertEqual(result["simulation_time_ps"], 20)
        self.assertEqual([a["status"] for a in json.loads(result["attempts_json"])], [1, 2])
        np.testing.assert_allclose(result["transmittance"], .7)

    def test_divergence_is_not_retried(self):
        simulator = self.make_simulator([3])
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "diverged"):
                simulator.run(Geometry(3200, 900, 1000, 200, 300), Path(directory)/"sample.fsp")
        self.assertEqual(simulator.fdtd.run.call_count, 1)

    def test_retry_is_bounded_and_never_accepts_nonconvergence(self):
        simulator = self.make_simulator([1, 1, 1, 1])
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "time limit"):
                simulator.run(Geometry(3200, 900, 1000, 200, 300), Path(directory)/"sample.fsp")
        self.assertEqual(simulator.fdtd.run.call_count, 4)


class PhysicalValidationTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimulationSettings()
        self.grid = self.settings.wavelength_grid()

    def test_edge_gaps_and_nonoverlap(self):
        g = Geometry(3200, 900, 1000, 200, 300)
        self.assertEqual(g.centers()[3], (550, 600))
        with self.assertRaises(ValueError):
            Geometry(1900, 900, 1000, 200, 300)

    def test_1000_point_and_config_validation(self):
        with self.assertRaises(ValueError):
            SimulationSettings(spectrum_points=999)
        with self.assertRaises(ValueError):
            SimulationSettings(threads=0)
        with self.assertRaises(ValueError):
            SimulationSettings(mesh_nm=float("nan"))

    def test_reversed_grid_preserves_spectrum_alignment(self):
        r = np.linspace(.1, .3, 1000)
        result = validate_spectrum(self.grid[::-1], r[::-1], np.full(1000, .6), self.settings, 2)
        np.testing.assert_array_equal(result["reflectance"], r)

    def test_reject_grid_not_wavelength_uniform(self):
        bad = 1 / np.linspace(1/4000, 1/8000, 1000)
        with self.assertRaisesRegex(ValueError, "grid"):
            validate_spectrum(bad, np.full(1000, .2), np.full(1000, .7), self.settings, 2)

    def test_reject_negative_transmitted_flux(self):
        with self.assertRaisesRegex(ValueError, "Negative"):
            validate_spectrum(self.grid, np.full(1000, .2), np.full(1000, -.7), self.settings, 2)

    def test_reject_nonfinite_data(self):
        r = np.full(1000, .2); r[9] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_spectrum(self.grid, r, np.full(1000, .7), self.settings, 2)

    def test_reject_nonconvergence_and_divergence(self):
        for status in (0, 1, 3):
            with self.assertRaises(ValueError):
                validate_spectrum(self.grid, np.full(1000, .2), np.full(1000, .7), self.settings, status)

    def test_reject_energy_creation(self):
        with self.assertRaisesRegex(ValueError, "R\\+T"):
            validate_spectrum(self.grid, np.full(1000, .6), np.full(1000, .6), self.settings, 2)


class DatasetPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.config = replace(load_config(PROJECT_ROOT / "configs/smoke.json"),
                              output_dir=Path(self.tmp.name)/"dataset", sample_count=4,
                              simulation=replace(SimulationSettings(), save_projects=True))
        FakeSimulator.calls = 0
        FakeSimulator.fail_first = False
        self.patcher = patch("fdtd_pipeline.pipeline.FDTDSimulator", FakeSimulator)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def test_seeded_sampling_and_splits(self):
        np.testing.assert_array_equal(self.config.sample_geometries()[0].array(), self.config.sample_geometries()[0].array())
        self.assertEqual(split_assignments(4, 42), split_assignments(4, 42))
        self.assertEqual(sorted(split_assignments(4, 42)), ["test", "train", "train", "val"])

    def test_resume_and_training_shapes_and_roundtrip(self):
        a = generate_dataset(self.config, limit=2)
        self.assertEqual(a["success"], 2)
        self.assertEqual(a["pending"], 2)
        b = generate_dataset(self.config)
        self.assertEqual(b["success"], 4)
        self.assertEqual(FakeSimulator.calls, 4)
        generate_dataset(self.config)
        self.assertEqual(FakeSimulator.calls, 4)
        dataset = SpectrumGeometryDataset(self.config.output_dir/"dataset.npz", "all", allow_validation_data=True)
        self.assertEqual(tuple(dataset.spectra.shape), (4, 1000))
        np.testing.assert_allclose(dataset.denormalize(dataset.labels).numpy(), dataset.geometry_nm, atol=.001)
        with self.assertRaisesRegex(ValueError, "pipeline-validation"):
            SpectrumGeometryDataset(self.config.output_dir/"dataset.npz")

    def test_failed_samples_excluded_and_retried(self):
        FakeSimulator.fail_first = True
        first = generate_dataset(self.config)
        self.assertEqual(first["success"], 3)
        self.assertEqual(first["failed"], 1)
        second = generate_dataset(self.config)
        self.assertEqual(second["success"], 4)
        self.assertEqual(FakeSimulator.calls, 5)

    def test_changed_configuration_cannot_mix_datasets(self):
        generate_dataset(self.config, limit=1)
        with self.assertRaisesRegex(ValueError, "different configuration"):
            generate_dataset(replace(self.config, seed=999))

    def test_corrupt_multiple_samples_regenerated(self):
        generate_dataset(self.config)
        for i in (0, 1):
            (self.config.output_dir/"samples"/f"sample_{i:06d}.npz").write_bytes(b"broken")
        result = generate_dataset(self.config)
        self.assertEqual(result["success"], 4)
        self.assertEqual(FakeSimulator.calls, 6)

    def test_directory_lock_prevents_concurrent_writers(self):
        with directory_lock(self.config.output_dir):
            with self.assertRaisesRegex(RuntimeError, "Another generator"):
                with directory_lock(self.config.output_dir):
                    pass


if __name__ == "__main__":
    unittest.main()
