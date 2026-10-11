"""Independent angles, metal-stack monitor placement, seven-label training and feature QA."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from fdtd_pipeline.config import Geometry, PROJECT_ROOT, QUADRANT_PARAMETER_NAMES, load_config
from fdtd_pipeline.lumerical import FDTDSimulator, mesh_grid_summary
from fdtd_pipeline.pipeline import generate_dataset
from metaformer import MetaformerConfig, InverseMetaformer, geometry_mse
from spectrum_dataset import SpectrumGeometryDataset
from train_metaformer import TrainingConfig, train, regression_metrics
from explore_spectra import features, study_plan
from explore_double_dips import double_pair, control_plan, source_candidate
from test_fdtd_pipeline import FakeSimulator


class QuadrantTests(unittest.TestCase):
    def test_fresh_project_can_read_committed_real_candidate_without_coarse_generation(self):
        config = load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json')
        with TemporaryDirectory() as directory:
            fresh = replace(config, output_dir=Path(directory)/'as2se3_au_pilot')
            values, source = source_candidate(fresh, 3)
            self.assertEqual(len(values), 7)
            self.assertIsNotNone(source['dominant_pair'])
            self.assertFalse(fresh.output_dir.exists())
            wrong = replace(config, output_dir=Path(directory)/'different_user_dataset')
            with self.assertRaises(FileNotFoundError):
                source_candidate(wrong, 3)

    def test_strict_mesh_limit_covers_full_domain_and_checks_actual_cells(self):
        config = load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json')
        for metal in ('au', 'ag'):
            formal = load_config(PROJECT_ROOT/f'configs/as2se3_{metal}_fine8.json')
            self.assertEqual(formal.simulation.maximum_mesh_nm, 8)
            self.assertEqual(formal.simulation.mesh_nm, 8)
            self.assertEqual(formal.simulation.metal_mesh_nm, 2.5)
            self.assertEqual(formal.parameter_names, QUADRANT_PARAMETER_NAMES)
        self.assertNotIn('maximum_mesh_nm', config.metadata()['simulation'])
        for value in (10, 20, 0, float('nan')):
            with self.assertRaises(ValueError):
                replace(config.simulation, mesh_nm=8, maximum_mesh_nm=value)
        settings = replace(config.simulation, mesh_nm=8, metal_mesh_nm=2.5, maximum_mesh_nm=8)
        simulator = object.__new__(FDTDSimulator)
        simulator.settings, simulator.fdtd = settings, MagicMock()
        g = Geometry(667, 167, 369, 97, 119, theta_odd_deg=-5, theta_even_deg=41)
        simulator.build(g)
        mesh = {c.kwargs['name']: c.kwargs for c in simulator.fdtd.addmesh.call_args_list}['global_mesh_limit']
        self.assertAlmostEqual(mesh['z_min'], -1700e-9)
        self.assertAlmostEqual(mesh['z_max'], (369+1200)*1e-9)
        for axis in ('x', 'y', 'z'):
            self.assertAlmostEqual(mesh['d'+axis], 8e-9)
            self.assertTrue(mesh['override_'+axis+'_mesh'])
        grids = {axis: np.array([-16, -8, 0, 8, 16])*1e-9 for axis in ('x', 'y', 'z')}
        summary = mesh_grid_summary(grids, 8)
        self.assertAlmostEqual(summary['z']['maximum_step_nm'], 8)
        grids['z'] = np.array([-16, -8, 0, 12])*1e-9
        with self.assertRaisesRegex(ValueError, 'Actual z mesh'):
            mesh_grid_summary(grids, 8)
        grids['z'] = np.array([0, 0, 8])*1e-9
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            mesh_grid_summary(grids, 8)
    def test_double_pair_rejects_single_unresolved_and_edge_dips(self):
        w = np.linspace(700, 1700, 3000)
        def trough(center, width=3):
            return .9*np.exp(-.5*((w-center)/width)**2)
        r = .95-trough(838)-trough(924)
        pair = double_pair(w, r)['dominant_pair']
        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair['separation_nm'], 86, delta=.7)
        self.assertIsNone(double_pair(w, .95-trough(838))['dominant_pair'])
        self.assertIsNone(double_pair(w, .95-trough(838)-trough(924, .2))['dominant_pair'])
        self.assertIsNone(double_pair(w, .95-trough(710)-trough(924))['dominant_pair'])
        values = [667, 167, 369, 97, 119, -5, 41]
        plan = control_plan(values)
        self.assertEqual(len(plan), 6)
        for entry in plan:
            self.assertEqual(entry['geometry_values'][:5], values[:5])
            self.assertAlmostEqual(sum(entry['geometry_values'][5:])/2, 18)

    def test_research_resolution_cannot_enter_training_pipeline(self):
        config = load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json')
        simulation = replace(config.simulation, research_grid=True, spectrum_points=3000)
        self.assertEqual(len(simulation.wavelength_grid()), 3000)
        with self.assertRaisesRegex(ValueError, 'cannot be packed'):
            replace(config, simulation=simulation)
        self.assertNotIn('research_grid', config.metadata()['simulation'])

    def test_angles_and_self_centers_preserve_C2(self):
        g = Geometry(660, 190, 300, 120, 120, theta_odd_deg=20, theta_even_deg=-10)
        self.assertEqual(g.angles(), [20, -10, -10, 20])
        vertices = g.vertices()
        for v, center in zip(vertices, g.centers()):
            np.testing.assert_allclose(v.mean(axis=0), center)
            np.testing.assert_allclose(np.linalg.norm(np.roll(v, -1, axis=0)-v, axis=1), 190)
        np.testing.assert_allclose(vertices[0], -np.roll(vertices[3], 2, axis=0))
        with self.assertRaises(ValueError):
            Geometry(660, 190, 300, 120, 120, theta_odd_deg=10)
        with self.assertRaises(ValueError):
            Geometry(660, 190, 300, 5, 5, theta_odd_deg=45, theta_even_deg=-45)
        with self.assertRaises(ValueError):
            Geometry(450, 190, 300, 40, 40, theta_odd_deg=45, theta_even_deg=0)

    def test_metal_layer_is_below_finite_substrate_and_T_below_metal(self):
        simulator = object.__new__(FDTDSimulator)
        s = load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json').simulation
        simulator.settings, simulator.fdtd = s, MagicMock()
        g = Geometry(660, 190, 300, 120, 120, theta_odd_deg=20, theta_even_deg=-10)
        simulator.build(g)
        objects = {call.kwargs['name']:call.kwargs for call in simulator.fdtd.addrect.call_args_list}
        self.assertAlmostEqual(objects['substrate']['z_min'], -350e-9)
        self.assertAlmostEqual(objects['back_metal']['z_min'], -500e-9)
        self.assertAlmostEqual(objects['back_metal']['z_max'], objects['substrate']['z_min'])
        for i in range(4):
            self.assertEqual(objects[f'si_{i}']['material'], s.resonator_material)
        rotations = [c.args[1] for c in simulator.fdtd.set.call_args_list if c.args[0]=='rotation 1']
        self.assertEqual(rotations, [-20, 10, 10, -20])
        monitors = {c.kwargs['name']:c.kwargs for c in simulator.fdtd.addpower.call_args_list}
        self.assertLess(monitors['T']['z'], objects['back_metal']['z_min'])

    def test_periodic_angles_do_not_independently_discard_signs(self):
        target = torch.full((1,7), .5, dtype=torch.float64)
        predicted = target.clone()
        target[0,5] = 0
        predicted[0,5] = 1
        self.assertAlmostEqual(geometry_mse(predicted,target,'square_periodic').item(), 0)
        predicted[0,5], target[0,5] = 65/90, 25/90
        self.assertGreater(geometry_mse(predicted,target,'square_periodic').item(), 0)
        predicted.requires_grad_()
        geometry_mse(predicted,target,'square_periodic').backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        metric=regression_metrics(predicted.detach().numpy(),target.numpy(),[620,160,240,90,90,-45,-45],
                                  [690,210,380,150,150,45,45],angle_symmetry='square_periodic')
        self.assertEqual(set(metric['mae_deg']),set(QUADRANT_PARAMETER_NAMES[5:]))

    def test_resume_pack_seven_outputs_and_optimizer_updates(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with TemporaryDirectory() as directory:
                root=Path(directory)
                cfg=replace(load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json'),output_dir=root/'data',sample_count=6)
                FakeSimulator.calls,FakeSimulator.fail_first=0,False
                with patch('fdtd_pipeline.pipeline.FDTDSimulator',FakeSimulator):
                    generate_dataset(cfg,limit=2)
                    summary=generate_dataset(cfg)
                    generate_dataset(cfg)
                self.assertEqual(FakeSimulator.calls,6)
                self.assertEqual(summary['geometry_shape'],[6,7])
                ds=SpectrumGeometryDataset(cfg.output_dir/'dataset.npz','all',True)
                np.testing.assert_allclose(ds.denormalize(ds.labels),ds.geometry_values,atol=.001)
                self.assertEqual(ds.parameter_units,('nm',)*5+('deg',)*2)
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from pycharm_demo import build_dataset_figure
                figure=build_dataset_figure(cfg.output_dir/'dataset.npz')
                try:
                    polygons=[p for p in figure.axes[1].patches if hasattr(p,'get_xy') and np.asarray(p.get_xy()).shape==(5,2)]
                    geometry=cfg.make_geometry(ds.geometry_values[0])
                    for polygon,vertices in zip(polygons,geometry.vertices()):
                        np.testing.assert_allclose(polygon.get_xy()[:4],vertices)
                    figure._sample_slider.set_val(1)
                    self.assertEqual(figure._preview_state['index'],1)
                finally:
                    plt.close(figure)
                model=MetaformerConfig(geometry_dim=7,angle_symmetry='square_periodic',patch_count=10,
                                       embedding_dim=8,attention_heads=2,head_dim=4,encoder_layers=1,mlp_dim=16)
                self.assertEqual(InverseMetaformer(model)(ds.spectra[:2]).shape,(2,7))
                training=TrainingConfig(cfg.output_dir/'dataset.npz',root/'runs',allow_validation_data=True,
                                        epochs=2,batch_size=2,model=model)
                with patch('plot_training.plot_training'):
                    run_dir,report=train(training)
                self.assertGreater(report['optimizer_steps'],0)
                self.assertEqual(set(report['metrics']['test']['mae_deg']),set(QUADRANT_PARAMETER_NAMES[5:]))
                header=(run_dir/'test_predictions.csv').read_text().splitlines()[0]
                self.assertIn('true_deg_theta_even_deg',header)
        finally:
            torch.set_num_threads(previous)

    def test_spectral_scoring_and_control_plan(self):
        wavelength=np.linspace(700,1700,1000)
        flat=features(wavelength,np.full(1000,.9))
        self.assertEqual(flat,[])
        dip=.9-.6*np.exp(-((wavelength-1200)/25)**2)
        found=features(wavelength,dip)
        self.assertEqual(len(found),1)
        self.assertAlmostEqual(found[0]['wavelength_nm'],1200,delta=1)
        self.assertTrue(found[0]['resolved'])
        self.assertAlmostEqual(found[0]['half_prominence_width_nm'],2*25*np.sqrt(np.log(2)),delta=1)
        self.assertEqual(len(study_plan()),14)

    def test_material_coverage_extension_and_content_fingerprint(self):
        with TemporaryDirectory() as directory:
            csv=Path(directory)/'nk.csv'
            csv.write_text('wavelength_nm,n,k\n700,2.9,.1\n1000,2.7,.03\n1688,2.6,.06\n')
            config=load_config(PROJECT_ROOT/'configs/as2se3_au_pilot.json')
            config=replace(config,simulation=replace(config.simulation,resonator_nk_csv=str(csv)))
            initial=config.metadata()['resonator_nk_sha256']
            csv.write_text(csv.read_text().replace('2.9','2.8'))
            self.assertNotEqual(initial,config.metadata()['resonator_nk_sha256'])
            simulator=object.__new__(FDTDSimulator)
            simulator.settings=replace(config.simulation,material_max_extension_nm=0)
            simulator.lumapi=MagicMock()
            with self.assertRaisesRegex(ValueError,'does not cover'):
                simulator.__enter__()
            self.assertIsNone(simulator.fdtd)
            # A narrow source band between two measured knots is also supported.
            simulator.settings=replace(config.simulation,wavelength_start_nm=800,wavelength_stop_nm=801)
            with simulator:
                self.assertEqual(len(simulator.material_data),2)


if __name__=='__main__':
    unittest.main()
