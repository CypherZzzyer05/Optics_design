"""Exercise Matplotlib mouse events and keep preview geometry separate from real data."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np

from fdtd_pipeline.config import Geometry, PARAMETER_NAMES, ROTATION_PARAMETER_NAMES
from pycharm_demo import build_dataset_figure


def sample_archive(path, rotated=True, count=2):
    values = np.array([[3200, 900, 1000, 250, 300, 10], [3300, 950, 950, 320, 340, 25]], dtype=float)[:count]
    lower = np.array([3000, 850, 800, 150, 150, -45], dtype=float)
    upper = np.array([3400, 1050, 1100, 350, 350, 45], dtype=float)
    spectra = np.stack([np.linspace(.1 + .1 * i, .4 + .1 * i, 1000) for i in range(count)])
    labels = (values - lower) / (upper - lower)
    metadata = {'config': {'quality_level': 'pipeline_validation'}}
    arrays = dict(spectra=spectra, transmittance=.9 - spectra, absorptance=np.full_like(spectra, .1),
                  wavelength_nm=np.linspace(4000, 8000, 1000), geometry_nm=values[:, :5],
                  geometry_01=labels if rotated else labels[:, :5],
                  parameter_names=np.array(ROTATION_PARAMETER_NAMES if rotated else PARAMETER_NAMES),
                  sample_ids=np.arange(count), split_names=np.array(['train'] * count),
                  status=np.full(count, 2), train_indices=np.arange(count),
                  val_indices=np.array([], dtype=int), test_indices=np.array([], dtype=int),
                  metadata_json=np.array(json.dumps(metadata)))
    if rotated:
        arrays.update(geometry_values=values, rotation_deg=values[:, 5], parameter_units=np.array(['nm'] * 5 + ['deg']))
    np.savez(path, **arrays)
    return arrays


def drag_slider(figure, slider, value):
    for name, position in [('button_press_event', slider.val), ('motion_notify_event', value),
                           ('button_release_event', value)]:
        x, y = slider.ax.transData.transform((position, .5))
        event = MouseEvent(name, figure.canvas, x, y, button=1)
        figure.canvas.callbacks.process(name, event)
    figure.canvas.draw()


class PreviewInteractionTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_mouse_drag_rotates_all_squares_without_changing_spectrum_or_labels(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.npz'
            arrays = sample_archive(path)
            original_bytes = path.read_bytes()
            fig = build_dataset_figure(path)
            fig.canvas.draw()
            drag_slider(fig, fig._angle_slider, 30)
            self.assertAlmostEqual(fig._preview_state['angle_deg'], 30, delta=.15)
            self.assertTrue(fig._preview_state['preview_only'])
            expected = Geometry(*arrays['geometry_nm'][0], rotation_deg=fig._preview_state['angle_deg'])
            for patch, vertices in zip(fig.axes[1].patches[1:], expected.vertices()):
                np.testing.assert_allclose(patch.get_xy()[:-1], vertices)
            np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), arrays['spectra'][0])
            np.testing.assert_allclose([bar.get_height() for bar in fig.axes[2].patches], arrays['geometry_01'][0])
            drag_slider(fig, fig._angle_slider, 40)
            self.assertFalse(fig._preview_state['geometry_valid'])
            self.assertIn('不能采样', fig.texts[-1].get_text())
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_sample_mouse_click_and_reset_restore_the_real_sample_angle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.npz'
            arrays = sample_archive(path)
            fig = build_dataset_figure(path)
            fig.canvas.draw()
            drag_slider(fig, fig._angle_slider, 30)
            drag_slider(fig, fig._sample_slider, .95)
            self.assertEqual(fig._preview_state['index'], 1)
            self.assertEqual(fig._angle_slider.val, 25)
            self.assertFalse(fig._preview_state['preview_only'])
            np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), arrays['spectra'][1])
            drag_slider(fig, fig._angle_slider, -15)
            x, y = fig._reset_button.ax.transAxes.transform((.5, .5))
            for name in ('button_press_event', 'button_release_event'):
                fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=1))
            self.assertEqual(fig._angle_slider.val, 25)
            self.assertFalse(fig._preview_state['preview_only'])

    def test_single_sample_angle_preview_and_legacy_five_parameter_browser(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.npz'
            sample_archive(path, count=1)
            fig = build_dataset_figure(path)
            self.assertFalse(hasattr(fig, '_sample_slider'))
            fig._angle_slider.set_val(20)
            self.assertTrue(fig._preview_state['preview_only'])
            plt.close(fig)
            arrays = sample_archive(path, rotated=False)
            fig = build_dataset_figure(path)
            self.assertFalse(hasattr(fig, '_angle_slider'))
            fig.canvas.draw()
            drag_slider(fig, fig._sample_slider, .95)
            np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), arrays['spectra'][1])


if __name__ == '__main__':
    unittest.main()
