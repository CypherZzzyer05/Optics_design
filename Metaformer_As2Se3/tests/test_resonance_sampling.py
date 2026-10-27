import unittest

import torch

from metaformer import SinusoidalPositionEncoding
from resonance_sampling import (build_extra_sampling_grid, continuous_position_embedding,
                                linear_sample)


def valley(center, sigma, depth=.4):
    axis = torch.arange(1000, dtype=torch.float32)
    return 1 - depth * torch.exp(-.5 * ((axis - center) / sigma).square())


class ResonanceSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_uniform_grid_and_flat_resonance_fallback(self):
        raw = torch.full((3, 1000), .75)
        uniform = build_extra_sampling_grid(raw, "uniform_extra")
        detected = build_extra_sampling_grid(raw, "resonance")
        self.assertEqual(detected.shape, (3, 32, 10))
        self.assertEqual(detected.dtype, torch.float32)
        torch.testing.assert_close(detected, uniform)
        self.assertEqual(uniform.min().item(), 0)
        self.assertEqual(uniform.max().item(), 999)
        self.assertTrue((uniform.flatten(1).diff() > 0).all())

    def test_single_valley_local_window_and_global_second_block(self):
        raw = valley(380, 7)[None].requires_grad_()
        before = raw.detach().clone()
        grid = build_extra_sampling_grid(raw, "resonance")
        self.assertAlmostEqual(grid[0, :16].mean().item(), 380, places=4)
        self.assertLess(grid[0, :16].max().item() - grid[0, :16].min().item(), 100)
        torch.testing.assert_close(grid[0, 16:].flatten(), torch.linspace(0, 999, 160))
        torch.testing.assert_close(raw.detach(), before, atol=0, rtol=0)
        self.assertFalse(grid.requires_grad)

    def test_two_valleys_are_ordered_by_wavelength_not_depth(self):
        raw = (valley(240, 8, .2) + valley(720, 16, .6) - 1)[None]
        grid = build_extra_sampling_grid(raw, "resonance")
        self.assertAlmostEqual(grid[0, :16].mean().item(), 240, places=4)
        self.assertAlmostEqual(grid[0, 16:].mean().item(), 720, places=4)
        self.assertLess(grid[0, :16].max().item(), grid[0, 16:].min().item())

    def test_only_two_most_prominent_valleys_are_selected(self):
        raw = (valley(200, 8, .4) + valley(500, 8, .015)
               + valley(800, 8, .3) - 2)[None]
        grid = build_extra_sampling_grid(raw, "resonance")
        self.assertAlmostEqual(grid[0, :16].mean().item(), 200, places=4)
        self.assertAlmostEqual(grid[0, 16:].mean().item(), 800, delta=1e-3)

    def test_shallow_ripple_does_not_create_resonance_windows(self):
        raw = valley(350, 10, .005)[None]
        torch.testing.assert_close(build_extra_sampling_grid(raw, "resonance"),
                                   build_extra_sampling_grid(raw, "uniform_extra"))

    def test_narrow_and_wide_windows_are_bounded(self):
        raw = torch.stack((valley(450, 1), valley(550, 80)))
        grid = build_extra_sampling_grid(raw, "resonance")
        spans = grid[:, :16, :].amax((1, 2)) - grid[:, :16, :].amin((1, 2))
        self.assertGreaterEqual(spans[0].item(), 16)
        self.assertLessEqual(spans[0].item(), 50)
        self.assertAlmostEqual(spans[1].item(), 240, places=4)

    def test_boundary_windows_are_clipped_without_center_shift(self):
        grid = build_extra_sampling_grid(torch.stack((valley(8, 6), valley(991, 6))),
                                         "resonance")
        self.assertEqual(grid[0, 0, 0].item(), 0)
        self.assertEqual(grid[1, 15, -1].item(), 999)
        self.assertLess(grid[0, 15, -1].item(), 40)
        self.assertGreater(grid[1, 0, 0].item(), 959)
        torch.testing.assert_close(grid[0, :16].flatten(),
                                   999 - grid[1, :16].flatten().flip(0), atol=2e-4, rtol=0)

    def test_linear_sampling_preserves_linear_signal_and_gradients(self):
        signal = (2 * torch.arange(1000, dtype=torch.float32) + 3)[None, None]
        signal.requires_grad_()
        grid = build_extra_sampling_grid(torch.ones(1, 1000), "uniform_extra")
        sampled = linear_sample(signal, grid)
        self.assertEqual(sampled.shape, (1, 32, 10))
        torch.testing.assert_close(sampled, 2 * grid + 3, atol=2e-4, rtol=1e-6)
        sampled.sum().backward()
        self.assertTrue(torch.isfinite(signal.grad).all())
        self.assertAlmostEqual(signal.grad.sum().item(), 320, places=4)
        self.assertEqual(sampled[0, 0, 0].item(), 3)
        self.assertEqual(sampled[0, -1, -1].item(), 2001)

    def test_native_center_and_span_match_existing_absolute_positions(self):
        centers = (torch.arange(100) * 10 + 4.5)[None]
        spans = torch.full_like(centers, 9)
        for dim in (16, 17, 96):
            with self.subTest(dim=dim):
                expected = SinusoidalPositionEncoding(100, dim).encoding
                actual = continuous_position_embedding(centers, spans, dim)
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
                shifted = continuous_position_embedding(centers + 10, spans, dim)
                torch.testing.assert_close(shifted[:, :-1], expected[:, 1:], atol=1e-7, rtol=1e-6)
                wider = continuous_position_embedding(centers, spans * 2, dim)
                self.assertGreater((wider - actual).abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_grid_and_positions_follow_input_device(self):
        raw = valley(400, 10)[None].cuda()
        grid = build_extra_sampling_grid(raw, "resonance")
        sampled = linear_sample(raw[:, None], grid)
        position = continuous_position_embedding(grid.mean(-1), grid[..., -1] - grid[..., 0], 16)
        self.assertEqual(grid.device, raw.device)
        self.assertEqual(sampled.device, raw.device)
        self.assertEqual(position.device, raw.device)
        self.assertTrue(torch.isfinite(position).all())


if __name__ == "__main__":
    unittest.main()
