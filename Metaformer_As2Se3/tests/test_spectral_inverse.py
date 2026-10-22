import unittest
import torch

from spectral_inverse import (SpectralConfig, SpectralInverse, SpectralViews,
                               inverse_loss, mirror_labels, parameter_errors)


class SpectralInverseTests(unittest.TestCase):
    def test_joint_mirror_not_independent_coordinate_minimum(self):
        target = torch.tensor([[.2, .3, .4, .5, .6, .1, .6]])
        mirror = mirror_labels(target)
        self.assertLess(inverse_loss(mirror, target, True).item(), 1e-12)
        self.assertGreater(inverse_loss(mirror, target, False).item(), .01)
        mixed = target.clone()
        mixed[:, 5] = mirror[:, 5]
        self.assertGreater(inverse_loss(mixed, target, True).item(), .001)
        self.assertTrue(torch.allclose(mirror_labels(mirror), target, atol=1e-7))

    def test_square_period_boundary_and_gradients(self):
        target = torch.tensor([[.2, .3, .4, .5, .6, .99, .01]])
        prediction = target.clone()
        prediction[:, 5:] = torch.tensor([[.01, .99]])
        prediction.requires_grad_(True)
        errors = parameter_errors(prediction, target)
        self.assertTrue(torch.allclose(errors[:, 5:].abs(), torch.full((1, 2), .02), atol=1e-6))
        inverse_loss(prediction, target, True).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_preprocessing_uses_train_only_and_preserves_raw(self):
        config = SpectralConfig(view='shape_fft')
        views = SpectralViews(config)
        training = torch.stack((torch.linspace(0, 1, 1000), torch.linspace(.1, .9, 1000)))
        before = training.clone()
        views.fit(training)
        saved = views.mean.clone()
        views(torch.ones(2, 1000) * 9)
        self.assertTrue(torch.equal(views.mean, saved))
        self.assertTrue(torch.equal(training, before))
        self.assertTrue(torch.equal(views.raw_views(training)[:, 0], training))
        # Fixed SG kernels differentiate linear functions correctly away from padded edges.
        derivatives = views.raw_views(training)[0]
        self.assertTrue(torch.allclose(derivatives[1, 10:-10], torch.full((980,), .001), atol=1e-7))
        self.assertLess(derivatives[2, 10:-10].abs().max().item(), 1e-7)

    def test_all_views_update_and_reload(self):
        torch.set_num_threads(2)
        x, y = torch.rand(4, 1000), torch.rand(4, 7)
        for view in ('raw', 'shape', 'shape_fft'):
            with self.subTest(view=view):
                config = SpectralConfig(view=view, embedding_dim=16, encoder_layers=1, dropout=0)
                model = SpectralInverse(config)
                model.fit_input_normalization(x)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                before = model.head[-2].weight.detach().clone()
                prediction = model(x)
                self.assertEqual(prediction.shape, (4, 7))
                inverse_loss(prediction, y, True).backward()
                optimizer.step()
                self.assertFalse(torch.equal(model.head[-2].weight, before))
                restored = SpectralInverse(config)
                restored.load_state_dict(model.state_dict())
                model.eval(), restored.eval()
                self.assertTrue(torch.equal(model(x), restored(x)))

    def test_fourier_retains_phase(self):
        views = SpectralViews(SpectralConfig())
        phase = torch.arange(1000) * (2 * torch.pi / 1000)
        first = torch.sin(3 * phase)[None]
        shifted = torch.cos(3 * phase)[None]
        first_fft, shifted_fft = torch.fft.rfft(first), torch.fft.rfft(shifted)
        self.assertTrue(torch.allclose(first_fft.abs(), shifted_fft.abs(), atol=.001))
        self.assertFalse(torch.allclose(views.fourier(first), views.fourier(shifted), atol=.001))


if __name__ == '__main__':
    unittest.main()
