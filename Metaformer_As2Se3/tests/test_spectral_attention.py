"""Check raw-spectrum embedding, attention gradients, and checkpoint compatibility."""
from copy import deepcopy
from dataclasses import asdict
import unittest

import torch
from torch.nn import functional as F

from metaformer import (InverseMetaformer, MetaformerConfig, MultiHeadSelfAttention,
                        SinusoidalPositionEncoding, geometry_mse)
from train_metaformer import matching_training_config


class SpectralAttentionTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_attention_outputs_and_gradients_match_independent_sdpa(self):
        attention = MultiHeadSelfAttention(128, 10, 64, 0).double().eval()
        reference = deepcopy(attention)
        tokens = torch.randn(2, 7, 128, dtype=torch.float64, requires_grad=True)
        reference_tokens = tokens.detach().clone().requires_grad_(True)
        output, maps = attention(tokens, return_attention=True)
        qkv = reference.qkv(reference_tokens).reshape(2, 7, 3, 10, 64)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0)
        expected = reference.output(attended.transpose(1, 2).reshape(2, 7, 640))
        torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(maps.sum(-1), torch.ones(2, 10, 7, dtype=torch.float64))
        target = torch.randn_like(output)
        ((output - target) ** 2).mean().backward()
        ((expected - target) ** 2).mean().backward()
        torch.testing.assert_close(tokens.grad, reference_tokens.grad, atol=1e-12, rtol=1e-10)
        for (name, parameter), (_, expected_parameter) in zip(attention.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=1e-12, rtol=1e-10,
                                       msg=lambda message: name + ': ' + message)
        for gradient in attention.qkv.weight.grad.chunk(3, dim=0):
            self.assertGreater(float(gradient.norm()), 0)

    def test_raw_patch_projection_matches_independent_linear_values_and_gradients(self):
        model = InverseMetaformer(MetaformerConfig(embedding_dim=16, encoder_layers=1))
        spectrum = torch.randn(2, 1000, requires_grad=True)
        projected = model.patch_embedding(spectrum.unsqueeze(1)).transpose(1, 2)
        reference_spectrum = spectrum.detach().clone().requires_grad_(True)
        expected = F.linear(reference_spectrum.reshape(2, 100, 10),
                            model.patch_embedding.weight[:, 0, :], model.patch_embedding.bias)
        torch.testing.assert_close(projected, expected)
        probe = torch.randn_like(projected)
        (projected * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(spectrum.grad, reference_spectrum.grad)
        self.assertFalse(hasattr(model, 'spectral_frontend'))
        self.assertFalse(any(isinstance(m, torch.nn.GroupNorm) for m in model.modules()))

    def test_position_encoding_matches_paper_equation(self):
        encoded = SinusoidalPositionEncoding(100, 128)(torch.zeros(1, 100, 128))
        for pos in (0, 1, 26, 63, 99):
            for i in (0, 5, 63):
                angle = torch.tensor(pos / (10000 ** (2 * i / 128)))
                torch.testing.assert_close(encoded[0, pos, 2*i], angle.sin(), atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(encoded[0, pos, 2*i+1], angle.cos(), atol=1e-5, rtol=1e-5)

    def test_plain_embedding_and_attention_receive_gradients_during_real_updates(self):
        config = MetaformerConfig(patch_count=10, embedding_dim=16, attention_heads=2,
                    head_dim=8, encoder_layers=2, mlp_dim=32, geometry_dim=7,
                    angle_symmetry='square_periodic')
        model = InverseMetaformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        spectra, target = torch.rand(3, 1000), torch.rand(3, 7)
        before = model.patch_embedding.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction, maps = model(spectra, return_attention=True)
            self.assertEqual(prediction.shape, (3, 7))
            self.assertEqual(maps[0].shape, (3, 2, 10, 10))
            geometry_mse(prediction, target, config.angle_symmetry).backward()
            for block in model.blocks:
                for gradient in block.attention.qkv.weight.grad.chunk(3):
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.norm()), 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.patch_embedding.weight))

    def test_legacy_plain_metadata_is_compatible_but_resnet_resume_is_rejected(self):
        current = {'model': asdict(MetaformerConfig()), 'learning_rate': .0003}
        legacy = deepcopy(current)
        legacy['model'].pop('spectral_frontend')
        self.assertTrue(matching_training_config(legacy, current))
        legacy['model'].update(resnet_channels=32, resnet_blocks=3, resnet_kernel_size=7)
        self.assertTrue(matching_training_config(legacy, current))
        residual = deepcopy(legacy)
        residual['model']['spectral_frontend'] = 'resnet1d'
        self.assertFalse(matching_training_config(residual, current))
        with self.assertRaisesRegex(ValueError, 'removed'):
            MetaformerConfig.from_dict(residual['model'])


if __name__ == '__main__':
    unittest.main()
