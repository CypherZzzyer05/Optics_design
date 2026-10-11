"""Check residual paths and independently compare attention gradients to PyTorch SDPA."""
from copy import deepcopy
from dataclasses import asdict
import unittest

import torch
from torch.nn import functional as F

from metaformer import (InverseMetaformer, MetaformerConfig, MultiHeadSelfAttention,
                        SpectralResidualBlock, geometry_mse)
from train_metaformer import matching_training_config


class ResidualAttentionTests(unittest.TestCase):
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

    def test_residual_block_starts_as_identity_and_preserves_input_gradient(self):
        block = SpectralResidualBlock(8, 7, 4)
        features = torch.randn(2, 8, 1000, requires_grad=True)
        output = block(features)
        torch.testing.assert_close(output, features, atol=0, rtol=0)
        output.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features), atol=0, rtol=0)
        self.assertGreater(float(block.norm2.weight.grad.norm()), 0)

    def test_resnet_and_attention_receive_gradients_during_real_updates(self):
        config = MetaformerConfig(patch_count=10, embedding_dim=16, attention_heads=2,
                    head_dim=8, encoder_layers=2, mlp_dim=32, spectral_frontend='resnet1d',
                    resnet_channels=8, resnet_blocks=2, resnet_kernel_size=7)
        model = InverseMetaformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        spectra, target = torch.rand(3, 1000), torch.rand(3, 5)
        before = model.spectral_frontend.blocks[0].conv1.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction, maps = model(spectra, return_attention=True)
            self.assertEqual(prediction.shape, (3, 5))
            self.assertEqual(maps[0].shape, (3, 2, 10, 10))
            geometry_mse(prediction, target).backward()
            for block in model.blocks:
                for gradient in block.attention.qkv.weight.grad.chunk(3):
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient.norm()), 0)
            optimizer.step()
        self.assertFalse(torch.equal(before, model.spectral_frontend.blocks[0].conv1.weight))

    def test_legacy_model_defaults_are_compatible_but_resnet_resume_is_rejected(self):
        current = {'model': asdict(MetaformerConfig()), 'learning_rate': .0003}
        legacy = deepcopy(current)
        for key in ('spectral_frontend', 'resnet_channels', 'resnet_blocks', 'resnet_kernel_size'):
            legacy['model'].pop(key)
        self.assertTrue(matching_training_config(legacy, current))
        residual = deepcopy(current)
        residual['model']['spectral_frontend'] = 'resnet1d'
        self.assertFalse(matching_training_config(legacy, residual))


if __name__ == '__main__':
    unittest.main()
