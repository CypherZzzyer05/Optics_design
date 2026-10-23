import unittest
import torch

from metaformerpro import (MetaformerPro, MetaformerProConfig, decode_circular, import_v01_state)
from spectral_inverse import SpectralInverse, SpectralConfig, inverse_loss


class MetaformerProTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)

    def test_legacy_conversion_preserves_predictions(self):
        x = torch.rand(4, 1000)
        old = SpectralInverse(SpectralConfig(view='raw', embedding_dim=16, encoder_layers=1))
        old.fit_input_normalization(x)
        new = MetaformerPro(MetaformerProConfig(embedding_dim=16, encoder_layers=1))
        new.load_state_dict(import_v01_state(old.state_dict()), strict=True)
        old.eval(), new.eval()
        self.assertTrue(torch.equal(old(x), new(x)))

    def test_circular_wrap_and_gradient(self):
        phase = torch.tensor([[.99, .01]]) * 2 * torch.pi
        raw = torch.cat((torch.zeros(1, 5), torch.stack((phase.cos(), phase.sin()), -1).flatten(1)), -1)
        raw.requires_grad_(True)
        prediction = decode_circular(raw)
        torch.testing.assert_close(prediction[:, 5:], torch.tensor([[.99, .01]]))
        target = prediction.detach().clone()
        target[:, 5:] = torch.tensor([[.01, .99]])
        inverse_loss(prediction, target, True).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(raw.grad[:, 5:].abs().sum().item(), 0)

    def test_all_readouts_update_reload_and_token_count(self):
        x, y = torch.rand(4, 1000), torch.rand(4, 7)
        for readout, angle, overlap in [('global', 'circular', False), ('queries', 'sigmoid', False),
                                        ('queries', 'circular', False), ('queries', 'circular', True)]:
            with self.subTest(readout=readout, angle=angle, overlap=overlap):
                config = MetaformerProConfig(readout=readout, angle_head=angle, overlap=overlap,
                                             embedding_dim=16, encoder_layers=1, dropout=0)
                model = MetaformerPro(config)
                model.fit_input_normalization(x)
                counts = []
                hook = model.tokenizer.register_forward_hook(lambda m, a, out: counts.append(out.shape[-1]))
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                before = model.tokenizer.weight.detach().clone()
                prediction = model(x)
                inverse_loss(prediction, y, True).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                hook.remove()
                self.assertEqual(counts, [100])
                self.assertFalse(torch.equal(before, model.tokenizer.weight))
                copy = MetaformerPro(config)
                copy.load_state_dict(model.state_dict())
                model.eval(), copy.eval()
                torch.testing.assert_close(model(x), copy(x), atol=0, rtol=0)

    def test_selection_rejects_lower_total_error_with_worse_dimensions(self):
        from train_metaformerpro import selection_from_validation
        records = []
        for name, mse, length, dx in [('pro_v01', .02, .01, 7.), ('eligible', .018, .01, 7.),
                                       ('bad_dx', .015, .01, 9.), ('bad_lengths', .014, .02, 6.)]:
            for seed in (42, 43, 44):
                records.append({'variant': name, 'seed': seed, 'parameters': 100,
                    'validation': {'mirror_equivalent': {'mse': mse, 'length_mse': length,
                                                          'mae_nm': [1., 1., 1., dx, 1.]}}})
        plan = {'variants': dict.fromkeys(('pro_v01', 'eligible', 'bad_dx', 'bad_lengths')), 'selection': 'test rule'}
        selected = selection_from_validation(records, plan)
        self.assertEqual(selected['selected'], 'eligible')
        self.assertFalse(next(r for r in selected['ranking'] if r['variant'] == 'bad_dx')['eligible'])


if __name__ == '__main__':
    unittest.main()
