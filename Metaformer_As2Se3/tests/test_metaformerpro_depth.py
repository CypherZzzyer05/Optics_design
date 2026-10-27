import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from metaformerpro import MetaformerPro, MetaformerProConfig
from spectral_inverse import inverse_loss
from train_metaformerpro import load_pro


class DepthTests(unittest.TestCase):
    def test_real_width_deep_models_use_every_layer_and_reload(self):
        torch.set_num_threads(4)
        torch.manual_seed(17)
        x, target = torch.rand(2, 1000), torch.rand(2, 7)
        counts = []
        for depth in (8, 12):
            with self.subTest(depth=depth):
                config = MetaformerProConfig(encoder_layers=depth, angle_head='circular')
                model = MetaformerPro(config)
                model.fit_input_normalization(x)
                loss = inverse_loss(model(x), target, True)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(len(model.blocks), depth)
                for block in model.blocks:
                    grads = [p.grad for p in block.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                counts.append(sum(p.numel() for p in model.parameters()))
                with TemporaryDirectory() as folder:
                    path = Path(folder)/'deep.pt'
                    torch.save({'model_config': asdict(config), 'model_state': model.state_dict()}, path)
                    reloaded, _ = load_pro(path)
                    model.eval()
                    with torch.inference_mode():
                        torch.testing.assert_close(model(x), reloaded(x), atol=0, rtol=0)
                print(f'depth={depth}, parameters={counts[-1]}')
        self.assertGreater(counts[1], counts[0])


if __name__ == '__main__':
    unittest.main()
