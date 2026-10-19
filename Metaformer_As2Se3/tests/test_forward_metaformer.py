"""Forward gradients, physical periodicity, inference, split and valley checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forward_metaformer import ForwardMetaformer, ForwardMetaformerConfig
from forward_data import load_forward_data
from forward_metrics import spectral_metrics
from predict_forward import load_forward_checkpoint, normalize_physical_geometry
from train_metaformer import validate_training_splits
from train_forward_metaformer import configure_torch

ROOT = Path(__file__).resolve().parents[1]


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch(1)

    def model(self):
        torch.manual_seed(17)
        return ForwardMetaformer(ForwardMetaformerConfig(embedding_dim=16, attention_heads=2,
                   head_dim=8, encoder_layers=1, mlp_dim=32, dropout=0))

    def test_periodic_square_endpoints_and_no_batch_coupling(self):
        model = self.model().eval()
        x = torch.tensor([[.2,.4,.6,.3,.7,0.,.27], [.2,.4,.6,.3,.7,1.,.27]])
        with torch.no_grad():
            output = model(x)
            single = model(x[:1])
        self.assertEqual(tuple(output.shape), (2,1000))
        torch.testing.assert_close(output[0], output[1], atol=2e-6, rtol=0)
        torch.testing.assert_close(output[:1], single, atol=2e-6, rtol=0)
        self.assertTrue(torch.all((output>=0)&(output<=1)))

    def test_all_seven_inputs_have_gradients(self):
        model = self.model()
        x = torch.tensor([[.2,.4,.6,.3,.7,.19,.68]], requires_grad=True)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.all(x.grad.abs()>1e-9))

    def test_optimizer_update_and_checkpoint_reload(self):
        model = self.model()
        x = torch.rand(2,7)
        before = model.geometry_embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        self.assertGreater(float((before-model.geometry_embedding.weight).abs().max()), 0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'forward.pt'
            from dataclasses import asdict
            torch.save({'format':'forward_metaformer_v1', 'model_config':asdict(model.config),
                        'model_state':model.state_dict(), 'metadata':{}}, path)
            restored, _ = load_forward_checkpoint(path)
            torch.testing.assert_close(model.eval()(x), restored(x), atol=0, rtol=0)

    def test_real_splits_and_training_only_baseline(self):
        path = ROOT/'examples/au100_metal8_100/dataset.npz'
        datasets, ids, mean, metadata = load_forward_data(path, True)
        self.assertEqual([len(datasets[k]) for k in ['train','val','test']], [80,10,10])
        self.assertEqual(len(set(ids['train'])&set(ids['test'])), 0)
        self.assertEqual(metadata['target'], 'reflectance')
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        expected=arrays['spectra'][arrays['train_indices']].mean(0,dtype=np.float64).astype(np.float32)
        np.testing.assert_array_equal(mean,expected)
        arrays['test_indices'][0]=arrays['train_indices'][0]
        with self.assertRaises(ValueError):
            validate_training_splits(arrays)

    def test_quality_flag_is_not_implicitly_accepted(self):
        with self.assertRaises(ValueError):
            load_forward_data(ROOT/'examples/au100_metal8_100/dataset.npz')

    def test_geometry_bounds_are_not_silently_clipped(self):
        metadata={'lower_values':[1]*5+[-45,-45], 'upper_values':[2]*5+[45,45]}
        x=normalize_physical_geometry([1.5]*5+[0,0],metadata)
        np.testing.assert_allclose(x,np.full((1,7),.5))
        with self.assertRaises(ValueError):
            normalize_physical_geometry([3]*5+[0,0],metadata)

    def test_shifted_valley_and_missing_valley_are_distinguished(self):
        wave=np.linspace(700,1700,1000)
        truth=1-.8*np.exp(-.5*((wave-850)/3)**2)
        shifted=1-.8*np.exp(-.5*((wave-852)/3)**2)
        result=spectral_metrics(shifted[None],truth[None],wave)
        self.assertEqual(result['matched_valleys'],1)
        self.assertAlmostEqual(result['matched_center_mae_nm'],2.002002,places=5)
        flat=spectral_metrics(np.ones((1,1000)),truth[None],wave)
        self.assertEqual(flat['missing_valleys'],1)
        self.assertIsNone(flat['matched_center_mae_nm'])


if __name__=='__main__':
    unittest.main()
