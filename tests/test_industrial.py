from __future__ import annotations
import argparse
import csv
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lipnet.utils import set_seed
from scripts.reproduce_paper import aggregate, command, METRICS
from scripts.run_mvtec_feature_lipnet_fcn_sweep import _train_models
from lipnet.checkpoint import resume_epoch


class Backbone(torch.nn.Module):
    def forward(self, x):
        return {'layer2': x}


class ResumeTests(unittest.TestCase):
    def run_training(self, root, stop=-1, resume=False):
        set_seed(42)
        model = torch.nn.Conv2d(3, 3, 1)
        x = torch.rand(7, 3, 8, 8)
        args = argparse.Namespace(lr=.001, weight_decay=0., epochs=3, feature_normalize=True,
                                  l1_weight=0., resume=resume, epoch_checkpoint=root/'epoch.pt',
                                  checkpoint_fingerprint='test-v1', diagnostic_stop_after_epoch=stop)
        loader = DataLoader(x, batch_size=3, shuffle=True)
        _train_models({'layer2': model}, Backbone(), loader, torch.device('cpu'), args, .1, 1.)
        return model.state_dict(), torch.get_rng_state()

    def test_interrupted_training_equals_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            direct, rng = self.run_training(root/'direct')
            with self.assertRaisesRegex(RuntimeError, 'Diagnostic interruption'):
                self.run_training(root/'resume', stop=1)
            resumed, resumed_rng = self.run_training(root/'resume', resume=True)
            for k in direct:
                torch.testing.assert_close(direct[k], resumed[k], rtol=0, atol=0)
            torch.testing.assert_close(rng, resumed_rng, rtol=0, atol=0)
            model = torch.nn.Conv2d(3,3,1)
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                resume_epoch({'layer2': model}, torch.optim.Adam(model.parameters()), torch.device('cpu'), root/'resume/epoch.pt', 'other')


class MatrixTests(unittest.TestCase):
    def test_macro_is_per_seed_and_missing_categories_fail(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for seed in (42,43):
                p = Path(d)/str(seed);p.mkdir();paths.append(p)
                with (p/'summary_arch_sweep.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=['category',*METRICS]);w.writeheader()
                    for c,x in [('a',.5 if seed==42 else .7),('b',.7 if seed==42 else .9)]:
                        w.writerow({'category':c,**{k:x for k in METRICS}})
            with self.assertRaisesRegex(ValueError, 'seed matrix'):
                aggregate({'main': paths[:1]}, ['a','b'], [42,43])
            records,summary=aggregate({'main':paths},['a','b'],[42,43])
            self.assertAlmostEqual(summary['main']['metrics']['image_auc']['mean'],.7)
            self.assertAlmostEqual(summary['main']['metrics']['image_auc']['sample_sd'],np.sqrt(.02))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                aggregate({'main':paths},['a','b','missing'],[42,43])

    def test_fixed_paper_protocol(self):
        a=argparse.Namespace(experiment='mvtec',data_root=Path('data'),categories=['bottle'],device='cpu',resume=False)
        cmd=command(a,'main',42,Path('runs'))
        for flag,value in [('--epochs','10'),('--image_resize','512'),('--batch_size','8'),('--pro_eval_size','256'),('--backbone_weights','default')]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertIn('--save_artifacts',cmd)
        self.assertIn('--no-feature_inhibition',command(a,'no_inhibition',42,Path('runs')))

if __name__=='__main__':
    unittest.main()
