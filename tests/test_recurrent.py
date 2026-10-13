"""CPU protocol checks with real PIDSR R2P and a small stochastic task surrogate.

These tests do not establish PolarFree performance or CUDA memory feasibility.
"""
import copy
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment as run


class SmallTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior = nn.Conv2d(12, 12, 1).requires_grad_(False)
        self.net_g = nn.Conv2d(12, 3, 1)
        self.features = nn.ModuleDict({f"T{i}": nn.Conv2d(12, c, 1) for i, c in enumerate((96, 96, 192), 1)})
        self.noises = []

    @property
    def model(self):
        return self

    def forward(self, polar, rgb, noise=None):
        noise = torch.randn_like(polar) * .001
        self.noises.append(noise.detach().clone())
        x = self.prior(polar) + noise
        f = {name: head(torch.nn.functional.avg_pool2d(x, 2**i))
             for i, (name, head) in enumerate(self.features.items())}
        return torch.sigmoid(self.net_g(x) + rgb), f


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = run.read_yaml(ROOT / 'config.yaml')

    def setUp(self):
        run.set_seed(2026)
        c = self.config
        self.model = run.JointR2P(Path(c['paths']['reference_r2p']), Path(c['paths']['pretrained_r2p']), 48, .1, (96, 96, 192))
        self.model.enable_recurrence((96, 96, 192), 48, True, (.5, 1.))
        self.model.protocol_id = 'synthetic'
        self.task = SmallTask()
        self.batch = {'polar': torch.rand(1, 12, 16, 16), 'condition': torch.rand(1, 3, 16, 16),
                      'official_rgb': torch.rand(1, 3, 16, 16), 'target': torch.rand(1, 3, 16, 16),
                      'sample_id': ['synthetic/a'], 'scene': ['a']}

    def test_round1_parity_and_zero_gate_control(self):
        m, b = self.model, self.batch
        with torch.no_grad():
            ref, _ = m.r2p(b['condition'], b['polar'], return_features=True)
            a = run.recurrent_forward(m, self.task, b)
            self.assertTrue(torch.equal(ref, a.first.refined_polar))
            self.assertTrue(torch.equal(self.task.noises[-2], self.task.noises[-1]))
            torch.manual_seed(500)
            a = run.recurrent_forward(m, self.task, b)
            m.feedback_enabled = False
            torch.manual_seed(500)
            control = run.recurrent_forward(m, self.task, b)
            self.assertTrue(torch.equal(a.refined_polar, control.refined_polar))
            self.assertTrue(torch.equal(a.t2, control.t2))
            self.assertFalse(torch.equal(a.first.refined_polar, a.refined_polar))

    def test_gate_activation_and_full_feedback_gradient(self):
        m = self.model
        out, _, tl, al = run.forward_losses(m, self.task, nn.MSELoss(), self.batch)
        (tl + .01 * al).backward()
        self.assertEqual(run.gradient_norm(m.feedback_adapter.parameters()), 0)
        self.assertGreater(abs(m.feedback_scale.grad.item()), 0)
        self.assertTrue(all(p.grad is None for p in self.task.prior.parameters()))
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        opt.step()
        m.zero_grad(set_to_none=True)
        self.task.zero_grad(set_to_none=True)
        out, _, tl, al = run.forward_losses(m, self.task, nn.MSELoss(), self.batch)
        grad = torch.autograd.grad(out.loss2, tuple(out.first_task_features.values()), retain_graph=True)
        self.assertTrue(all(g.abs().sum() > 0 for g in grad))
        (tl + .01 * al).backward()
        self.assertGreater(run.gradient_norm(m.feedback_adapter.parameters()), 0)
        self.assertGreater(run.gradient_norm(m.r2p.parameters()), 0)

    def test_no_feedback_has_no_adapter_gradient(self):
        self.model.feedback_enabled = False
        _, _, tl, al = run.forward_losses(self.model, self.task, nn.MSELoss(), self.batch)
        (tl + .01 * al).backward()
        self.assertIsNone(self.model.feedback_scale.grad)
        self.assertTrue(all(p.grad is None for p in self.model.feedback_adapter.parameters()))

    def test_alignment_targets_detach(self):
        projected = {f'P{i}': torch.randn(1, c, 8, 8, requires_grad=True) for i, c in enumerate((96, 96, 192), 1)}
        targets = {f'T{i}': torch.randn(1, c, 4, 4, requires_grad=True) for i, c in enumerate((96, 96, 192), 1)}
        run.feature_alignment_loss(projected, targets).backward()
        self.assertTrue(all(v.grad is None for v in targets.values()))
        self.assertTrue(all(v.grad is not None for v in projected.values()))

    def test_checkpoint_round_trip(self):
        opt = torch.optim.Adam(self.model.parameters())
        state = run.checkpoint_payload(self.model, self.task, opt, 200, .01)
        clone = copy.deepcopy(self.model)
        clone.load_state_dict(state['r2p_and_projections'], strict=True)
        self.assertEqual(state['schema_version'], 2)
        self.assertTrue(state['feedback_enabled'])
        with torch.no_grad():
            torch.manual_seed(1)
            a = run.recurrent_forward(self.model, self.task, self.batch)
            torch.manual_seed(1)
            b = run.recurrent_forward(clone, self.task, self.batch)
            self.assertTrue(torch.equal(a.t2, b.t2))

    def test_actual_clamped_modification(self):
        before = torch.ones(1, 12, 2, 2)
        stats = run.polar_stats(before, (before + .1).clamp(0, 1))
        self.assertEqual(stats['mean_abs'], 0)

    def test_preflight_and_training_resume_integration(self):
        # Exercise actual runner logic with a small task; no real experiment files.
        run.add_source_path(self.config)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            config = copy.deepcopy(self.config)
            config['paths']['output_root'] = tmp
            config['optimization']['validation_steps'] = [0, 1, 2]
            identity = run.protocol_id(config)
            def components(c, device):
                model = copy.deepcopy(self.model)
                task = copy.deepcopy(self.task)
                task.features.requires_grad_(False)
                model.protocol_id = identity
                return model, task, nn.MSELoss(), None
            calibration = {'lambda_feature_alignment': .01, 'protocol_id': identity}
            def calibrate(c, device):
                run.atomic_json(Path(tmp) / 'config/alignment_calibration.json', calibration)
                return calibration
            def scheduled(c, base, end_step, start_step=0):
                return [self.batch] * (end_step - start_step)
            with patch.object(run, 'build_components', components), patch.object(run, 'calibrate', calibrate), \
                 patch.object(run, 'scheduled_loader', scheduled), patch.object(run, 'proxy_loader', lambda *args: [self.batch]):
                report = run.run_preflight(config, torch.device('cpu'))
                self.assertTrue(report['passed'])
                first = run.run_training(config, torch.device('cpu'), 1, False)
                second = run.run_training(config, torch.device('cpu'), 2, True)
                self.assertEqual(second['timing']['start_step'], 1)
                self.assertIn('round2', first['validation'])
                self.assertIn('iteration_gain', second['validation'])
                self.assertTrue((Path(tmp) / 'checkpoints/step_2.pt').exists())


if __name__ == '__main__':
    unittest.main()
