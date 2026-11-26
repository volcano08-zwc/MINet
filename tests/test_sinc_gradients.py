"""Verify frequency initialization, gradient flow and constrained boundaries."""

import unittest

import torch

from model.bci2a import SincFilterBank
from utils import load_config


class SincTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.module = SincFilterBank(**load_config()["model"]["sinc"])

    def assert_legal_edges(self):
        edges = self.module.cutoff_edges().detach()
        self.assertTrue(torch.isfinite(edges).all())
        self.assertGreaterEqual(edges[0].item(), self.module.min_freq_hz)
        self.assertLessEqual(edges[-1].item(), self.module.max_freq_hz + 1e-5)
        self.assertTrue((edges.diff() >= self.module.min_bandwidth_hz - 1e-5).all())

    def test_initial_edges_and_preserved_length(self):
        torch.testing.assert_close(
            self.module.cutoff_edges(), torch.tensor([4.0, 8.0, 13.0, 20.0, 30.0, 40.0])
        )
        self.assertEqual(tuple(self.module(torch.randn(2, 22, 1000)).shape), (2, 5, 22, 1000))
        self.assert_legal_edges()
        kernels = self.module.filter_kernels()
        torch.testing.assert_close(kernels, kernels.flip(-1))

    def test_cutoff_gradients_and_optimizer_updates(self):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.05)
        initial_edges = self.module.cutoff_edges().detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            self.module(torch.randn(2, 22, 1000)).square().mean().backward()
            gradient = self.module.raw_gap_logits.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()
            self.assert_legal_edges()
        self.assertFalse(torch.equal(self.module.cutoff_edges().detach(), initial_edges))

    def test_extreme_parameters_still_respect_constraints(self):
        with torch.no_grad():
            self.module.raw_gap_logits.copy_(torch.tensor([-1000., 1000., -1000., 1000., -1000., 1000., -1000.]))
        self.assert_legal_edges()
        self.assertTrue(torch.isfinite(self.module(torch.randn(1, 22, 1000))).all())

    def test_invalid_even_kernel_rejected(self):
        with self.assertRaises(ValueError):
            SincFilterBank(kernel_size=250)


if __name__ == "__main__":
    unittest.main()
