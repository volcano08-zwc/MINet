"""Check the DCT descriptor independently against SciPy's orthonormal DCT."""

import unittest

import numpy as np
from scipy.fft import dct
import torch

from model.bci2a import DCTPSCA
from utils import load_config


class DCTPSCATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.module = DCTPSCA(110, 16, load_config()["model"]["psca"])

    def test_reference_coefficients_and_product_sum(self):
        x = torch.randn(3, 110, 16, requires_grad=True)
        output, weights = self.module(x, return_weights=True)
        descriptors = self.module.describe(x)
        orders = np.resize(np.array([1, 2, 3]), 110)
        reference = dct(x.detach().numpy(), type=2, norm="ortho", axis=-1)
        selected = reference[:, np.arange(110), orders]
        np.testing.assert_allclose(descriptors.detach().numpy(), selected, rtol=2e-5, atol=3e-6)
        np.testing.assert_array_equal(np.bincount(orders)[1:], [37, 37, 36])
        self.assertEqual(tuple(weights.shape), (3, 110))
        torch.testing.assert_close(output, x * weights.unsqueeze(-1) + weights.unsqueeze(-1))
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_nonzero_orders_ignore_constant_time_offset(self):
        x = torch.randn(2, 110, 16)
        torch.testing.assert_close(
            self.module.describe(x + 3.0), self.module.describe(x), rtol=2e-5, atol=6e-6
        )


if __name__ == "__main__":
    unittest.main()
