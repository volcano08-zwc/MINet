"""Verify every documented interface, masks, and probability axes."""

import copy
import unittest

import torch

from model.bci2a import DualBranchMINet
from utils import load_config


class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.config = load_config()
        cls.model = DualBranchMINet(cls.config).eval()
        cls.x = torch.randn(2, 22, 1000)
        with torch.no_grad():
            cls.logits, cls.aux = cls.model(cls.x, cls.x, return_aux=True)

    def test_documented_shapes(self):
        self.assertEqual(tuple(self.logits.shape), (2, 4))
        expected = {
            "power_features": (2, 5, 22, 16),
            "dct_descriptors": (2, 110),
            "psca_weights": (2, 110),
            "psca_features": (2, 110, 16),
            "time_tokens": (2, 16, 64),
            "electrode_features": (2, 22, 64),
            "graph_features": (2, 22, 64),
            "roi_assignment": (2, 22, 3),
            "region_tokens": (2, 3, 64),
            "cross_time_tokens": (2, 16, 64),
            "cross_region_tokens": (2, 3, 64),
            "cross_attention_time_to_region": (2, 4, 16, 3),
            "cross_attention_region_to_time": (2, 4, 3, 16),
            "fusion_gate": (2, 64),
            "fused_features": (2, 64),
        }
        for name, shape in expected.items():
            with self.subTest(tensor=name):
                self.assertEqual(tuple(self.aux[name].shape), shape)

    def test_auxiliary_tensors_are_finite(self):
        def check(value, name):
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.isfinite(value).all(), name)
            elif isinstance(value, dict):
                for key, item in value.items():
                    check(item, f"{name}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    check(item, f"{name}[{index}]")

        check(self.aux, "aux")

    def test_roi_probabilities_and_cross_attention(self):
        assignment = self.aux["roi_assignment"]
        torch.testing.assert_close(assignment.sum(dim=-1), torch.ones(2, 22))
        self.assertTrue((assignment >= 0).all())
        for name in ("cross_attention_time_to_region", "cross_attention_region_to_time"):
            weights = self.aux[name]
            torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))

    def test_graph_mask_and_probability_axis(self):
        adjacency = self.model.gat.adjacency
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertTrue(adjacency.diag().all())
        attention = self.aux["gat_attention"]
        self.assertEqual(tuple(attention.shape), (2, 2, 4, 22, 22))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones_like(attention[..., 0]))
        self.assertTrue((attention[..., ~adjacency] == 0).all())

    def test_cross_attention_uses_original_tokens_in_both_directions(self):
        time_tokens = self.aux["time_tokens"]
        region_tokens = self.aux["region_tokens"]
        cross = self.model.cross_attention
        with torch.no_grad():
            expected_time = cross.time_to_region(time_tokens, region_tokens)
            expected_region = cross.region_to_time(region_tokens, time_tokens)
        torch.testing.assert_close(expected_time, self.aux["cross_time_tokens"])
        torch.testing.assert_close(expected_region, self.aux["cross_region_tokens"])

    def test_default_forward_matches_auxiliary_forward(self):
        with torch.no_grad():
            logits = self.model(self.x, self.x)
        torch.testing.assert_close(logits, self.logits)

    def test_batch_one_and_real_filter_context(self):
        config = copy.deepcopy(self.config)
        config["model"]["sinc"]["context_samples"] = 125
        model = DualBranchMINet(config).eval()
        with torch.no_grad():
            logits, aux = model(
                torch.randn(1, 22, 1000), torch.randn(1, 22, 1250), return_aux=True
            )
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertEqual(tuple(aux["power_features"].shape), (1, 5, 22, 16))

    def test_wrong_context_length_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.x, self.x[..., :-1])


if __name__ == "__main__":
    unittest.main()
