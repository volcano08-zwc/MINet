"""Exercise independent v1 presets, actual training losses and checkpoint reloads."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from data.bci2a import SessionEpochs
from evaluate import evaluate_checkpoint
from model.bci2a import DualBranchMINet, V1_EXPERIMENTS, apply_v1_experiment
import train_2a
from utils import load_config


class V1ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.base = load_config()

    def test_presets_are_independent_and_legacy_configuration_still_loads(self):
        original = copy.deepcopy(self.base)
        regularized = apply_v1_experiment(self.base, "regularized")
        self.assertEqual(self.base, original)
        self.assertEqual(regularized["training"]["weight_decay"], 1e-3)
        self.assertEqual(regularized["training"]["label_smoothing"], 0.1)
        drops = [layer.p for layer in DualBranchMINet(regularized).modules() if isinstance(layer, nn.Dropout)]
        self.assertTrue(drops and all(value == 0.4 for value in drops))
        for name in ("lightweight", "groupnorm"):
            config = apply_v1_experiment(self.base, name)
            self.assertEqual(config["training"]["label_smoothing"], 0)
            self.assertEqual(config["training"]["weight_decay"], 1e-4)
            self.assertEqual(config["model"]["dropout"], 0.3)
        legacy = copy.deepcopy(self.base)
        legacy["training"].pop("label_smoothing")
        legacy["model"]["spatial"].pop("encoder_normalization")
        legacy["model"]["spatial"].pop("encoder_norm_groups")
        old_model, new_model = DualBranchMINet(legacy), DualBranchMINet(self.base)
        new_model.load_state_dict(old_model.state_dict())
        with self.assertRaises(ValueError):
            apply_v1_experiment(self.base, "unknown")
        bad = apply_v1_experiment(self.base, "groupnorm")
        bad["model"]["spatial"]["encoder_norm_groups"] = 7
        with self.assertRaisesRegex(ValueError, "GroupNorm"):
            DualBranchMINet(bad)

    def test_all_variant_shapes_gradients_and_parameter_counts(self):
        expected = {"baseline": 288264, "regularized": 288264, "lightweight": 63912, "groupnorm": 288264}
        for name in V1_EXPERIMENTS:
            with self.subTest(name=name):
                config = apply_v1_experiment(self.base, name)
                model = DualBranchMINet(config)
                values = torch.randn(2, 22, 1000)
                logits, aux = model(values, values, return_aux=True)
                width = config["model"]["d_model"]
                self.assertEqual(tuple(logits.shape), (2, 4))
                self.assertEqual(tuple(aux["time_tokens"].shape), (2, 16, width))
                self.assertEqual(tuple(aux["region_tokens"].shape), (2, 3, width))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name])
                F.cross_entropy(logits, torch.tensor([0, 1]), label_smoothing=model.label_smoothing).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                norms = list(model.electrode_encoder.modules())
                self.assertEqual(sum(isinstance(layer, nn.GroupNorm) for layer in norms), 2 if name == "groupnorm" else 0)
                self.assertEqual(sum(isinstance(layer, nn.BatchNorm1d) for layer in norms), 0 if name == "groupnorm" else 2)

    def test_label_smoothing_applies_only_to_training_loss(self):
        class FixedLogits(nn.Module):
            label_smoothing = 0.1

            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.tensor([3.0, -1.0, 0.5, 0.0]))

            def forward(self, spatial, frequency):
                return self.logits.expand(len(spatial), -1)

        model = FixedLogits()
        values, labels = torch.zeros(2, 22, 1000), torch.tensor([0, 2])
        loader = [{"x_spatial": values, "x_frequency": values, "label": labels}]
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        train, _, _ = train_2a.run_epoch(model, loader, torch.device("cpu"), optimizer)
        test, _, _ = train_2a.run_epoch(model, loader, torch.device("cpu"))
        logits = model.logits.detach().expand(2, -1)
        self.assertAlmostEqual(train["loss"], float(F.cross_entropy(logits, labels, label_smoothing=0.1)), places=6)
        self.assertAlmostEqual(test["loss"], float(F.cross_entropy(logits, labels)), places=6)
        self.assertNotAlmostEqual(train["loss"], test["loss"], places=5)

    def test_each_preset_trains_with_frequency_shift_and_reloads_for_evaluation(self):
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(12, 22, 1000)).astype(np.float32)
        x_test = rng.normal(size=(8, 22, 1000)).astype(np.float32)
        sessions = {
            "T": SessionEpochs(x_train, x_train.copy(), np.tile(np.arange(4), 3), "A01", "T", np.arange(12)),
            "E": SessionEpochs(x_test, x_test.copy(), np.tile(np.arange(4), 2), "A01", "E", np.arange(8)),
        }
        self.base["training"].update(batch_size=4, num_workers=0)
        root = Path(__file__).resolve().parents[1] / "output"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".test_v1_", dir=root) as temporary:
            destination = Path(temporary)
            config_path = destination / "config.yaml"
            config_path.write_text(yaml.safe_dump(self.base), encoding="utf-8")
            for name in V1_EXPERIMENTS[1:]:
                with self.subTest(name=name):
                    result_root = destination / name
                    args = ["train_2a.py", "--config", str(config_path), "--subjects", "1", "--seeds", "42",
                            "--epochs", "1", "--device", "cpu", "--experiment", name, "--output-dir", str(result_root)]
                    with patch("sys.argv", args), patch.object(train_2a, "read_session",
                        side_effect=lambda cfg, subject, session: sessions[session]), contextlib.redirect_stdout(io.StringIO()):
                        train_2a.main()
                    directory = result_root / "A01" / "seed_42"
                    result = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
                    self.assertEqual(result["version"], "v1")
                    self.assertEqual(result["experiment_variant"], name)
                    self.assertEqual(result["augmentation"]["method"], "frequency_shift")
                    self.assertEqual(result["training_label_smoothing"], 0.1 if name == "regularized" else 0)
                    self.assertEqual(result["test_label_smoothing"], 0)
                    with patch("evaluate.read_session", side_effect=lambda cfg, subject, session: sessions[session]):
                        evaluated = evaluate_checkpoint(directory / "best_model.pt", device_name="cpu")
                    self.assertEqual(evaluated["version"], "v1")
                    self.assertEqual(evaluated["experiment_variant"], name)
                    self.assertEqual(evaluated["accuracy"], result["accuracy"])
                    with np.load(directory / "test_predictions.npz") as selected, \
                            np.load(directory / "evaluation" / "test_predictions.npz") as reloaded:
                        np.testing.assert_array_equal(selected["predictions"], reloaded["predictions"])


if __name__ == "__main__":
    unittest.main()
