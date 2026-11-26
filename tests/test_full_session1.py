"""Verify full-session training, fixed epochs, and best-test checkpoint selection."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from data.bci2a import SessionEpochs
from metrics import classification_metrics
import train_2a as training
from train_2a import load_checkpoint
from utils import load_config


class FullSessionTests(unittest.TestCase):
    def test_full_training_and_earliest_maximum_test_epoch(self):
        torch.set_num_threads(2)
        config = copy.deepcopy(load_config())
        config["training"].update({"patience": 1, "batch_size": 8, "num_workers": 0})
        rng = np.random.default_rng(42)
        values = rng.normal(size=(20, 22, 1000)).astype(np.float32)
        evaluation_values = rng.normal(size=(8, 22, 1000)).astype(np.float32) + 100
        sessions = {
            "T": SessionEpochs(values, values.copy(), np.tile(np.arange(4), 5),
                               "A01", "T", np.arange(20)),
            "E": SessionEpochs(evaluation_values, evaluation_values.copy(), np.tile(np.arange(4), 2),
                               "A01", "E", np.arange(8)),
        }
        calls, test_states, test_predictions = [], [], []
        original_run = training.run_epoch
        correct_counts = [2, 6, 6, 1]

        def run_epoch(model, loader, device, optimizer=None, n_classes=4):
            calls.append((loader.dataset.epochs.session, optimizer is not None, len(loader.dataset)))
            if optimizer is not None:
                return original_run(model, loader, device, optimizer, n_classes)
            self.assertEqual(loader.dataset.epochs.session, "E")
            labels = sessions["E"].labels.copy()
            predictions = (labels + 1) % 4
            correct = correct_counts[len(test_states)]
            predictions[:correct] = labels[:correct]
            metrics = classification_metrics(labels, predictions)
            metrics["loss"] = 1.0
            test_states.append(model.classifier[-1].weight.detach().cpu().clone())
            test_predictions.append(predictions.copy())
            return metrics, labels, predictions

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix=".test_full_session_", dir=project_root) as name:
            root = Path(name)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            arguments = ["train_full_session1.py", "--config", str(config_path), "--subjects", "1",
                         "--seeds", "42", "--epochs", "4", "--device", "cpu", "--output-dir", str(root)]
            with patch("sys.argv", arguments), \
                    patch.object(training, "read_session", side_effect=lambda cfg, subject, session: sessions[session]), \
                    patch.object(training, "run_epoch", side_effect=run_epoch), \
                    contextlib.redirect_stdout(io.StringIO()):
                training.main()
            self.assertEqual(calls, [("T", True, 20), ("E", False, 8)] * 4)
            directory = root / "A01" / "seed_42"
            result = json.loads((directory / "test_metrics.json").read_text())
            self.assertEqual(result["epochs_run"], 4)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["accuracy"], 0.75)
            self.assertTrue(result["test_used_for_model_selection"])
            self.assertEqual(result["version"], "v1")
            checkpoint = load_checkpoint(directory / "best_model.pt")
            self.assertEqual(checkpoint["best_epoch"], 2)
            self.assertEqual(checkpoint["version"], "v1")
            self.assertEqual(checkpoint["config"]["experiment"]["version"], "v1")
            torch.testing.assert_close(checkpoint["model_state_dict"]["classifier.3.weight"], test_states[1])
            self.assertEqual(checkpoint["split"]["train_indices"], list(range(20)))
            self.assertEqual(checkpoint["split"]["validation_indices"], [])
            expected_mean = values.mean(axis=(0, 2), dtype=np.float64)
            recorded = np.array(checkpoint["normalizer"]["statistics"]["spatial"]["mean"])[0, :, 0]
            np.testing.assert_allclose(recorded, expected_mean, atol=1e-6)
            with np.load(directory / "test_predictions.npz") as saved:
                np.testing.assert_array_equal(saved["predictions"], test_predictions[1])
                self.assertEqual(saved["version"].item(), "v1")
            history = json.loads((directory / "history.json").read_text())
            self.assertEqual(len(history), 4)
            self.assertTrue(all(row["version"] == "v1" for row in history))
            self.assertTrue(all("test" in row and "validation" not in row for row in history))
            with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(row["test_accuracy"]) for row in rows], [0.25, 0.75, 0.75, 0.125])
            self.assertTrue(all(row["version"] == "v1" for row in rows))
            svg = (directory / "accuracy_curve.svg").read_text(encoding="utf-8")
            self.assertIn("Test", svg)
            self.assertIn("v1", svg)
            self.assertNotIn("Validation", svg)
            self.assertTrue((directory / "confusion_matrix.png").is_file())
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["n_completed_runs"], 1)
            self.assertEqual(summary["version"], "v1")
            self.assertTrue(summary["test_used_for_model_selection"])
            with (root / "results.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["version"], "v1")


if __name__ == "__main__":
    unittest.main()
