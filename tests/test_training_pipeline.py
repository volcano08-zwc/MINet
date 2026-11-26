"""Exercise normal CLI orchestration with a mocked reader and synthetic EEG."""

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
from evaluate import evaluate_checkpoint
from metrics import classification_metrics, summarize_results
import train_2a
from utils import load_config


class TrainingPipelineTests(unittest.TestCase):
    def test_cli_training_checkpoint_evaluation_and_incremental_summary(self):
        torch.set_num_threads(2)
        config = copy.deepcopy(load_config())
        config["training"].update({
            "max_epochs": 1, "patience": 1, "batch_size": 8, "device": "cpu", "num_workers": 0,
        })
        rng = np.random.default_rng(2026)
        training_values = rng.normal(size=(20, 22, 1000)).astype(np.float32)
        test_values = rng.normal(size=(8, 22, 1000)).astype(np.float32)
        sessions = {
            "T": SessionEpochs(training_values, training_values.copy(), np.tile(np.arange(4), 5),
                               "A01", "T", np.arange(20) * 2000 + 500),
            "E": SessionEpochs(test_values, test_values.copy(), np.tile(np.arange(4), 2),
                               "A01", "E", np.arange(8) * 2000 + 500),
        }
        read_order = []
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix=".test_pipeline_", dir=project_root) as directory:
            destination = Path(directory)
            self.assertEqual(destination.resolve().parent, project_root)
            config["paths"]["output_dir"] = str(destination)
            config_path = destination / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            checkpoint_path = destination / "A01" / "seed_42" / "best_model.pt"

            def read_synthetic(settings, subject, session):
                self.assertEqual(subject, "A01")
                read_order.append(session)
                if session == "E":
                    self.assertTrue(checkpoint_path.is_file(), "Session 2 read before model selection")
                return sessions[session]

            arguments = ["train_2a.py", "--protocol", "validation", "--config", str(config_path), "--subjects", "1", "--seeds", "42"]
            with patch("train_2a.read_session", side_effect=read_synthetic), \
                    patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                train_2a.main()
            self.assertEqual(read_order, ["T", "E"])
            saved = train_2a.load_checkpoint(checkpoint_path)
            self.assertEqual(saved["best_epoch"], 1)
            self.assertIn("normalizer", saved)
            self.assertEqual(len(saved["split"]["train_indices"]), 16)
            result_path = checkpoint_path.parent / "test_metrics.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for filename in (
                "accuracy_curve.png", "accuracy_curve.svg", "loss_curve.png", "loss_curve.svg",
                "history.csv", "confusion_matrix.png", "confusion_matrix.svg",
                "confusion_matrix_normalized.png", "confusion_matrix_normalized.svg", "confusion_matrix.csv",
            ):
                with self.subTest(artifact=filename):
                    self.assertGreater((checkpoint_path.parent / filename).stat().st_size, 0)
            with (checkpoint_path.parent / "history.csv").open(encoding="utf-8", newline="") as stream:
                history_rows = list(csv.DictReader(stream))
            self.assertEqual(len(history_rows), 1)
            self.assertEqual(int(history_rows[0]["epoch"]), 1)
            with patch("evaluate.read_session", return_value=sessions["E"]):
                repeated = evaluate_checkpoint(checkpoint_path, device_name="cpu")
            for name in ("accuracy", "kappa", "macro_f1", "loss", "confusion_matrix"):
                self.assertEqual(result[name], repeated[name])
            for filename in (
                "confusion_matrix.png", "confusion_matrix.svg", "confusion_matrix_normalized.png",
                "confusion_matrix_normalized.svg", "confusion_matrix.csv",
            ):
                self.assertGreater((checkpoint_path.parent / "evaluation" / filename).stat().st_size, 0)
            for name in ("labels", "predictions", "cue_samples"):
                with np.load(checkpoint_path.parent / "test_predictions.npz") as original, \
                        np.load(checkpoint_path.parent / "evaluation" / "test_predictions.npz") as evaluated:
                    np.testing.assert_array_equal(original[name], evaluated[name])

            # A later invocation must preserve previously completed runs in the total table.
            additional = dict(result, subject="A02")
            additional_directory = destination / "A02" / "seed_42"
            additional_directory.mkdir(parents=True)
            train_2a.save_json(additional_directory / "test_metrics.json", additional)
            train_2a.write_results(destination, [additional], 1)
            summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_completed_runs"], 2)
            self.assertEqual(len((destination / "results.csv").read_text(encoding="utf-8").splitlines()), 3)

    def test_metrics_have_four_classes_and_subject_weighted_summary(self):
        metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
        for name in ("accuracy", "kappa", "macro_f1"):
            self.assertEqual(metrics[name], 1.0)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (4, 4))
        rows = [
            {"subject": "A01", "seed": 42, "accuracy": 0., "kappa": 0., "macro_f1": 0.},
            {"subject": "A01", "seed": 2026, "accuracy": 1., "kappa": 1., "macro_f1": 1.},
            {"subject": "A02", "seed": 42, "accuracy": 1., "kappa": 1., "macro_f1": 1.},
        ]
        summary = summarize_results(rows)
        overall = summary["overall_across_subject_means"]["accuracy"]
        self.assertEqual(overall["mean"], 0.75)
        self.assertAlmostEqual(overall["std"], np.std([0.5, 1.0], ddof=1))


if __name__ == "__main__":
    unittest.main()
