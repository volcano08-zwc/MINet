"""Verify plot data, missing-class normalization and historical-result redraws."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from matplotlib.axes import Axes
import numpy as np
from PIL import Image

from plot_results import main, save_confusion_matrices, save_learning_curves


class ResultPlotTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix=".test_plots_", dir=self.project_root)
        self.directory = Path(self.temporary.name).resolve()
        self.assertEqual(self.directory.parent, self.project_root)
        self.addCleanup(self.temporary.cleanup)
        self.history = [
            {"epoch": 1, "train": {"accuracy": 0.25, "loss": 1.5},
             "validation": {"accuracy": 0.25, "loss": 1.4}},
            {"epoch": 2, "train": {"accuracy": 0.5, "loss": 1.1},
             "validation": {"accuracy": 0.5, "loss": 1.05}},
            {"epoch": 3, "train": {"accuracy": 0.75, "loss": 0.7},
             "validation": {"accuracy": 0.5, "loss": 1.2}},
        ]
        self.matrix = np.array([[2, 1, 0, 1], [0, 0, 0, 0], [0, 1, 3, 0], [2, 1, 1, 0]])

    def test_learning_curves_plot_recorded_values(self):
        calls = []
        original_plot = Axes.plot

        def record_plot(axis, *args, **kwargs):
            if len(args) >= 2:
                calls.append((np.asarray(args[0]), np.asarray(args[1])))
            return original_plot(axis, *args, **kwargs)

        with patch.object(Axes, "plot", new=record_plot):
            files = save_learning_curves(self.history, self.directory, best_epoch=2)
        self.assertEqual({Path(filename).name for filename in files}, {
            "accuracy_curve.png", "accuracy_curve.svg", "loss_curve.png", "loss_curve.svg", "history.csv",
        })
        expected = ([25, 50, 75], [25, 50, 50], [1.5, 1.1, 0.7], [1.4, 1.05, 1.2])
        for values in expected:
            self.assertTrue(any(
                np.array_equal(x, [1, 2, 3]) and np.allclose(y, values)
                for x, y in calls if x.shape == y.shape == (3,)
            ), f"Recorded values missing from plot: {values}")
        with (self.directory / "history.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["train_accuracy"]) for row in rows], [0.25, 0.5, 0.75])
        for name in ("accuracy_curve", "loss_curve"):
            with Image.open(self.directory / f"{name}.png") as image:
                self.assertGreaterEqual(image.width, 1200)
                image.verify()
            ET.parse(self.directory / f"{name}.svg")

    def test_confusion_counts_and_zero_row_normalization(self):
        files = save_confusion_matrices(self.matrix, self.directory)
        self.assertEqual({Path(filename).name for filename in files}, {
            "confusion_matrix.png", "confusion_matrix.svg", "confusion_matrix_normalized.png",
            "confusion_matrix_normalized.svg", "confusion_matrix.csv",
        })
        with (self.directory / "confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        counts = [[int(value) for value in row[-4:]] for row in rows[1:]]
        np.testing.assert_array_equal(counts, self.matrix)
        body = (self.directory / "confusion_matrix_normalized.svg").read_text(encoding="utf-8")
        for value in ("75.0%", "0.0%", "True label", "Predicted label"):
            self.assertIn(value, body)
        self.assertNotIn("nan%", body.lower())
        with Image.open(self.directory / "confusion_matrix.png") as image:
            image.verify()

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            save_learning_curves([], self.directory)
        with self.assertRaises(ValueError):
            save_learning_curves(self.history, self.directory, best_epoch=99)
        self.history[0]["train"]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            save_learning_curves(self.history, self.directory)
        for matrix in ([[1.5, 0], [0, 2]], [[1, -1], [0, 2]], [[1, 2, 3], [4, 5, 6]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                save_confusion_matrices(matrix, self.directory)

    def test_redraw_cli_supports_history_only_and_metrics_only(self):
        history_directory = self.directory / "history_only"
        history_directory.mkdir()
        (history_directory / "history.json").write_text(json.dumps(self.history), encoding="utf-8")
        metrics_directory = self.directory / "metrics_only"
        metrics_directory.mkdir()
        (metrics_directory / "test_metrics.json").write_text(
            json.dumps({"subject": "A01", "seed": 42, "confusion_matrix": self.matrix.tolist()}),
            encoding="utf-8",
        )
        for directory, expected in (
            (history_directory, "accuracy_curve.png"), (metrics_directory, "confusion_matrix.png"),
        ):
            with patch.object(sys, "argv", ["plot_results.py", "--run-dir", str(directory)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((directory / expected).is_file())


if __name__ == "__main__":
    unittest.main()
