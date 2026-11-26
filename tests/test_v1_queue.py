"""Verify sequential execution, failure handling and local queue artifacts."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_v1_experiments as queue


class V1QueueTests(unittest.TestCase):
    def setUp(self):
        (queue.PROJECT_ROOT / "output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=".test_queue_", dir=queue.PROJECT_ROOT / "output")
        self.root = Path(self.temporary.name) / "queue"
        self.addCleanup(self.temporary.cleanup)

    def test_dry_run_does_not_write_results_or_start_training(self):
        with patch.object(queue, "run_training") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(queue.main(["--gpu", "3", "--dry-run", "--output-dir", str(self.root)]), 0)
        run.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_sequence_gpu_mapping_and_completed_comparison(self):
        events = []

        def run(command, environment, log_file):
            name = command[command.index("--experiment") + 1]
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(command[command.index("--augmentation") + 1], "frequency_shift")
            manifest = json.loads((self.root / "run_manifest.json").read_text(encoding="utf-8"))
            states = {job["experiment_variant"]: job["status"] for job in manifest["jobs"]}
            index = queue.EXPERIMENT_ORDER.index(name)
            self.assertTrue(all(states[prior] == "completed" for prior in queue.EXPERIMENT_ORDER[:index]))
            self.assertTrue(all(states[later] == "pending" for later in queue.EXPERIMENT_ORDER[index + 1:]))
            events.append(name)
            directory = Path(command[command.index("--output-dir") + 1])
            subject = directory / "A01" / "seed_42"
            subject.mkdir(parents=True)
            # Synthetic fixtures validate queue mechanics, not scientific performance.
            (subject / "test_metrics.json").write_text(json.dumps({
                "version": "v1", "experiment_variant": name, "seed": 42, "epochs_run": 200,
                "augmentation": {"method": "frequency_shift"}, "trainable_parameters": 123}), encoding="utf-8")
            (directory / "summary.json").write_text(json.dumps({"complete": True,
                "overall_across_subject_means": {"accuracy": {"mean": 0.25, "std": None}}}), encoding="utf-8")
            return 0

        with patch.object(queue, "run_training", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            code = queue.main(["--gpu", "3", "--subjects", "1", "--output-dir", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(events, list(queue.EXPERIMENT_ORDER))
        manifest = json.loads((self.root / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(all(job["status"] == "completed" for job in manifest["jobs"]))
        comparison = json.loads((self.root / "experiment_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual([row["experiment_variant"] for row in comparison], list(queue.EXPERIMENT_ORDER))

    def test_failure_stops_queue_and_preserves_log(self):
        with patch.object(queue, "run_training", return_value=7) as run, contextlib.redirect_stdout(io.StringIO()):
            code = queue.main(["--output-dir", str(self.root)])
        self.assertEqual(code, 7)
        self.assertEqual(run.call_count, 1)
        manifest = json.loads((self.root / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([job["status"] for job in manifest["jobs"]], ["failed", "pending", "pending"])
        self.assertFalse((self.root / "v1_lightweight").exists())

    def test_existing_later_experiment_prevents_any_training(self):
        (self.root / "v1_groupnorm").mkdir(parents=True)
        with patch.object(queue, "run_training") as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                queue.main(["--output-dir", str(self.root)])
        run.assert_not_called()

    def test_real_child_output_and_exit_code_are_preserved(self):
        self.root.mkdir()
        log = self.root / "child.log"
        command = [sys.executable, "-u", "-c", "print('queue-child-output'); raise SystemExit(7)"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = queue.run_training(command, os.environ.copy(), log)
        self.assertEqual(code, 7)
        self.assertIn("queue-child-output", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
