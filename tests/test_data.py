"""Regression tests for cue offsets, label alignment, and training-only stats."""

import copy
import unittest
import warnings

import numpy as np
import torch

from data.bci2a import (
    BCI2aDataset,
    BranchNormalizer,
    SessionEpochs,
    align_channel_indices,
    epoch_session,
    stratified_split,
)
from utils import load_config


class DataTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["data"]["notch_frequency_hz"] = None
        self.signal = np.broadcast_to(np.arange(5000, dtype=np.float32), (22, 5000)).copy()
        self.events = np.array([[0, 768], [500, 769], [2000, 768], [2500, 772]])

    def test_cue_relative_half_open_interval_and_context(self):
        self.config["model"]["sinc"]["context_samples"] = 125
        epochs = epoch_session(self.signal, self.events, self.config, 1, "T")
        self.assertEqual(epochs.x_spatial.shape, (2, 22, 1000))
        self.assertEqual(epochs.x_frequency.shape, (2, 22, 1250))
        np.testing.assert_array_equal(epochs.x_spatial[0, 0], np.arange(625, 1625))
        np.testing.assert_array_equal(epochs.x_frequency[0, 0], np.arange(500, 1750))
        np.testing.assert_array_equal(epochs.labels, [0, 3])
        batch = BCI2aDataset(epochs)[0]
        self.assertEqual(tuple(batch["label"].shape), ())
        self.assertEqual(batch["label"].dtype, torch.long)
        self.assertEqual(batch["subject"], "A01")

    def test_external_labels_aligned_before_artifact_rejection(self):
        events = self.events.copy()
        events[[1, 3], 1] = 783
        events = np.vstack((events, [0, 1023]))
        self.config["data"]["drop_artifacts"] = True
        epochs = epoch_session(self.signal, events, self.config, 1, "E", [2, 4])
        np.testing.assert_array_equal(epochs.labels, [3])
        np.testing.assert_array_equal(epochs.cue_samples, [2500])
        self.assertEqual(epochs.dropped_trials[0]["reason"], "artifact_1023")
        with self.assertRaisesRegex(ValueError, "classlabel count"):
            epoch_session(self.signal, events, self.config, 1, "E", [4])

    def test_run_crossing_and_nonfinite_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_boundary_crossing"):
            epoch_session(
                self.signal, np.vstack((self.events, [1000, 32766])), self.config, 1, "T"
            )
        signal = self.signal.copy()
        signal[0, 700] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite_signal"):
            epoch_session(signal, self.events, self.config, 1, "T")
        self.config["data"]["invalid_epoch_policy"] = "drop"
        epochs = epoch_session(signal, self.events, self.config, 1, "T")
        np.testing.assert_array_equal(epochs.labels, [3])

    def test_context_outside_record_rejected(self):
        self.config["model"]["sinc"]["context_samples"] = 126
        with self.assertRaisesRegex(ValueError, "context_out_of_recording"):
            epoch_session(self.signal, [[0, 769]], self.config, 1, "T")

    def test_channel_alignment_named_and_official_mixed_names(self):
        names = self.config["data"]["channel_names"]
        self.assertEqual(align_channel_indices(list(reversed(names)), names), list(reversed(range(22))))
        official = [
            "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
            "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
            "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertEqual(align_channel_indices(official + ["EOG-left"], names), list(range(22)))
        with self.assertRaises(ValueError):
            align_channel_indices([f"unknown-{index}" for index in range(22)], names)

    def test_statistics_use_only_training_subset_and_roundtrip(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(8, 22, 1000)).astype(np.float32)
        values[6:] += 1000
        epochs = SessionEpochs(
            values, values * 2, np.arange(8) % 4, "A01", "T", np.arange(8)
        )
        normalizer = BranchNormalizer().fit(epochs, np.arange(6))
        expected_mean = values[:6].mean(axis=(0, 2), dtype=np.float64)
        np.testing.assert_allclose(normalizer.statistics["spatial"]["mean"][0, :, 0], expected_mean)
        original_state = copy.deepcopy(normalizer.state_dict())
        normalizer.transform(values[6:], "spatial")
        self.assertEqual(normalizer.state_dict(), original_state)
        restored = BranchNormalizer.from_state_dict(original_state)
        np.testing.assert_array_equal(
            restored.transform(values, "frequency"), normalizer.transform(values, "frequency")
        )
        train_values = normalizer.transform(values[:6], "spatial")
        np.testing.assert_allclose(train_values.mean(axis=(0, 2)), 0, atol=1e-6)

    def test_stratified_split_is_disjoint_and_reproducible(self):
        labels = np.repeat(np.arange(4), 10)
        train, validation = stratified_split(labels, 0.2, 42)
        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertEqual(set(train).intersection(validation), set())
        np.testing.assert_array_equal(np.bincount(labels[validation]), [2, 2, 2, 2])
        for expected, actual in zip((train, validation), stratified_split(labels, 0.2, 42)):
            np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
