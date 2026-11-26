"""Check aligned augmentations, sampling sources and the normal training CLI."""

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
from torch.utils.data import DataLoader
import yaml

from data.augmentation import TrainingAugmenter, frequency_shift
from data.bci2a import BCI2aDataset, BranchNormalizer, SessionEpochs, notch_trial
import train_2a
from utils import load_config


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = copy.deepcopy(load_config())
        self.config["data"]["notch_frequency_hz"] = None
        self.config["model"]["sinc"]["context_samples"] = 25
        rng = np.random.default_rng(42)
        frequency = rng.normal(size=(20, 22, 1050)).astype(np.float32)
        # Electrode offsets make it possible to detect electrode-wise donor changes.
        frequency += np.arange(22, dtype=np.float32)[None, :, None] * 10
        self.epochs = SessionEpochs(frequency[:, :, 25:1025].copy(), frequency,
                                    np.tile(np.arange(4), 5), "A01", "T", np.arange(20))
        self.normalizer = BranchNormalizer(mode="none").fit(self.epochs, np.arange(20))
        self.batch = next(iter(DataLoader(BCI2aDataset(self.epochs), batch_size=8)))

    def augmenter(self, method, **settings):
        config = copy.deepcopy(self.config)
        config["augmentation"] = {"method": method, **settings}
        return TrainingAugmenter(self.epochs, config, self.normalizer, 42)

    def test_sr_donors_share_class_time_position_and_electrodes(self):
        augmenter = self.augmenter("sr", sr_segments=8, sr_ratio=1)
        output = augmenter(self.batch)
        self.assertEqual(tuple(output["x_frequency"].shape), (16, 22, 1050))
        torch.testing.assert_close(output["x_frequency"][:8], self.batch["x_frequency"])
        np.testing.assert_array_equal(np.bincount(output["label"][8:].numpy()), [2, 2, 2, 2])
        boundaries = np.linspace(0, 1000, 9, dtype=int)
        for index in range(8, 16):
            label = int(output["label"][index])
            signal = output["x_frequency"][index].numpy()
            for segment in range(8):
                start, stop = 25 + boundaries[segment], 25 + boundaries[segment + 1]
                if segment == 0:
                    start = 0
                if segment == 7:
                    stop = 1050
                matches = [np.array_equal(signal[:, start:stop], donor[:, start:stop])
                           for donor in self.epochs.x_frequency[self.epochs.labels == label]]
                self.assertTrue(any(matches))
        torch.testing.assert_close(output["x_spatial"], output["x_frequency"][..., 25:1025])
        duplicate = self.augmenter("sr")(self.batch)
        torch.testing.assert_close(output["x_frequency"], duplicate["x_frequency"])
        following = augmenter(self.batch)
        self.assertFalse(torch.equal(output["x_frequency"][8:], following["x_frequency"][8:]))
        tail = {key: value[:1] for key, value in self.batch.items() if isinstance(value, torch.Tensor)}
        self.assertEqual(self.augmenter("sr", sr_ratio=3)(tail)["label"].shape, (4,))

    def test_frequency_shift_probability_and_shared_branch_signal(self):
        zero = self.augmenter("frequency_shift", frequency_probability=0)(self.batch)
        torch.testing.assert_close(zero["x_frequency"], self.batch["x_frequency"])
        output = self.augmenter("frequency_shift", frequency_probability=1)(self.batch)
        self.assertEqual(tuple(output["x_frequency"].shape), (8, 22, 1050))
        self.assertFalse(torch.equal(output["x_frequency"], self.batch["x_frequency"]))
        torch.testing.assert_close(output["x_spatial"], output["x_frequency"][..., 25:1025])
        torch.testing.assert_close(output["label"], self.batch["label"])
        with_notch = copy.deepcopy(self.config)
        with_notch["data"]["notch_frequency_hz"] = 50
        with_notch["augmentation"] = {"method": "frequency_shift", "frequency_probability": 1}
        shifted = TrainingAugmenter(self.epochs, with_notch, self.normalizer, 42)(self.batch)
        expected = notch_trial(shifted["x_frequency"].numpy()[..., 25:1025], 250, 50)
        np.testing.assert_array_equal(shifted["x_spatial"].numpy(), expected)

    def test_known_sinusoid_shift_and_validation(self):
        t = np.arange(2500) / 250
        signal = np.sin(2 * np.pi * 10 * t)[None].astype(np.float32)
        shifted = frequency_shift(signal, 0.5, 250)[0]
        frequencies = np.fft.rfftfreq(2500, 1 / 250)
        peak = frequencies[np.argmax(abs(np.fft.rfft(shifted)))]
        self.assertAlmostEqual(peak, 10.5, places=1)
        np.testing.assert_array_equal(frequency_shift(signal, 0, 250), signal)
        self.epochs.session = "E"
        with self.assertRaises(ValueError):
            self.augmenter("sr")
        self.epochs.session = "T"
        with self.assertRaises(ValueError):
            self.augmenter("sr", sr_segments=1001)
        with self.assertRaises(ValueError):
            self.augmenter("frequency_shift", frequency_probability=2)

    def test_both_methods_run_through_main_cli(self):
        root = Path(__file__).resolve().parents[1]
        self.config["training"].update({"device": "cpu", "num_workers": 0, "batch_size": 8})
        evaluation = copy.copy(self.epochs)
        evaluation.session = "E"
        for method, count in (("sr", 40), ("frequency_shift", 20)):
            with self.subTest(method=method), tempfile.TemporaryDirectory(prefix=".test_aug_", dir=root) as name:
                destination = Path(name)
                config_path = destination / "input.yaml"
                config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
                arguments = ["train_2a.py", "--config", str(config_path), "--subjects", "1", "--seeds", "42",
                             "--epochs", "1", "--device", "cpu", "--augmentation", method,
                             "--output-dir", str(destination)]
                with patch("sys.argv", arguments), patch.object(train_2a, "read_session",
                    side_effect=lambda cfg, subject, session: self.epochs if session == "T" else evaluation), \
                        contextlib.redirect_stdout(io.StringIO()):
                    train_2a.main()
                directory = destination / "A01" / "seed_42"
                history = json.loads((directory / "history.json").read_text())
                self.assertEqual(history[0]["train"]["n_trials"], count)
                self.assertEqual(history[0]["test"]["n_trials"], 20)
                result = json.loads((directory / "test_metrics.json").read_text())
                self.assertEqual(result["augmentation"]["method"], method)
                self.assertTrue((directory / "accuracy_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
