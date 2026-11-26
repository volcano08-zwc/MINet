"""Training-only S&R and synchronized frequency shifts for both EEG branches.

Augment raw frequency inputs, derive spatial inputs with the existing notch,
then apply statistics fitted on clean Session 1. The loader wrapper executes
in the parent process so its seeded random stream is independent of workers.
"""

import numpy as np
from scipy.signal import hilbert
import torch

from data.bci2a import notch_trial


def frequency_shift(signal, shift_hz, sampling_rate):
    """Shift every electrode by the same Hz offset without reordering time."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or not np.isfinite(signal).all():
        raise ValueError("Frequency shift expects finite [electrodes, samples] EEG.")
    if sampling_rate <= 0 or not np.isfinite(shift_hz) or abs(shift_hz) >= sampling_rate / 2:
        raise ValueError("Frequency shift must be finite and smaller than Nyquist.")
    if shift_hz == 0:
        return signal.copy()
    # Reflection reduces the FFT boundary jump. Modulation keeps sample indices.
    padding = signal.shape[-1] // 2
    padded = np.pad(signal, ((0, 0), (padding, padding)), mode="reflect")
    analytic = hilbert(padded, axis=-1)
    time = (np.arange(padded.shape[-1]) - padding) / sampling_rate
    shifted = (analytic * np.exp(2j * np.pi * shift_hz * time)).real
    return shifted[:, padding:padding + signal.shape[-1]].astype(np.float32)


class TrainingAugmenter:
    """Apply one selected method using only this subject's training session."""

    def __init__(self, epochs, config, normalizer, seed):
        self.settings = dict(config.get("augmentation", {"method": "none"}))
        self.method = self.settings.get("method", "none")
        if self.method not in {"none", "sr", "frequency_shift"}:
            raise ValueError("Augmentation must be none, sr, or frequency_shift.")
        if str(epochs.session).upper() != "T":
            raise ValueError("Training augmentation sources must be Session 1 (T).")
        self.normalizer = normalizer
        self.rng = np.random.default_rng(seed)
        self.data = config["data"]
        self.context = int(config["model"]["sinc"]["context_samples"])
        self.samples = epochs.x_spatial.shape[-1]
        self.pool = epochs.x_frequency
        self.labels = epochs.labels
        expected = (len(self.labels), self.data["n_channels"], self.samples + 2 * self.context)
        if self.pool.shape != expected or not np.isfinite(self.pool).all():
            raise ValueError("Training frequency pool has invalid shape or nonfinite values.")
        self.classes = np.arange(config["model"]["num_classes"])
        self.class_indices = [np.flatnonzero(self.labels == label) for label in self.classes]
        self.segments = self.settings.get("sr_segments", 8)
        self.ratio = self.settings.get("sr_ratio", 1)
        self.probability = float(self.settings.get("frequency_probability", 0.5))
        self.max_shift_hz = float(self.settings.get("frequency_max_hz", 0.5))
        if self.method == "sr":
            if (not isinstance(self.segments, int) or not 1 <= self.segments <= self.samples
                    or not isinstance(self.ratio, int) or self.ratio < 1):
                raise ValueError("S&R segments and added-batch ratio must be positive integers.")
            if any(len(indices) == 0 for indices in self.class_indices):
                raise ValueError("S&R requires training examples from every configured class.")
        if not np.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Frequency-shift probability must be in [0, 1].")
        if not np.isfinite(self.max_shift_hz) or not 0 <= self.max_shift_hz < self.data["sampling_rate"] / 2:
            raise ValueError("Maximum frequency shift must be in [0, Nyquist).")

    def spatial_from_frequency(self, frequency):
        target = frequency[..., self.context:self.context + self.samples]
        return notch_trial(target, self.data["sampling_rate"], self.data.get("notch_frequency_hz"),
                           self.data.get("notch_quality", 30.0))

    def reconstruct(self, count):
        """Sample same-class, same-position segments, shared across electrodes."""
        cycles = np.concatenate([self.rng.permutation(self.classes)
                                 for _ in range((count + len(self.classes) - 1) // len(self.classes))])
        labels = cycles[:count]
        self.rng.shuffle(labels)
        result = np.empty((count, *self.pool.shape[1:]), dtype=np.float32)
        boundaries = np.linspace(0, self.samples, self.segments + 1, dtype=int)
        for index, label in enumerate(labels):
            donors = self.rng.choice(self.class_indices[int(label)], size=self.segments, replace=True)
            for segment, donor in enumerate(donors):
                start = self.context + boundaries[segment]
                stop = self.context + boundaries[segment + 1]
                result[index, :, start:stop] = self.pool[donor, :, start:stop]
            if self.context:
                result[index, :, :self.context] = self.pool[donors[0], :, :self.context]
                result[index, :, self.context + self.samples:] = self.pool[donors[-1], :, self.context + self.samples:]
        return result, labels.astype(np.int64)

    def __call__(self, batch):
        spatial = batch["x_spatial"].numpy().copy()
        frequency = batch["x_frequency"].numpy().copy()
        labels = batch["label"].numpy().copy()
        if self.method == "sr":
            generated, generated_labels = self.reconstruct(len(labels) * self.ratio)
            frequency = np.concatenate((frequency, generated))
            spatial = np.concatenate((spatial, self.spatial_from_frequency(generated)))
            labels = np.concatenate((labels, generated_labels))
        elif self.method == "frequency_shift":
            selected = self.rng.random(len(labels)) < self.probability
            for index in np.flatnonzero(selected):
                offset = self.rng.uniform(-self.max_shift_hz, self.max_shift_hz)
                frequency[index] = frequency_shift(frequency[index], offset, self.data["sampling_rate"])
                spatial[index] = self.spatial_from_frequency(frequency[index])
        return {
            "x_spatial": torch.from_numpy(self.normalizer.transform(spatial, "spatial")),
            "x_frequency": torch.from_numpy(self.normalizer.transform(frequency, "frequency")),
            "label": torch.from_numpy(labels),
        }


class AugmentedTrainingLoader:
    """Transform raw training batches just before the existing epoch runner."""

    def __init__(self, loader, augmenter):
        self.loader = loader
        self.augmenter = augmenter
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield self.augmenter(batch)
