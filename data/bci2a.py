"""Event-based BCI IV 2a GDF reading, aligned branches and training-only statistics.

The target interval is cue-relative [0.5, 4.5) seconds by default. Spatial
notching is applied independently to each extracted trial, avoiding filtering
across run separators or train/validation trials. Frequency inputs remain raw.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import filtfilt, iirnotch
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
    "P2", "POz",
]
CUE_CODES = {769, 770, 771, 772, 783}


@dataclass
class SessionEpochs:
    x_spatial: np.ndarray
    x_frequency: np.ndarray
    labels: np.ndarray
    subject: str
    session: str
    cue_samples: np.ndarray
    dropped_trials: list = field(default_factory=list)


def resolve_directory(value, field_name, required=True):
    """Resolve configured relative directories against the project root."""
    if value is None or not str(value).strip():
        if required:
            raise ValueError(f"Set paths.{field_name} in config/bci2a.yaml before reading data.")
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_subject(subject):
    text = str(subject).strip().upper()
    match = re.fullmatch(r"(?:A)?0?([1-9])(?:[TE])?", text)
    if match is None:
        raise ValueError(f"Invalid BCI 2a subject {subject!r}. Use 1..9 or A01..A09.")
    return f"A{int(match.group(1)):02d}"


def _channel_key(name):
    key = str(name).strip()
    key = re.sub(r"^EEG[\s:_-]*", "", key, flags=re.IGNORECASE)
    return key.casefold()


def align_channel_indices(raw_names, channel_names, aliases=None, allow_official_generic=True):
    """Map named electrodes or the complete known official GDF generic layout.

    aliases maps raw file names to canonical electrode names. The official
    22-channel generic sequence is accepted only as a complete sequence in its
    original file order. Arbitrary unnamed 22-channel files are rejected.
    """
    aliases = aliases or {}
    wanted = [_channel_key(name) for name in channel_names]
    if len(set(wanted)) != len(wanted):
        raise ValueError("Configured channel names must be unique.")
    mapped = [_channel_key(aliases.get(name, name)) for name in raw_names]
    if all(name in mapped for name in wanted):
        if any(mapped.count(name) != 1 for name in wanted):
            raise ValueError("Electrode names become ambiguous after channel alias mapping.")
        return [mapped.index(name) for name in wanted]
    official = ["EEG-Fz"] + [f"EEG-{index}" for index in range(1, 22)]
    numbered = [f"EEG-{index}" for index in range(22)]
    mixed = [
        "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
        "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
        "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
    ]
    prefix = [str(name).casefold() for name in raw_names[:22]]
    layouts = [[name.casefold() for name in layout] for layout in (official, numbered, mixed)]
    if allow_official_generic and prefix in layouts:
        lookup = {name.casefold(): index for index, name in enumerate(CANONICAL_CHANNELS)}
        if all(name in lookup for name in wanted):
            warnings.warn(
                "Mapping the validated official GDF EEG sequence to the documented "
                "BCI 2a electrode order. Use channel_aliases for renamed files.",
                UserWarning,
            )
            return [lookup[name] for name in wanted]
    missing = [name for name, key in zip(channel_names, wanted) if key not in mapped]
    raise ValueError(f"Unable to identify electrodes {missing}. Configure data.channel_aliases.")


def notch_trial(x, sampling_rate, frequency, quality=30.0):
    """Zero-phase IIR notch per trial and electrode, with no bandpass."""
    if frequency is None or float(frequency) == 0:
        return np.asarray(x, dtype=np.float32).copy()
    if not 0 < frequency < sampling_rate / 2 or quality <= 0:
        raise ValueError("Notch frequency must be inside (0, Nyquist), with positive quality.")
    numerator, denominator = iirnotch(frequency, quality, fs=sampling_rate)
    return filtfilt(numerator, denominator, x, axis=-1).astype(np.float32)


def _integer_samples(seconds, sampling_rate, name):
    samples = float(seconds) * sampling_rate
    if not np.isclose(samples, round(samples), atol=1e-7, rtol=0):
        raise ValueError(f"{name} must fall on an exact sampling-grid boundary.")
    return int(round(samples))


def epoch_session(signal, events, config, subject, session, external_labels=None):
    """Extract aligned trial arrays from raw [channel,sample] EEG and event codes.

    events has columns [sample, code], in sample coordinates of signal. External
    labels are official one-based classlabel entries, aligned to ALL cue events
    before artifact rejection. This function never infers cue times from order.
    """
    settings = config["data"]
    fs = float(settings["sampling_rate"])
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[0] != int(settings["n_channels"]):
        raise ValueError("Signal must have shape [configured EEG channels, continuous samples].")
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 2 or not np.isfinite(events).all():
        raise ValueError("events must be a finite array with [sample, code] columns.")
    if not np.equal(events, np.round(events)).all():
        raise ValueError("Event sample indices and codes must be integers.")
    events = events.astype(np.int64)
    events = events[np.argsort(events[:, 0], kind="stable")]
    cues = events[np.isin(events[:, 1], list(CUE_CODES))]
    if len(cues) == 0 or np.any(np.diff(cues[:, 0]) <= 0):
        raise ValueError("No distinct, increasing MI cue events were found.")
    if external_labels is None:
        if np.any(cues[:, 1] == 783):
            raise ValueError("Unlabeled cue 783 requires the official MAT classlabel file.")
        labels = cues[:, 1] - 769
    else:
        labels_raw = np.asarray(external_labels).reshape(-1)
        if len(labels_raw) != len(cues):
            raise ValueError("classlabel count must equal the number of cue events before rejection.")
        if not np.isin(labels_raw, [1, 2, 3, 4]).all():
            raise ValueError("Official classlabel values must be integers in 1..4.")
        labels = labels_raw.astype(np.int64) - 1
        known = cues[:, 1] != 783
        if not np.array_equal(labels[known], cues[known, 1] - 769):
            raise ValueError("External class labels disagree with known GDF cue labels.")
    start_offset = _integer_samples(settings["epoch_start_sec"], fs, "epoch_start_sec")
    end_offset = _integer_samples(settings["epoch_end_sec"], fs, "epoch_end_sec")
    if start_offset < 0 or end_offset <= start_offset:
        raise ValueError("Epoch offsets must satisfy 0 <= start < end.")
    context_value = config["model"]["sinc"]["context_samples"]
    context = int(context_value)
    if context < 0 or context != context_value:
        raise ValueError("context_samples must be a nonnegative integer per side.")
    policy = settings.get("invalid_epoch_policy", "error")
    if policy not in {"error", "drop"}:
        raise ValueError("invalid_epoch_policy must be 'error' or 'drop'.")
    trial_starts = events[events[:, 1] == 768, 0]
    artifact_samples = events[events[:, 1] == 1023, 0]
    boundaries = events[events[:, 1] == 32766, 0]
    spatial, frequency, kept_labels, kept_cues, dropped = [], [], [], [], []
    for index, (cue, _) in enumerate(cues):
        trial_index = np.searchsorted(trial_starts, cue, side="right") - 1
        trial_start = trial_starts[trial_index] if trial_index >= 0 else cue
        next_trial_index = trial_index + 1
        trial_end = signal.shape[1]
        if next_trial_index < len(trial_starts):
            trial_end = trial_starts[next_trial_index]
        marked = np.any((artifact_samples >= trial_start) & (artifact_samples < trial_end))
        if settings.get("drop_artifacts", False) and marked:
            dropped.append({"trial_index": index, "cue_sample": int(cue), "reason": "artifact_1023"})
            continue
        start, end = int(cue + start_offset), int(cue + end_offset)
        context_start, context_end = start - context, end + context
        reason = None
        if context_start < 0 or context_end > signal.shape[1]:
            reason = "context_out_of_recording"
        elif np.any((boundaries > context_start) & (boundaries < context_end)):
            reason = "run_boundary_crossing"
        elif not np.isfinite(signal[:, context_start:context_end]).all():
            reason = "nonfinite_signal"
        if reason is not None:
            detail = {"trial_index": index, "cue_sample": int(cue), "reason": reason}
            if policy == "error":
                raise ValueError(f"Invalid {subject}/{session} epoch {detail}.")
            dropped.append(detail)
            continue
        spatial.append(notch_trial(
            signal[:, start:end], fs, settings.get("notch_frequency_hz", 50.0),
            settings.get("notch_quality", 30.0),
        ))
        frequency.append(signal[:, context_start:context_end].astype(np.float32))
        kept_labels.append(labels[index])
        kept_cues.append(cue)
    if not spatial:
        raise ValueError(f"No usable epochs remain for {subject}/{session}.")
    return SessionEpochs(
        np.stack(spatial), np.stack(frequency), np.asarray(kept_labels, dtype=np.int64),
        normalize_subject(subject), str(session).upper(), np.asarray(kept_cues, dtype=np.int64), dropped,
    )


def read_session(config, subject, session):
    """Read AxxT.gdf or AxxE.gdf and associated AxxE.mat class labels."""
    subject = normalize_subject(subject)
    session = str(session).upper()
    if session not in {"T", "E"}:
        raise ValueError("Session must be T (session 1) or E (session 2).")
    directory = resolve_directory(config["paths"].get("data_dir"), "data_dir")
    filename = directory / f"{subject}{session}.gdf"
    if not filename.is_file():
        raise FileNotFoundError(f"GDF file does not exist: {filename}")
    # Lazy import keeps model/unit tests independent of MNE user configuration.
    import mne

    raw = mne.io.read_raw_gdf(str(filename), preload=True, verbose="ERROR")
    try:
        settings = config["data"]
        if not np.isclose(raw.info["sfreq"], settings["sampling_rate"]):
            raise ValueError("GDF sampling rate differs from data.sampling_rate. Resample explicitly.")
        if len(settings["channel_names"]) != int(settings["n_channels"]):
            raise ValueError("channel_names length differs from n_channels.")
        indices = align_channel_indices(
            raw.ch_names, settings["channel_names"], settings.get("channel_aliases"),
        )
        signal = raw.get_data(picks=indices) * float(settings.get("unit_scale", 1e6))
        def numeric_event(description):
            return int(description) if description.strip().isdigit() else None
        annotated, _ = mne.events_from_annotations(raw, event_id=numeric_event, verbose="ERROR")
        events = np.column_stack((annotated[:, 0] - raw.first_samp, annotated[:, 2]))
    finally:
        raw.close()
    labels = None
    if session == "E":
        label_directory = resolve_directory(config["paths"].get("labels_dir"), "labels_dir", required=False)
        label_filename = (label_directory or directory) / f"{subject}E.mat"
        if not label_filename.is_file():
            raise FileNotFoundError(f"Official evaluation classlabel file does not exist: {label_filename}")
        label_file = loadmat(label_filename)
        if "classlabel" not in label_file:
            raise ValueError(f"Expected variable 'classlabel' in {label_filename}.")
        labels = label_file["classlabel"]
    return epoch_session(signal, events, config, subject, session, labels)


def stratified_split(labels, val_ratio, seed):
    """Split only the training session, preserving original trial indices."""
    if not 0 < float(val_ratio) < 1:
        raise ValueError("validation.val_ratio must be in (0, 1).")
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=int(seed), stratify=np.asarray(labels),
    )
    return np.sort(train_indices), np.sort(val_indices)


class BranchNormalizer:
    """Per-channel mean/std estimated exclusively from specified training trials."""

    def __init__(self, mode="train_session_stats", eps=1e-6):
        if mode not in {"train_session_stats", "none"} or eps <= 0:
            raise ValueError("Normalization requires mode train_session_stats/none and eps > 0.")
        self.mode = mode
        self.eps = float(eps)
        self.statistics = {}

    def fit(self, epochs, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("Normalization requires at least one training index.")
        for branch in ("spatial", "frequency"):
            values = getattr(epochs, f"x_{branch}")[indices]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite values encountered while fitting training statistics.")
            mean = values.mean(axis=(0, 2), keepdims=False, dtype=np.float64)[None, :, None]
            std = values.std(axis=(0, 2), keepdims=False, dtype=np.float64)[None, :, None]
            self.statistics[branch] = {
                "mean": mean.astype(np.float32),
                "std": np.maximum(std, self.eps).astype(np.float32),
            }
        return self

    def transform(self, values, branch):
        values = np.asarray(values, dtype=np.float32)
        if self.mode == "none":
            return values.copy()
        if branch not in self.statistics:
            raise RuntimeError("Fit the normalizer on training trials before transformation.")
        stats = self.statistics[branch]
        if values.ndim not in {2, 3} or values.shape[-2] != stats["mean"].shape[1]:
            raise ValueError("Normalizer expects [C,T] or [N,C,T] with matching electrode count.")
        mean, std = stats["mean"], stats["std"]
        if values.ndim == 2:
            mean, std = mean[0], std[0]
        return ((values - mean) / std).astype(np.float32)

    def state_dict(self):
        # Lists keep the checkpoint readable and compatible across NumPy versions.
        statistics = {
            branch: {name: value.tolist() for name, value in stats.items()}
            for branch, stats in self.statistics.items()
        }
        return {"mode": self.mode, "eps": self.eps, "statistics": statistics}

    @classmethod
    def from_state_dict(cls, state):
        normalizer = cls(state["mode"], state["eps"])
        normalizer.statistics = {
            branch: {name: np.asarray(value, dtype=np.float32) for name, value in stats.items()}
            for branch, stats in state["statistics"].items()
        }
        return normalizer


class BCI2aDataset(Dataset):
    """Aligned branch tensors and provenance for one subject/session."""

    def __init__(self, epochs, indices=None, normalizer=None):
        self.epochs = epochs
        self.indices = np.arange(len(epochs.labels)) if indices is None else np.asarray(indices)
        self.normalizer = normalizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        trial_index = self.indices[index]
        spatial = self.epochs.x_spatial[trial_index]
        frequency = self.epochs.x_frequency[trial_index]
        if self.normalizer is not None:
            spatial = self.normalizer.transform(spatial, "spatial")
            frequency = self.normalizer.transform(frequency, "frequency")
        return {
            "x_spatial": torch.from_numpy(np.ascontiguousarray(spatial)),
            "x_frequency": torch.from_numpy(np.ascontiguousarray(frequency)),
            "label": torch.tensor(self.epochs.labels[trial_index], dtype=torch.long),
            "subject": self.epochs.subject,
            "session": self.epochs.session,
        }
