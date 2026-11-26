"""BCI2a training with selectable protocols and training-only EEG augmentation.

The default full-session1 protocol runs every requested epoch and selects the
highest Session 2 test accuracy. --protocol validation enables the v1 split.
"""

import argparse
import copy
import csv
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.augmentation import AugmentedTrainingLoader, TrainingAugmenter
from data.bci2a import (
    BCI2aDataset, BranchNormalizer, normalize_subject, read_session,
    resolve_directory, stratified_split,
)
from metrics import classification_metrics, summarize_results
from model.bci2a import DualBranchMINet
from plot_results import PLOT_STYLE, _save_csv, _save_figure, save_confusion_matrices, save_learning_curves
import matplotlib
from matplotlib import pyplot as plt
from utils import load_config


def save_json(filename, content):
    filename = Path(filename)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=filename.parent,
            prefix=filename.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, filename)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(filename, content):
    filename = Path(filename)
    with tempfile.NamedTemporaryFile(
        dir=filename.parent, prefix=filename.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(content, temporary)
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(filename, device="cpu"):
    return torch.load(filename, map_location=device, weights_only=True)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this PyTorch installation.")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, device, optimizer=None, n_classes=4):
    """Run a training/validation/test pass without retaining computation graphs."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all, predictions_all = [], []
    loss_function = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for batch in loader:
            spatial = batch["x_spatial"].to(device)
            frequency = batch["x_frequency"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(spatial, frequency)
            loss = loss_function(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite logits or loss. Inspect input signals and model parameters.")
            if training:
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite gradient detected before the optimizer update.")
                optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    metrics = classification_metrics(labels_all, predictions_all, n_classes)
    metrics["loss"] = total_loss / len(labels_all)
    return metrics, np.asarray(labels_all), np.asarray(predictions_all)


def make_loader(dataset, config, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]), generator=generator,
        pin_memory=choose_device(config["training"]["device"]).type == "cuda",
    )


def train_subject(config, subject, seed, output_root):
    """Select the best model using validation loss, then read session 2 once."""
    subject = normalize_subject(subject)
    seed_everything(seed)
    training = config["training"]
    epochs_max, patience = int(training["max_epochs"]), int(training["patience"])
    if epochs_max <= 0 or patience <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("max_epochs, patience and batch_size must be positive.")
    if int(training["num_workers"]) < 0 or float(training["learning_rate"]) <= 0:
        raise ValueError("num_workers must be nonnegative and learning_rate must be positive.")
    device = choose_device(training["device"])
    # A blank data path fails before any result directory is created.
    train_epochs = read_session(config, subject, "T")
    train_indices, val_indices = stratified_split(
        train_epochs.labels, config["validation"]["val_ratio"], seed,
    )
    normalizer = BranchNormalizer(**config["normalization"]).fit(train_epochs, train_indices)
    augmenter = None
    if config.get("augmentation", {}).get("method", "none") != "none":
        from dataclasses import replace
        pool = replace(train_epochs, x_spatial=train_epochs.x_spatial[train_indices],
                       x_frequency=train_epochs.x_frequency[train_indices], labels=train_epochs.labels[train_indices],
                       cue_samples=train_epochs.cue_samples[train_indices])
        augmenter = TrainingAugmenter(pool, config, normalizer, seed)
    train_loader = make_loader(BCI2aDataset(train_epochs, train_indices, None if augmenter else normalizer), config, True, seed)
    if augmenter is not None:
        train_loader = AugmentedTrainingLoader(train_loader, augmenter)
    val_loader = make_loader(BCI2aDataset(train_epochs, val_indices, normalizer), config)
    model = DualBranchMINet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    run_directory = Path(output_root) / subject / f"seed_{seed}"
    run_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_filename = run_directory / "best_model.pt"
    split_metadata = {
        "train_indices": train_indices.tolist(), "validation_indices": val_indices.tolist(),
        "train_cue_samples": train_epochs.cue_samples[train_indices].tolist(),
        "validation_cue_samples": train_epochs.cue_samples[val_indices].tolist(),
        "session_1_dropped_trials": train_epochs.dropped_trials,
        "session_1_total_kept": len(train_epochs.labels),
        "model_selection": "minimum session-1 validation cross-entropy loss",
        "normalization_fit": "session-1 training subset only, independently for each branch",
    }
    save_json(run_directory / "config.json", config)
    save_json(run_directory / "split.json", split_metadata)
    save_json(run_directory / "normalization.json", normalizer.state_dict())
    best_loss, stale_epochs, history = float("inf"), 0, []
    n_classes = int(config["model"]["num_classes"])
    for epoch in range(1, epochs_max + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, n_classes)
        validation_metrics, _, _ = run_epoch(model, val_loader, device, n_classes=n_classes)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        save_json(run_directory / "history.json", history)
        print(
            f"{subject} seed={seed} epoch={epoch}/{epochs_max} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_accuracy={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f}", flush=True,
        )
        if validation_metrics["loss"] < best_loss:
            best_loss, stale_epochs = validation_metrics["loss"], 0
            save_checkpoint(checkpoint_filename, {
                "model_state_dict": model.state_dict(), "config": config,
                "normalizer": normalizer.state_dict(), "subject": subject, "seed": int(seed),
                "best_epoch": epoch, "validation_metrics": validation_metrics,
                "split": split_metadata,
            })
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    selected = load_checkpoint(checkpoint_filename, device)
    model.load_state_dict(selected["model_state_dict"])
    plot_title = f"{subject} / seed {seed}"
    save_learning_curves(history, run_directory, best_epoch=selected["best_epoch"], title=plot_title)
    # Evaluation data is first read only after all model-selection decisions.
    test_epochs = read_session(config, subject, "E")
    test_loader = make_loader(BCI2aDataset(test_epochs, normalizer=normalizer), config)
    test_metrics, labels, predictions = run_epoch(model, test_loader, device, n_classes=n_classes)
    np.savez_compressed(
        run_directory / "test_predictions.npz", labels=labels, predictions=predictions,
        cue_samples=test_epochs.cue_samples,
    )
    result = {
        "subject": subject, "seed": int(seed), "best_epoch": selected["best_epoch"],
        "epochs_run": len(history), **test_metrics,
        "session_2_dropped_trials": test_epochs.dropped_trials,
    }
    save_json(run_directory / "test_metrics.json", result)
    save_confusion_matrices(test_metrics["confusion_matrix"], run_directory, title=plot_title)
    print(f"{subject} seed={seed} test_accuracy={result['accuracy']:.4f}", flush=True)
    return result


@contextmanager
def _summary_lock(output_root):
    """Serialize result rescans. OS locks are released if a writer exits early."""
    with (Path(output_root) / ".summary.lock").open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + 30
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for the output summary lock.")
                    time.sleep(0.1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_results(output_root, results, expected_runs):
    with _summary_lock(output_root):
        stored_results = []
        for filename in sorted(Path(output_root).glob("A*/seed_*/test_metrics.json")):
            with filename.open("r", encoding="utf-8") as stream:
                stored_results.append(json.load(stream))
        summary = summarize_results(stored_results)
        summary["expected_runs_in_this_invocation"] = expected_runs
        summary["completed_runs_in_this_invocation"] = len(results)
        summary["complete"] = len(results) == expected_runs
        save_json(Path(output_root) / "summary.json", summary)
        columns = ("subject", "seed", "best_epoch", "epochs_run", "accuracy", "kappa", "macro_f1", "n_trials", "loss")
        filename = Path(output_root) / "results.csv"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=filename.parent,
                prefix=filename.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(stored_results)
            os.replace(temporary, filename)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


SELECTION = "maximum session-2 test accuracy across all requested epochs; earliest tie"


def save_test_curves(history, directory, best_epoch, title):
    """Save the actual train/test series with explicit test labels."""
    epochs = [row["epoch"] for row in history]
    _save_csv(Path(directory) / "history.csv",
              ["epoch", "train_accuracy", "test_accuracy", "train_loss", "test_loss", "version"],
              [[row["epoch"], row["train"]["accuracy"], row["test"]["accuracy"],
                row["train"]["loss"], row["test"]["loss"], row["version"]] for row in history])
    with matplotlib.rc_context(PLOT_STYLE):
        for metric, scale, label in (("accuracy", 100, "Accuracy (%)"), ("loss", 1, "Cross-entropy loss")):
            figure, axis = plt.subplots(figsize=(7.2, 4.8))
            try:
                for partition, color in (("train", "#2563a6"), ("test", "#d66a24")):
                    axis.plot(epochs, [row[partition][metric] * scale for row in history],
                              label=partition.capitalize(), color=color,
                              marker="o" if len(history) == 1 else None)
                axis.axvline(best_epoch, color="#697586", linestyle="--",
                             label=f"Highest test Acc at epoch {best_epoch}")
                axis.set(xlabel="Epoch", ylabel=label,
                         title=f"{title}\nFull Session 1 / test-accuracy selection")
                axis.set_xlim(0.5, epochs[-1] + 0.5)
                axis.set_ylim(bottom=0)
                if metric == "accuracy":
                    axis.set_ylim(0, 100)
                axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                axis.grid(axis="y", alpha=0.22)
                axis.legend(fontsize=9)
                figure.tight_layout()
                _save_figure(figure, directory, f"{metric}_curve")
            finally:
                plt.close(figure)


def train_full_subject(config, subject, seed, output_root):
    """Use Session 1 statistics and train batches, evaluating Session 2 each epoch."""
    subject = normalize_subject(subject)
    seed_everything(seed)
    training = config["training"]
    version = config.get("experiment", {}).get("version", "v1")
    epochs_max = int(training["max_epochs"])
    if epochs_max < 1 or int(training["batch_size"]) < 1:
        raise ValueError("Epochs and batch size must be positive.")
    device = choose_device(training["device"])
    directory = Path(output_root) / subject / f"seed_{seed}"
    if directory.exists():
        raise FileExistsError(f"Run directory already exists: {directory}. Choose another output directory.")
    training_epochs = read_session(config, subject, "T")
    indices = np.arange(len(training_epochs.labels))
    normalizer = BranchNormalizer(**config["normalization"]).fit(training_epochs, indices)
    augmentation = config.get("augmentation", {"method": "none"})
    augmenter = None
    if augmentation["method"] != "none":
        augmenter = TrainingAugmenter(training_epochs, config, normalizer, seed)
    test_epochs = read_session(config, subject, "E")
    train_dataset = BCI2aDataset(training_epochs, normalizer=None if augmenter else normalizer)
    train_loader = make_loader(train_dataset, config, True, seed)
    if augmenter is not None:
        train_loader = AugmentedTrainingLoader(train_loader, augmenter)
    test_loader = make_loader(BCI2aDataset(test_epochs, normalizer=normalizer), config)
    model = DualBranchMINet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"version": version, "name": "full_session1_test_accuracy_selection", "model_selection": SELECTION,
                "test_used_for_model_selection": True, "validation_enabled": False,
                "early_stopping_enabled": False, "normalization_fit": "full session 1 only",
                "augmentation": augmentation}
    split = {**protocol, "train_indices": indices.tolist(), "validation_indices": [],
             "train_cue_samples": training_epochs.cue_samples.tolist(), "validation_cue_samples": [],
             "session_1_total_kept": len(indices), "session_1_dropped_trials": training_epochs.dropped_trials}
    save_json(directory / "config.json", config)
    save_json(directory / "split.json", split)
    save_json(directory / "normalization.json", {"version": version, **normalizer.state_dict()})
    history, best_accuracy, best_epoch = [], -1.0, 0
    best_metrics, best_labels, best_predictions = None, None, None
    n_classes = int(config["model"]["num_classes"])
    print(f"[{version}] {subject} augmentation={augmentation} train_trials={len(indices)} "
          f"train_steps_per_epoch={len(train_loader)}", flush=True)
    for epoch in range(1, epochs_max + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, n_classes)
        test_metrics, labels, predictions = run_epoch(model, test_loader, device, n_classes=n_classes)
        history.append({"version": version, "epoch": epoch, "train": train_metrics, "test": test_metrics})
        save_json(directory / "history.json", history)
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = test_metrics["accuracy"], epoch
            best_metrics = copy.deepcopy(test_metrics)
            best_labels, best_predictions = labels.copy(), predictions.copy()
            save_checkpoint(directory / "best_model.pt", {
                "model_state_dict": model.state_dict(), "config": config,
                "normalizer": normalizer.state_dict(), "subject": subject, "seed": int(seed),
                "best_epoch": best_epoch, "test_metrics": best_metrics, "split": split,
                "protocol": protocol, "version": version,
            })
        print(f"[{version}] {subject} seed={seed} epoch={epoch}/{epochs_max} "
              f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} "
              f"test_loss={test_metrics['loss']:.6f} test_acc={test_metrics['accuracy']:.4f} "
              f"best_test_acc={best_accuracy:.4f} best_epoch={best_epoch}", flush=True)
    result = {"subject": subject, "seed": int(seed), "best_epoch": best_epoch,
              "epochs_run": len(history), **best_metrics, **protocol,
              "session_2_dropped_trials": test_epochs.dropped_trials}
    save_json(directory / "test_metrics.json", result)
    np.savez_compressed(directory / "test_predictions.npz", labels=best_labels,
                        predictions=best_predictions, cue_samples=test_epochs.cue_samples, version=version)
    title = f"{version} / {subject} / seed {seed} / {augmentation['method']} / highest test Acc"
    save_test_curves(history, directory, best_epoch, title)
    save_confusion_matrices(best_metrics["confusion_matrix"], directory, title=title)
    return result


def save_version_summary(root, results, expected_runs):
    """Attach the run version to the shared JSON and CSV summaries."""
    write_results(root, results, expected_runs)
    with _summary_lock(root):
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted(Path(root).glob("A*/seed_*/test_metrics.json"))]
        versions = sorted({row.get("version", "unversioned") for row in records})
        summary_path = Path(root) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["versions"] = versions
        if len(versions) == 1:
            summary["version"] = versions[0]
        summary["test_used_for_model_selection"] = True
        summary["model_selection"] = SELECTION
        summary["augmentations"] = sorted({row.get("augmentation", {}).get("method", "none") for row in records})
        save_json(summary_path, summary)
        columns = ["version", "augmentation", "subject", "seed", "best_epoch", "epochs_run",
                   "accuracy", "kappa", "macro_f1", "n_trials", "loss"]
        _save_csv(Path(root) / "results.csv", columns,
                  [[row.get("augmentation", {}).get("method", "none") if column == "augmentation"
                    else row.get(column) for column in columns] for row in records])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML configuration path")
    parser.add_argument("--subjects", nargs="+", default=None, help="1..9 or 'all'")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", choices=["full-session1", "validation"], default="full-session1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-version", default="v1")
    parser.add_argument("--augmentation", choices=["none", "sr", "frequency_shift"], default="none")
    parser.add_argument("--sr-segments", type=int, default=8)
    parser.add_argument("--sr-ratio", type=int, default=1)
    parser.add_argument("--frequency-probability", type=float, default=0.5)
    parser.add_argument("--frequency-max-hz", type=float, default=0.5)
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    for name in ("data_dir", "labels_dir", "output_dir"):
        if getattr(args, name) is not None:
            config["paths"][name] = getattr(args, name)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config["training"]["max_epochs"] = args.epochs
    if not args.run_version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.run_version):
        parser.error("Invalid --run-version")
    if args.sr_segments < 1 or args.sr_ratio < 1:
        parser.error("S&R segments and ratio must be positive")
    if not 0 <= args.frequency_probability <= 1 or not 0 <= args.frequency_max_hz < config["data"]["sampling_rate"] / 2:
        parser.error("Invalid frequency-shift parameters")
    config["augmentation"] = {"method": args.augmentation, "sr_segments": args.sr_segments,
        "sr_ratio": args.sr_ratio, "frequency_probability": args.frequency_probability,
        "frequency_max_hz": args.frequency_max_hz}
    config["experiment"] = {"version": args.run_version, "protocol": args.protocol}
    if args.protocol == "full-session1":
        config["validation"]["enabled"] = False
        config["training"].update({"early_stopping": False, "selection_metric": "test_accuracy"})
        if args.output_dir is None:
            config["paths"]["output_dir"] = f"output/{args.run_version}_full_session1_seed42"
            if args.augmentation != "none":
                config["paths"]["output_dir"] += "_" + args.augmentation
    subjects = args.subjects or config["training"]["subjects"]
    if subjects == ["all"]:
        subjects = list(range(1, 10))
    subjects = [normalize_subject(subject) for subject in subjects]
    seeds = args.seeds if args.seeds is not None else config["training"]["seeds"]
    if not subjects or not seeds or len(set(subjects)) != len(subjects) or len(set(seeds)) != len(seeds):
        raise ValueError("Supply nonempty, unique subject and seed selections.")
    config["training"].update({"subjects": subjects, "seeds": seeds})
    output_root = resolve_directory(config["paths"].get("output_dir"), "output_dir")
    for subject in subjects:
        for seed in seeds:
            run_directory = output_root / subject / f"seed_{seed}"
            if run_directory.exists():
                raise FileExistsError(f"Run directory already exists: {run_directory}. Select another --output-dir.")
    results = []
    expected_runs = len(subjects) * len(seeds)
    for subject in subjects:
        for seed in seeds:
            if args.protocol == "full-session1":
                results.append(train_full_subject(config, subject, seed, output_root))
                save_version_summary(output_root, results, expected_runs)
            else:
                results.append(train_subject(config, subject, seed, output_root))
                write_results(output_root, results, expected_runs)


if __name__ == "__main__":
    main()
