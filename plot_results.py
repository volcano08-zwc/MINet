"""Save learning curves and confusion matrices on headless training servers."""

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


CLASS_NAMES = ["Left hand", "Right hand", "Feet", "Tongue"]
PLOT_STYLE = {"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 11}


@contextmanager
def _atomic_output(filename):
    """Replace one artifact only after its complete temporary file is written."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=filename.parent, prefix="." + filename.stem + ".",
        suffix=filename.suffix, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(figure, output_dir, name):
    paths = []
    for extension in ("png", "svg"):
        filename = Path(output_dir) / f"{name}.{extension}"
        with _atomic_output(filename) as temporary:
            figure.savefig(temporary, format=extension, dpi=300, bbox_inches="tight")
        paths.append(filename)
    return paths


def _save_csv(filename, header, rows):
    filename = Path(filename)
    with _atomic_output(filename) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    return filename


def save_learning_curves(history, output_dir, best_epoch=None, title=None):
    """Plot actual training/validation history, retaining raw 0–1 accuracy in CSV."""
    if not history:
        raise ValueError("Learning curves require at least one completed epoch.")
    try:
        epochs = np.asarray([row["epoch"] for row in history], dtype=float)
        values = np.asarray([
            [row["train"]["accuracy"], row["validation"]["accuracy"],
             row["train"]["loss"], row["validation"]["loss"]]
            for row in history
        ], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("History requires epoch and train/validation accuracy and loss.") from error
    if not np.isfinite(epochs).all() or not np.isfinite(values).all():
        raise ValueError("Epochs, accuracy and loss must all be finite.")
    if np.any(epochs < 1) or np.any(epochs != np.floor(epochs)) or np.any(np.diff(epochs) <= 0):
        raise ValueError("Epoch numbers must be positive, strictly increasing integers.")
    if np.any(values[:, :2] < 0) or np.any(values[:, :2] > 1) or np.any(values[:, 2:] < 0):
        raise ValueError("Accuracy must be within [0, 1] and loss must be nonnegative.")
    if best_epoch is None:
        best_epoch = int(epochs[np.argmin(values[:, 3])])
    if best_epoch not in epochs:
        raise ValueError("The selected best_epoch must be present in the saved history.")
    best_index = int(np.flatnonzero(epochs == best_epoch)[0])
    rows = [[int(epoch), *row.tolist()] for epoch, row in zip(epochs, values)]
    paths = [_save_csv(
        Path(output_dir) / "history.csv",
        ["epoch", "train_accuracy", "validation_accuracy", "train_loss", "validation_loss"], rows,
    )]
    with matplotlib.rc_context(PLOT_STYLE):
        for name, columns, ylabel, scale in (
            ("accuracy_curve", (0, 1), "Accuracy (%)", 100),
            ("loss_curve", (2, 3), "Cross-entropy loss", 1),
        ):
            figure, ax = plt.subplots(figsize=(7.2, 4.8))
            try:
                marker = "o" if len(epochs) == 1 else None
                ax.plot(epochs, values[:, columns[0]] * scale, color="#2563a6",
                        label="Training", linewidth=1.8, marker=marker)
                ax.plot(epochs, values[:, columns[1]] * scale, color="#d66a24",
                        label="Validation", linewidth=1.8, marker=marker)
                ax.axvline(best_epoch, color="#697586", linestyle="--", linewidth=1.1,
                           label=f"Selected epoch {best_epoch}")
                ax.scatter([best_epoch], [values[best_index, columns[1]] * scale],
                           color="#d66a24", edgecolors="white", s=45, zorder=4)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(ylabel)
                ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
                upper = 100 if scale == 100 else max(0.1, float(values[:, columns].max()) * 1.08)
                ax.set_ylim(0, upper)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(axis="y", alpha=0.22)
                ax.set_axisbelow(True)
                ax.set_title(f"{title}\n{ylabel} by epoch" if title else f"{ylabel} by epoch")
                ax.legend(fontsize=9)
                figure.tight_layout()
                paths.extend(_save_figure(figure, output_dir, name))
            finally:
                plt.close(figure)
    return paths


def save_confusion_matrices(confusion_matrix, output_dir, class_names=None, title=None):
    """Save counts and true-class row percentages with explicit axis labels."""
    matrix = np.asarray(confusion_matrix)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The confusion matrix must be a nonempty square matrix.")
    if matrix.dtype.kind not in "biuf" or not np.isfinite(matrix).all():
        raise ValueError("Confusion matrix counts must be finite numbers.")
    if np.any(matrix < 0) or np.any(matrix != np.floor(matrix)) or np.any(matrix >= 2**63):
        raise ValueError("Confusion matrix counts must be nonnegative int64 values.")
    matrix = matrix.astype(np.int64)
    count = len(matrix)
    if class_names is None:
        class_names = CLASS_NAMES if count == 4 else [f"Class {index}" for index in range(count)]
    names = [str(name) for name in class_names]
    if len(names) != count or any(not name.strip() for name in names) or len(set(names)) != count:
        raise ValueError("Provide one unique nonempty class name per matrix row.")
    totals = matrix.sum(axis=1, keepdims=True, dtype=np.float64)
    percentages = np.divide(matrix * 100.0, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
    paths = [_save_csv(
        Path(output_dir) / "confusion_matrix.csv", ["True / Predicted", *names],
        [[name, *row.tolist()] for name, row in zip(names, matrix)],
    )]
    with matplotlib.rc_context(PLOT_STYLE):
        for name, shown, normalized in (
            ("confusion_matrix", matrix, False),
            ("confusion_matrix_normalized", percentages, True),
        ):
            figure, ax = plt.subplots(figsize=(7.4, 6.4))
            try:
                maximum = 100 if normalized else max(1, int(matrix.max()))
                image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=maximum, origin="upper")
                colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                colorbar.set_label("Within-class percentage (%)" if normalized else "Number of trials")
                ax.set_xticks(np.arange(count), names, rotation=30, ha="right")
                ax.set_yticks(np.arange(count), names)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                heading = "Row-normalized confusion matrix" if normalized else "Confusion matrix"
                ax.set_title(f"{title}\n{heading}" if title else heading)
                for row in range(count):
                    for column in range(count):
                        value = shown[row, column]
                        label = f"{value:.1f}%" if normalized else str(int(value))
                        ax.text(column, row, label, ha="center", va="center",
                                color="white" if value > maximum * 0.55 else "#172b4d", fontsize=11)
                figure.tight_layout()
                paths.extend(_save_figure(figure, output_dir, name))
            finally:
                plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Redraw figures from saved experiment results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.expanduser().resolve()
    history_file = directory / "history.json"
    metrics_file = directory / "test_metrics.json"
    if not history_file.is_file() and not metrics_file.is_file():
        parser.error("run-dir must contain history.json or test_metrics.json.")
    metrics = {}
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    title = directory.name
    if "subject" in metrics and "seed" in metrics:
        title = f"{metrics['subject']} | seed {metrics['seed']}"
    paths = []
    if history_file.is_file():
        history = json.loads(history_file.read_text(encoding="utf-8"))
        paths.extend(save_learning_curves(history, directory, metrics.get("best_epoch"), title))
    if metrics_file.is_file():
        if "confusion_matrix" not in metrics:
            raise ValueError("test_metrics.json must contain the saved confusion_matrix.")
        paths.extend(save_confusion_matrices(metrics["confusion_matrix"], directory, title=title))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
