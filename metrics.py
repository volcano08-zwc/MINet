"""Four-class evaluation and explicit subject/seed aggregation."""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


METRIC_NAMES = ("accuracy", "kappa", "macro_f1")


def classification_metrics(labels, predictions, n_classes=4):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0 or labels.shape != predictions.shape:
        raise ValueError("Metrics require matching nonempty one-dimensional label arrays.")
    classes = np.arange(n_classes)
    if not np.isin(labels, classes).all() or not np.isin(predictions, classes).all():
        raise ValueError("Labels and predictions must be valid class indices.")
    kappa = float(cohen_kappa_score(labels, predictions, labels=classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "kappa": kappa if np.isfinite(kappa) else None,
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "n_trials": int(len(labels)),
    }


def _metric_statistics(rows):
    output = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=float)
        output[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "count": int(len(values)),
        }
    return output


def summarize_results(rows):
    """Sample std (ddof=1). Average seeds per subject before overall statistics."""
    if not rows:
        raise ValueError("No completed experiment results were supplied.")
    per_subject = {
        subject: _metric_statistics([row for row in rows if row["subject"] == subject])
        for subject in sorted({row["subject"] for row in rows})
    }
    per_seed = {
        str(seed): _metric_statistics([row for row in rows if row["seed"] == seed])
        for seed in sorted({row["seed"] for row in rows})
    }
    subject_means = [
        {metric: statistics[metric]["mean"] for metric in METRIC_NAMES}
        for statistics in per_subject.values()
    ]
    return {
        "std_ddof": 1,
        "single_observation_std": None,
        "overall_unit": "subject mean after averaging the completed seeds for that subject",
        "per_subject_across_seeds": per_subject,
        "per_seed_across_subjects": per_seed,
        "overall_across_subject_means": _metric_statistics(subject_means),
        "n_completed_runs": len(rows),
    }
