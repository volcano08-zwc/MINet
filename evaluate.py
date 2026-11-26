"""Re-evaluate a selected checkpoint using its saved training normalization."""

import argparse
import copy
from pathlib import Path

import numpy as np

from data.bci2a import BCI2aDataset, BranchNormalizer, read_session, resolve_directory
from model.bci2a import DualBranchMINet
from plot_results import save_confusion_matrices
from train_2a import choose_device, load_checkpoint, make_loader, run_epoch, save_json


def evaluate_checkpoint(checkpoint_path, data_dir=None, labels_dir=None, device_name="auto", output_dir=None):
    checkpoint_path = Path(checkpoint_path)
    device = choose_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = copy.deepcopy(checkpoint["config"])
    if data_dir is not None:
        config["paths"]["data_dir"] = data_dir
    if labels_dir is not None:
        config["paths"]["labels_dir"] = labels_dir
    config["training"]["device"] = str(device)
    normalizer = BranchNormalizer.from_state_dict(checkpoint["normalizer"])
    model = DualBranchMINet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epochs = read_session(config, checkpoint["subject"], "E")
    loader = make_loader(BCI2aDataset(epochs, normalizer=normalizer), config)
    metrics, labels, predictions = run_epoch(
        model, loader, device, n_classes=int(config["model"]["num_classes"]),
    )
    result = {
        "version": checkpoint.get("version", config.get("experiment", {}).get("version", "v1")),
        "experiment_variant": config.get("experiment", {}).get("variant", "baseline"),
        "augmentation": config.get("augmentation", {"method": "none"}),
        "subject": checkpoint["subject"], "seed": checkpoint["seed"],
        "best_epoch": checkpoint["best_epoch"], **metrics,
        "session_2_dropped_trials": epochs.dropped_trials,
    }
    destination = checkpoint_path.parent / "evaluation"
    if output_dir is not None:
        destination = resolve_directory(output_dir, "output_dir")
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "test_metrics.json", result)
    np.savez_compressed(
        destination / "test_predictions.npz", labels=labels, predictions=predictions,
        cue_samples=epochs.cue_samples,
    )
    save_confusion_matrices(
        metrics["confusion_matrix"], destination,
        title=f"{checkpoint['subject']} / seed {checkpoint['seed']}",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint, args.data_dir, args.labels_dir, args.device, args.output_dir,
    )
    print(f"{metrics['subject']} seed={metrics['seed']} accuracy={metrics['accuracy']:.4f} "
          f"kappa={metrics['kappa']} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
