"""Trace and verify the implemented model using real BCI2a batches.

Examples from the project root
    python test/test_model_dimensions.py --subjects 1
    python test/test_model_dimensions.py --subjects all
    python test/test_model_dimensions.py --synthetic --context-samples 125

Only a training-subset batch receives one diagnostic optimizer step. Validation
and evaluation inputs are checked in eval mode using training-subset statistics.
The script writes a dimension report and does not save weights or accuracy.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data.bci2a import (
    BCI2aDataset, BranchNormalizer, SessionEpochs, normalize_subject,
    notch_trial, read_session, stratified_split,
)
from model.bci2a import DualBranchMINet
from utils import load_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    return []


def expected_interfaces(model, config, batch_size):
    """Independent shape calculations for actual module entry/exit hooks."""
    settings = config["model"]
    spatial = settings["spatial"]
    b, c, t, d = batch_size, model.n_channels, model.target_samples, settings["d_model"]
    f, r = settings["sinc"]["num_bands"], spatial["n_regions"]
    n = (t - settings["power"]["window_size"]) // settings["power"]["stride"] + 1
    tc, j = t + 2 * settings["sinc"]["context_samples"], f * c
    rules = []

    def add(name, module, inputs, outputs):
        rules.append((name, module, inputs, outputs))

    frequency = model.frequency_branch
    add("sinc", frequency.filterbank, [(b, c, tc)], [(b, f, c, tc)])
    add("crop_and_power", frequency.power, [(b, f, c, t)], [(b, f, c, n)])
    add("flatten_and_psca", frequency.psca, [(b, j, n)], [(b, j, n), (b, j)])
    add("dct_weight_mlp", frequency.psca.channel_weight, [(b, j)], [(b, j)])
    add("transpose_and_projection", frequency.projection, [(b, n, j)], [(b, n, d)])
    add("position_and_time_transformer", frequency.transformer, [(b, n, d)], [(b, n, d)])
    for index, layer in enumerate(frequency.transformer.layers):
        add(f"time_transformer.layer_{index}", layer, [(b, n, d)], [(b, n, d)])
    previous_channels, previous_length = 1, t
    for index, (channels, kernel, stride, padding) in enumerate(zip(
        spatial["encoder_channels"], spatial["encoder_kernel_sizes"],
        spatial["encoder_strides"], spatial["encoder_paddings"],
    )):
        length = (previous_length + 2 * padding - kernel) // stride + 1
        add(f"electrode_conv_{index}", model.electrode_encoder.encoder[3 * index],
            [(b * c, previous_channels, previous_length)], [(b * c, channels, length)])
        for offset, label in ((1, "batchnorm"), (2, "gelu")):
            add(f"electrode_{label}_{index}", model.electrode_encoder.encoder[3 * index + offset],
                [(b * c, channels, length)], [(b * c, channels, length)])
        previous_channels, previous_length = channels, length
    add("electrode_average_pool", model.electrode_encoder.encoder[-1],
        [(b * c, d, previous_length)], [(b * c, d, 1)])
    add("electrode_reshape", model.electrode_encoder, [(b, c, t)], [(b, c, d)])
    for index, layer in enumerate(model.gat.layers):
        require(layer.head_dim * layer.heads == d, "GAT concatenated head size differs from d_model")
        add(f"gat.layer_{index}", layer, [(b, c, d)], [(b, c, d), (b, spatial["gat_heads"], c, c)])
    add("roi", model.roi, [(b, c, d)], [(b, r, d), (b, c, r)])
    add("region_transformer", model.region_transformer, [(b, r, d)], [(b, r, d)])
    for index, layer in enumerate(model.region_transformer.encoder.layers):
        add(f"region_transformer.layer_{index}", layer, [(b, r, d)], [(b, r, d)])
    for name, module, query_length, context_length in (
        ("time_from_region", model.cross_attention.time_to_region, n, r),
        ("region_from_time", model.cross_attention.region_to_time, r, n),
    ):
        q, kv = (b, query_length, d), (b, context_length, d)
        add(f"cross.{name}.mha", module.attention, [q, kv, kv],
            [q, (b, settings["cross_attention"]["heads"], query_length, context_length)])
        for label, submodule in (("norm1", module.norm1), ("ffn", module.ffn), ("norm2", module.norm2)):
            add(f"cross.{name}.{label}", submodule, [q], [q])
    add("time_pool", model.time_pool, [(b, n, d)], [(b, d), (b, n)])
    add("region_pool", model.region_pool, [(b, r, d)], [(b, d), (b, r)])
    add("fusion_gate", model.fusion.gate, [(b, 2 * d)], [(b, d)])
    add("fusion_weighted_projection", model.fusion.projection, [(b, 2 * d)], [(b, d)])
    add("fusion_normalization", model.fusion.norm, [(b, d)], [(b, d)])
    hidden = settings["classifier"]["hidden_dim"]
    add("classifier_fc1", model.classifier[0], [(b, d)], [(b, hidden)])
    add("classifier_fc2", model.classifier[3], [(b, hidden)], [(b, settings["num_classes"])])
    return rules, (b, c, t, d, f, r, n, j)


def check_batch(model, config, batch, device, result):
    """Assert shape rules on live forward calls and auxiliary probabilities."""
    spatial = batch["x_spatial"].to(device)
    frequency = batch["x_frequency"].to(device)
    labels = batch["label"].to(device)
    b = len(labels)
    result.update({"batch_size": b, "dtype": str(spatial.dtype), "stages": [], "status": "RUNNING"})
    require(spatial.dtype == frequency.dtype == torch.float32, "EEG inputs must be float32")
    require(labels.dtype == torch.long and labels.shape == (b,), "Labels must be int64 [batch]")
    require(bool(((labels >= 0) & (labels < config["model"]["num_classes"])).all()), "Invalid label index")
    rules, dimensions = expected_interfaces(model, config, b)
    observed, handles = {}, []

    def make_hook(name, expected_inputs, expected_outputs):
        def hook(module, inputs, output):
            input_tensors, output_tensors = tensors(inputs), tensors(output)
            actual_inputs = [tuple(value.shape) for value in input_tensors]
            actual_outputs = [tuple(value.shape) for value in output_tensors]
            shapes_match = actual_inputs == expected_inputs and actual_outputs == expected_outputs
            finite = all(bool(torch.isfinite(value).all()) for value in input_tensors + output_tensors)
            passed = shapes_match and finite
            result["stages"].append({
                "name": name, "actual_inputs": actual_inputs, "expected_inputs": expected_inputs,
                "actual_outputs": actual_outputs, "expected_outputs": expected_outputs,
                "finite_values": finite,
                "status": "PASS" if passed else "FAIL",
            })
            if not passed:
                result["status"] = "FAIL"
            require(shapes_match, f"{name} shape mismatch: {actual_inputs} -> {actual_outputs}")
            require(finite, f"{name} produced nonfinite values")
            observed[name] = (input_tensors, output_tensors)
        return hook

    model.eval()
    try:
        for name, module, inputs, outputs in rules:
            handles.append(module.register_forward_hook(make_hook(name, inputs, outputs)))
        with torch.no_grad():
            logits, aux = model(spatial, frequency, return_aux=True)
        require(len(observed) == len(rules), "Some module hooks were not executed")
    finally:
        for handle in handles:
            handle.remove()
    b, c, t, d, f, r, n, j = dimensions
    cross_heads = config["model"]["cross_attention"]["heads"]
    expected_aux = {
        "sinc_cutoffs": (f, 2),
        "filtered_signal": (b, f, c, t), "power_features": (b, f, c, n),
        "dct_descriptors": (b, j), "psca_weights": (b, j), "psca_features": (b, j, n),
        "time_tokens": (b, n, d), "electrode_features": (b, c, d), "graph_features": (b, c, d),
        "gat_attention": (b, config["model"]["spatial"]["gat_layers"],
                          config["model"]["spatial"]["gat_heads"], c, c),
        "roi_assignment": (b, c, r), "raw_region_tokens": (b, r, d), "region_tokens": (b, r, d),
        "cross_time_tokens": (b, n, d), "cross_region_tokens": (b, r, d),
        "cross_attention_time_to_region": (b, cross_heads, n, r),
        "cross_attention_region_to_time": (b, cross_heads, r, n),
        "pooled_time": (b, d), "pooled_region": (b, d), "fusion_gate": (b, d), "fused_features": (b, d),
        "time_pool_weights": (b, n), "region_pool_weights": (b, r),
    }
    for name, shape in expected_aux.items():
        actual = tuple(aux[name].shape)
        finite = bool(torch.isfinite(aux[name]).all())
        passed = actual == shape and finite
        result["stages"].append({"name": f"aux.{name}", "actual_outputs": [actual],
                                  "expected_outputs": [shape], "finite_values": finite,
                                  "status": "PASS" if passed else "FAIL"})
        if not passed:
            result["status"] = "FAIL"
        require(actual == shape, f"aux.{name} shape mismatch: {actual} versus {shape}")
        require(finite, f"aux.{name} contains nonfinite values")
    require(tuple(logits.shape) == (b, config["model"]["num_classes"]), "Logits shape mismatch")
    require(all(bool(torch.isfinite(value).all()) for value in tensors(aux) + [logits]), "Nonfinite auxiliary value")
    torch.testing.assert_close(observed["flatten_and_psca"][0][0], aux["power_features"].flatten(1, 2))
    torch.testing.assert_close(observed["transpose_and_projection"][0][0], aux["psca_features"].transpose(1, 2))
    weights = aux["psca_weights"].unsqueeze(-1)
    torch.testing.assert_close(aux["psca_features"], aux["power_features"].flatten(1, 2) * weights + weights)
    for name in ("roi_assignment", "gat_attention", "cross_attention_time_to_region",
                 "cross_attention_region_to_time", "time_pool_weights", "region_pool_weights"):
        probability = aux[name]
        require(bool((probability >= 0).all()), f"{name} contains negative probability")
        torch.testing.assert_close(probability.sum(-1), torch.ones_like(probability[..., 0]), atol=1e-5, rtol=1e-5)
    adjacency = model.gat.adjacency
    require(bool(torch.equal(adjacency, adjacency.T) and adjacency.diag().all()), "Invalid symmetric/self-loop graph")
    require(bool((aux["gat_attention"][..., ~adjacency] == 0).all()), "GAT attention escaped its KNN mask")
    for name, query, context in (("time_from_region", "time_tokens", "region_tokens"),
                                 ("region_from_time", "region_tokens", "time_tokens")):
        inputs = observed[f"cross.{name}.mha"][0]
        torch.testing.assert_close(inputs[0], aux[query])
        torch.testing.assert_close(inputs[1], aux[context])
        torch.testing.assert_close(inputs[2], aux[context])
    filterbank = model.frequency_branch.filterbank
    edges = filterbank.cutoff_edges().detach()
    require(bool((edges.diff() >= filterbank.min_bandwidth_hz - 1e-5).all()), "Sinc bandwidth constraint failed")
    require(filterbank.min_freq_hz - 1e-5 <= edges[0] and edges[-1] <= filterbank.max_freq_hz + 1e-5,
            "Sinc frequency bounds failed")
    with torch.no_grad():
        torch.testing.assert_close(model(spatial, frequency), logits, rtol=1e-4, atol=1e-5)
    result.update({"status": "PASS", "checks": "shapes, finite values, normalization axes, masks, branch alignment",
                   "sinc_edges_hz": edges.cpu().tolist()})


def check_backward(model, config, batch, device):
    """One disposable training update, with finite-gradient and update checks."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    sinc = model.frequency_branch.filterbank.raw_gap_logits
    before_sinc = sinc.detach().clone()
    before_classifier = model.classifier[-1].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["x_spatial"].to(device), batch["x_frequency"].to(device))
    loss = torch.nn.functional.cross_entropy(logits, batch["label"].to(device))
    require(bool(torch.isfinite(loss)), "Nonfinite cross-entropy loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None, f"Missing gradient for {name}")
            require(bool(torch.isfinite(parameter.grad).all()), f"Nonfinite gradient for {name}")
    gradient_norm = None
    if sinc.requires_grad:
        gradient_norm = float(sinc.grad.norm())
        require(gradient_norm > 0, "Learnable Sinc boundaries received zero gradient")
    optimizer.step()
    require(not torch.equal(before_classifier, model.classifier[-1].weight.detach()), "Classifier update failed")
    if sinc.requires_grad:
        require(not torch.equal(before_sinc, sinc.detach()), "Learnable Sinc update failed")
    model.eval()
    return {"status": "PASS", "source": "training subset only", "loss": float(loss.detach()),
            "sinc_gradient_norm": gradient_norm, "finite_gradients": True, "optimizer_step": True}


def session_description(epochs):
    return {"session": epochs.session, "spatial_shape": list(epochs.x_spatial.shape),
            "frequency_shape": list(epochs.x_frequency.shape), "labels_shape": list(epochs.labels.shape),
            "class_counts": np.bincount(epochs.labels, minlength=4).tolist(),
            "dropped_trials": epochs.dropped_trials}


def synthetic_session(config, subject, session, batch_size):
    """Generate explicitly synthetic aligned branch inputs for a standalone check."""
    rng = np.random.default_rng(42 if session == "T" else 2026)
    c, fs = config["data"]["n_channels"], config["data"]["sampling_rate"]
    t = round((config["data"]["epoch_end_sec"] - config["data"]["epoch_start_sec"]) * fs)
    context = config["model"]["sinc"]["context_samples"]
    count = ((max(40, batch_size * 2) + 3) // 4) * 4
    frequency = rng.normal(size=(count, c, t + 2 * context)).astype(np.float32)
    spatial = notch_trial(frequency[..., context:context + t], fs,
                          config["data"]["notch_frequency_hz"], config["data"]["notch_quality"])
    return SessionEpochs(spatial, frequency, np.tile(np.arange(4), count // 4),
                         subject, session, np.arange(count) * (t + 2 * context + 1))


def run_subject(config, subject, args, device, subject_report, show_table):
    model = DualBranchMINet(config).to(device)
    read = synthetic_session if args.synthetic else read_session
    training = read(config, subject, "T", args.batch_size) if args.synthetic else read(config, subject, "T")
    subject_report["sessions"] = [session_description(training)]
    train_indices, val_indices = stratified_split(training.labels, config["validation"]["val_ratio"], 42)
    normalizer = BranchNormalizer(**config["normalization"]).fit(training, train_indices)
    subject_report["normalization_fit"] = {"session": "T", "training_trials": len(train_indices),
                                             "validation_trials": len(val_indices)}
    subject_report["batches"] = []

    def verify(dataset, partition, backward=False):
        batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
        record = {"partition": partition}
        subject_report["batches"].append(record)
        check_batch(model, config, batch, device, record)
        if show_table and partition == "train":
            print(f"\n{'Stage':45} {'Observed output shapes':54} Result")
            for row in record["stages"]:
                print(f"{row['name']:45} {str(row['actual_outputs']):54} {row['status']}")
        print(f"PASS {subject}/{partition} batch={record['batch_size']} logits=[{record['batch_size']},4]", flush=True)
        tail_size = len(dataset) % args.batch_size
        if tail_size and len(dataset) > args.batch_size:
            tail_dataset = Subset(dataset, range(len(dataset) - tail_size, len(dataset)))
            tail_batch = next(iter(DataLoader(tail_dataset, batch_size=tail_size, num_workers=0)))
            tail_record = {"partition": partition + "_tail"}
            subject_report["batches"].append(tail_record)
            check_batch(model, config, tail_batch, device, tail_record)
            print(f"PASS {subject}/{partition}_tail batch={tail_size}", flush=True)
        if backward:
            if len(batch["label"]) > 1:
                single = {name: value[:1] for name, value in batch.items() if isinstance(value, torch.Tensor)}
                single_record = {"partition": "train_batch_one"}
                subject_report["batches"].append(single_record)
                check_batch(model, config, single, device, single_record)
            subject_report["backward"] = check_backward(model, config, batch, device)
            print(f"PASS {subject}/train diagnostic backward and optimizer step", flush=True)

    verify(BCI2aDataset(training, train_indices, normalizer), "train", backward=True)
    verify(BCI2aDataset(training, val_indices, normalizer), "validation")
    evaluation = read(config, subject, "E", args.batch_size) if args.synthetic else read(config, subject, "E")
    subject_report["sessions"].append(session_description(evaluation))
    verify(BCI2aDataset(evaluation, normalizer=normalizer), "test")
    subject_report["status"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "bci2a.yaml")
    parser.add_argument("--subjects", nargs="+", default=["1"], help="Subject IDs or all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu, auto, cuda, or cuda:0")
    parser.add_argument("--synthetic", action="store_true", help="Use explicitly synthetic inputs")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("dimension_check_report.json"))
    args = parser.parse_args()
    report = {"status": "RUNNING", "mode": "synthetic" if args.synthetic else "real_gdf",
              "started_utc": datetime.now(timezone.utc).isoformat(), "subjects": []}
    exit_code = 0
    try:
        require(args.batch_size > 0, "batch-size must be positive")
        config = load_config(args.config)
        if args.context_samples is not None:
            require(args.context_samples >= 0, "context-samples must be nonnegative")
            config["model"]["sinc"]["context_samples"] = args.context_samples
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        device = torch.device("cpu" if device_name == "auto" else device_name)
        require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
        report.update({"device": str(device), "torch_version": str(torch.__version__),
                       "config": str(args.config.resolve()), "data_paths": config["paths"],
                       "context_samples": config["model"]["sinc"]["context_samples"]})
        subjects = list(range(1, 10)) if args.subjects == ["all"] else args.subjects
        subjects = [normalize_subject(subject) for subject in subjects]
        require(len(set(subjects)) == len(subjects), "Subject IDs must be unique")
        for index, subject in enumerate(subjects):
            subject_report = {"subject": subject, "status": "RUNNING"}
            report["subjects"].append(subject_report)
            run_subject(config, subject, args, device, subject_report, show_table=index == 0)
        report["status"] = "PASS"
    except Exception as error:
        exit_code = 1
        report.update({"status": "FAIL", "error": f"{type(error).__name__}: {error}"})
        if report["subjects"]:
            report["subjects"][-1]["status"] = "FAIL"
            for batch in report["subjects"][-1].get("batches", []):
                if batch.get("status") == "RUNNING":
                    batch.update({"status": "FAIL", "error": str(error)})
        traceback.print_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n{report['status']} — dimension report saved to {args.report.resolve()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
