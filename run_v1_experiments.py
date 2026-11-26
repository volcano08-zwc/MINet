"""Run the three v1 frequency-shift experiments sequentially on one GPU."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ORDER = ("regularized", "lightweight", "groupnorm")


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "bci2a.yaml"))
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU index; otherwise inherit CUDA_VISIBLE_DEVICES")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_ORDER, default=list(EXPERIMENT_ORDER))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training or writing output")
    args = parser.parse_args(argv)
    if args.epochs < 1 or (args.gpu is not None and args.gpu < 0):
        parser.error("Epochs must be positive and GPU index nonnegative")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("Experiment names must be unique")
    if "all" in args.subjects and args.subjects != ["all"]:
        parser.error("Use --subjects all or a list of subjects")
    if args.gpu is not None and args.device not in {"cuda", "cuda:0", "cpu"}:
        parser.error("--gpu exposes one physical GPU; use --device cuda:0")
    return args


def build_jobs(args):
    root = project_path(args.output_dir or f"output/v1_improvements_seed{args.seed}")
    jobs = []
    for name in args.experiments:
        directory = root / f"v1_{name}"
        command = [sys.executable, "-u", str(PROJECT_ROOT / "train_2a.py"),
                   "--config", str(project_path(args.config)), "--subjects", *args.subjects,
                   "--seeds", str(args.seed), "--epochs", str(args.epochs),
                   "--device", args.device, "--protocol", "full-session1", "--run-version", "v1",
                   "--experiment", name, "--augmentation", "frequency_shift",
                   "--frequency-probability", "0.5", "--frequency-max-hz", "0.5",
                   "--output-dir", str(directory)]
        for option, value in (("--data-dir", args.data_dir), ("--labels-dir", args.labels_dir)):
            if value is not None:
                command.extend((option, value))
        jobs.append({"experiment_variant": name, "output_dir": str(directory), "command": command,
                     "log_file": str(root / "logs" / f"v1_{name}.log"), "status": "pending"})
    return root, jobs


def save_manifest(root, manifest):
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_training(command, environment, log_file):
    """Wait for one child to exit, streaming its output to Screen and its log."""
    with Path(log_file).open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def collect_completed_job(job, args):
    directory = Path(job["output_dir"])
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected = 9 if args.subjects == ["all"] else len(args.subjects)
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(directory.glob("A*/seed_*/test_metrics.json"))]
    if not summary.get("complete") or len(records) != expected:
        raise RuntimeError(f"Incomplete result set: {directory}")
    for record in records:
        if (record.get("version") != "v1" or record.get("experiment_variant") != job["experiment_variant"]
                or record["seed"] != args.seed or record["epochs_run"] != args.epochs
                or record.get("augmentation", {}).get("method") != "frequency_shift"):
            raise RuntimeError(f"Result settings do not match the queued experiment: {directory}")
    accuracy = summary["overall_across_subject_means"]["accuracy"]
    return {"version": "v1", "experiment_variant": job["experiment_variant"],
            "augmentation": "frequency_shift", "seed": args.seed, "epochs": args.epochs,
            "subjects": len(records), "mean_best_test_accuracy": accuracy["mean"],
            "std_across_subjects": accuracy["std"], "trainable_parameters": records[0]["trainable_parameters"],
            "output_dir": str(directory)}


def save_comparison(root, rows):
    (root / "experiment_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (root / "experiment_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    root, jobs = build_jobs(args)
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"v1 queue: seed={args.seed}, epochs={args.epochs}, device={args.device}, "
          f"CUDA_VISIBLE_DEVICES={environment.get('CUDA_VISIBLE_DEVICES', 'inherited/all')}", flush=True)
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
    if args.dry_run:
        return 0
    # Check the whole queue before starting, so existing results are never overwritten.
    if (root / "run_manifest.json").exists():
        raise FileExistsError(f"A queue manifest already exists in {root}. Choose a new --output-dir.")
    for job in jobs:
        if Path(job["output_dir"]).exists() or Path(job["log_file"]).exists():
            raise FileExistsError(f"Experiment output already exists: {job['output_dir']}. "
                                     "Choose a new --output-dir.")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    manifest = {"version": "v1", "protocol": "full-session1", "seed": args.seed, "epochs": args.epochs,
                "augmentation": {"method": "frequency_shift", "probability": 0.5, "max_hz": 0.5},
                "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"), "jobs": jobs}
    save_manifest(root, manifest)
    comparison = []
    for index, job in enumerate(jobs, 1):
        job["status"] = "running"
        save_manifest(root, manifest)
        print(f"\n[{index}/{len(jobs)}] v1/{job['experiment_variant']} started", flush=True)
        try:
            code = run_training(job["command"], environment, job["log_file"])
            job["return_code"] = code
            if code:
                job["status"] = "failed"
                save_manifest(root, manifest)
                print(f"Training failed (exit {code}); queue stopped. Log: {job['log_file']}", flush=True)
                return code
            comparison.append(collect_completed_job(job, args))
            save_comparison(root, comparison)
        except (Exception, KeyboardInterrupt) as error:
            job["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            job["error"] = str(error)
            save_manifest(root, manifest)
            raise
        job["status"] = "completed"
        save_manifest(root, manifest)
        print(f"v1/{job['experiment_variant']} completed: "
              f"mean highest test Acc={comparison[-1]['mean_best_test_accuracy']:.2%}", flush=True)
    print(f"All requested v1 experiments completed. Comparison: {root / 'experiment_comparison.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
