"""Serial paired native training runs. Python is measurement tooling only.

Example: python backend/benchmarks/train_compare.py --before old/fyodor-train
  --after build/fyodor-train --output new-evidence-directory
Each process starts from the same seed; warmup updates are excluded from timing,
not training. Exact loss trajectories, checkpoint hashes and exported model
hashes must agree.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--base", type=Path, help="Optional real GGUF: benchmark rank-2 CPT at context 8")
    parser.add_argument("--workload", nargs="+", choices=("small","medium","long","tail"),
                        help="Limit synthetic runs when investigating a specific shape")
    args = parser.parse_args()
    if args.repetitions < 2 or args.warmup < 0 or args.steps <= args.warmup:
        parser.error("need at least two repetitions and steps > warmup >= 0")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    corpus = out / "corpus.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. Fyodor learns a repeated training corpus. " * 1000,
                      encoding="utf-8")
    env = dict(os.environ)
    env.pop("NYA_TRAIN_PROFILE", None)
    env["NYA_COMPUTE"] = "cpu"
    executables = {name: path.resolve() for name, path in (("before", args.before), ("after", args.after))}
    meta = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "corpus_sha256": sha(corpus),
            "executables": {name: {"path": str(path), "sha256": sha(path)} for name, path in executables.items()},
            "warmup_updates": args.warmup, "steps": args.steps,
            "environment": {k: v for k, v in env.items() if k.startswith("NYA_")},
            "limitations": "Shared desktop, CPU only; no thermal isolation or peak RAM sampling. Graph bytes exclude parameters, optimizer staging and process overhead.",
            "runs": [], "summary": []}
    reference = {}
    # Width, FF width, layers, heads, KV heads, context. Include an odd tail and
    # longer attention workload, not just one favorable matrix shape.
    workloads = {"small": (64, 176, 2, 4, 2, 32), "medium": (256, 704, 2, 8, 2, 128),
                 "long": (128, 352, 2, 4, 2, 512), "tail": (68, 183, 1, 2, 1, 31)}
    if args.workload:
        if args.base: parser.error("--workload selects synthetic shapes and cannot be used with --base")
        workloads = {name: workloads[name] for name in dict.fromkeys(args.workload)}
    if args.base:
        args.base = args.base.resolve()
        meta["base_model"] = {"path": str(args.base), "sha256": sha(args.base)}
        workloads = {"real-lora": ()}
    for workload, shape in workloads.items():
        for repetition in range(args.repetitions):
            for version in (["before", "after"] if repetition % 2 == 0 else ["after", "before"]):
                name = f"{workload}-{repetition}-{version}"
                model, metrics = out / (name + ".gguf"), out / (name + ".csv")
                checkpoint = out / (name + ".ckpt")
                command = [str(executables[version]), "--data", str(corpus), "--output", str(model),
                           "--checkpoint", str(checkpoint), "--metrics", str(metrics),
                           "--steps", str(args.steps), "--memory-mib", "512", "--seed", "42"]
                for key, value in zip(("dimension", "ff", "layers", "heads", "kv-heads", "context"), shape):
                    command.extend(["--" + key, str(value)])
                if args.base:
                    command.extend(["--mode", "cpt", "--base", str(args.base), "--rank", "2", "--context", "8"])
                start = time.perf_counter()
                run = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
                elapsed = time.perf_counter() - start
                (out / (name + ".log")).write_text(run.stdout + run.stderr, encoding="utf-8")
                if run.returncode:
                    raise RuntimeError(f"{name} failed; see preserved log")
                with metrics.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != args.steps:
                    raise RuntimeError("Missing per-update metrics")
                for row in rows:
                    for key in ("loss", "forward_ms", "backward_ms", "optimizer_ms", "step_ms", "tokens_per_second"):
                        if not math.isfinite(float(row[key])):
                            raise RuntimeError(f"Invalid {key} in {name}")
                    if float(row["step_ms"]) <= 0 or int(row["tokens"]) <= 0:
                        raise RuntimeError(f"Invalid elapsed time or token count in {name}")
                measured = rows[args.warmup:]
                model_hash, checkpoint_hash = sha(model), sha(checkpoint)
                contract = (model_hash, checkpoint_hash, [r["loss"] for r in rows])
                if workload in reference and reference[workload] != contract:
                    raise RuntimeError(f"Training trajectory or model differs in {name}")
                reference[workload] = contract
                entry = {"name": name, "workload": workload, "version": version, "command": command,
                         "wall_seconds_including_startup_checkpoint_export": elapsed,
                         "model_sha256": model_hash, "checkpoint_sha256": checkpoint_hash,
                         "graph_bytes": max(int(r["graph_bytes"]) for r in rows),
                         "first_loss": float(rows[0]["loss"]), "last_loss": float(rows[-1]["loss"])}
                for phase in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms"):
                    entry[phase] = statistics.mean(float(r[phase]) for r in measured)
                entry["tokens_per_second"] = sum(int(r["tokens"]) for r in measured) / sum(float(r["step_ms"]) / 1000 for r in measured)
                meta["runs"].append(entry)
                (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                # Delete only the artifacts just created here, after hashing.
                model.unlink()
                checkpoint.unlink()
                print(name, round(entry["tokens_per_second"], 2), flush=True)
        for version in executables:
            runs = [r for r in meta["runs"] if r["workload"] == workload and r["version"] == version]
            summary = {"workload": workload, "version": version, "repetitions": len(runs)}
            for key in ("tokens_per_second", "forward_ms", "backward_ms", "optimizer_ms", "step_ms"):
                values = [r[key] for r in runs]
                summary[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
            meta["summary"].append(summary)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
