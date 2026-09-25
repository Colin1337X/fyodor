"""Exercise real trainer processes: stop -> exact checkpoint -> resume.

Python is a test driver only. The trainer and control protocol remain native C.
An optional --evidence JSON preserves commands, losses and output hashes.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

parser = argparse.ArgumentParser()
parser.add_argument("trainer", type=Path)
parser.add_argument("--evidence", type=Path)
args = parser.parse_args()
trainer = str(args.trainer.resolve())
evidence = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with tempfile.TemporaryDirectory(prefix="fyodor-stop-") as directory:
    root = Path(directory)
    corpus = root / "corpus.txt"
    corpus.write_text("abc" * 15, encoding="utf-8")
    sft = root / "sft.tsv"
    sft.write_text("a\tbc\nabc\tabcabc\nc\tacba\n", encoding="utf-8")
    dpo = root / "dpo.tsv"
    dpo.write_text("a\tbc\tc\nabc\tabcabc\tcabc\nc\tacba\tabc\n", encoding="utf-8")
    dimensions = ["--dimension", "32", "--ff", "64", "--layers", "1",
                  "--heads", "2", "--kv-heads", "1", "--context", "16"]

    def run(name, steps, settings=(), stop=None, resume=None, existing=False):
        model, checkpoint, metrics = (root / (name + ext) for ext in (".gguf", ".ckpt", ".csv"))
        if existing:
            model.write_bytes(b"preserve existing output")
        command = [trainer, "--data", str(corpus), "--steps", str(steps),
                   "--output", str(model), "--checkpoint", str(checkpoint), "--metrics", str(metrics),
                   "--accumulate", "3", *dimensions, *settings]
        if resume:
            command += ["--resume", str(resume)]
        if stop:
            command += ["--control-stdin", "1"]
            before_pipe = None
            if stop == "before":
                read_end, write_end = os.pipe()
                os.write(write_end,b"S")
                os.close(write_end)
                before_pipe = os.fdopen(read_end,"rb")
            child = subprocess.Popen(command, stdin=before_pipe or subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            if before_pipe:
                before_pipe.close()
            first = ""
            try:
                if stop != "before":
                    ready = queue.Queue()
                    threading.Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
                    first = ready.get(timeout=30)
                    assert first.startswith("step=1 "), first
                if stop not in ("eof", "disconnect", "before"):
                    child.stdin.write("S")
                    child.stdin.flush()
                if child.stdin is not None:
                    child.stdin.close()
                    child.stdin = None
                if stop == "disconnect":
                    child.stdout.close()
                    child.stdout = None
                output, error = child.communicate(timeout=30)
                output = first + (output or "")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            code = child.returncode
        else:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            output, error, code = result.stdout, result.stderr, result.returncode
        assert code == (1 if existing else 0), (command, output, error, code)
        with metrics.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if stop:
            step = len(rows) if stop == "disconnect" else int(output.split("stopping step=")[1].split()[0])
            assert step == len(rows) and step < steps, (step, len(rows), output)
            assert step == 0 if stop == "before" else step >= 1
        if existing:
            assert model.read_bytes() == b"preserve existing output"
            assert "GGUF output must be a new writable path" in error
        record = {"name": name, "command": command, "exit_code": code, "stdout": output,
                  "stderr": error, "steps": len(rows), "losses": [r["loss"] for r in rows],
                  "model_sha256": sha(model), "checkpoint_sha256": sha(checkpoint)}
        evidence.append(record)
        return record, checkpoint, model

    _, _, base = run("base", 2)
    for mode in ("pretrain", "cpt", "sft", "dpo"):
        settings = ["--mode", mode]
        if mode != "pretrain":
            settings += ["--base", str(base), "--rank", "4"]
        if mode in ("sft", "dpo"):
            settings += ["--data", str(sft if mode == "sft" else dpo)]
        for control in ("command", "eof", "disconnect"):
            name = mode + "-" + control
            stopped, checkpoint, _ = run(name, 5000, settings, stop=control)
            count = stopped["steps"]
            reference, _, _ = run(name + "-reference", count, settings)
            resumed, _, _ = run(name + "-resumed", 3, settings, resume=checkpoint)
            full, _, _ = run(name + "-full", count + 3, settings)
            for key in ("model_sha256", "checkpoint_sha256", "losses"):
                assert stopped[key] == reference[key], (name, key)
            for key in ("model_sha256", "checkpoint_sha256"):
                assert resumed[key] == full[key], (name, key)
            assert stopped["losses"] + resumed["losses"] == full["losses"]
    zero, checkpoint, _ = run("before-first-update", 5000, stop="before")
    resumed, _, _ = run("zero-resumed", 3, resume=checkpoint)
    full, _, _ = run("zero-reference", 3)
    for key in ("model_sha256", "checkpoint_sha256", "losses"):
        assert resumed[key] == full[key], key
    run("save-failure", 5000, stop="command", existing=True)
    # Never interpret ordinary file input as a control pipe or consume it.
    invalid_output = root / "invalid.gguf"
    invalid = [trainer, "--data", str(corpus), "--output", str(invalid_output),
               "--control-stdin", "1", *dimensions]
    with corpus.open("rb") as stream:
        result = subprocess.run(invalid, stdin=stream, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "requires a pipe" in result.stderr
    assert not invalid_output.exists()
    evidence.append({"name": "reject-regular-stdin", "command": invalid,
                     "exit_code": result.returncode, "stderr": result.stderr})

if args.evidence:
    args.evidence.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print(f"Safe stop: {len(evidence)} process runs passed; exact checkpoints, exports and losses.")
