"""Verify default-one compatibility against the preserved previous CLI."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parents[3]
out = root / "build-train-profile/default-accumulation-compatibility"
out.mkdir(exist_ok=False)
(out / "corpus.txt").write_text("abcabcabcabcabcabcabcabcabcabc")
(out / "sft.tsv").write_text("a\tbc\nabc\tabcabc\nc\tacba\n")
(out / "dpo.tsv").write_text("a\tbc\tc\nabc\tabcabc\tcabc\nc\tacba\tabc\n")
executables = {"before": root / ".tools/fyodor-train-before-accumulation.exe",
               "after": root / "build-cpu/fyodor-train.exe"}
dimensions = ["--dimension", "16", "--ff", "32", "--layers", "1", "--heads", "2", "--kv-heads", "1", "--context", "16"]
records = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for mode in ("pretrain", "cpt", "sft", "dpo"):
    for version, executable in executables.items():
        stem = out / f"{mode}-{version}"
        data = out / (f"{mode}.tsv" if mode in ("sft", "dpo") else "corpus.txt")
        command = [str(executable), "--mode", mode, "--data", str(data), "--output", str(stem.with_suffix(".gguf")),
                   "--checkpoint", str(stem.with_suffix(".ckpt")), "--steps", "5", *dimensions]
        if mode != "pretrain":
            command += ["--base", str(out / "pretrain-before.gguf"), "--rank", "4"]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        records.append({"mode": mode, "version": version, "command": command, "stdout": result.stdout,
                        "executable_sha256": digest(executable),
                        "gguf_sha256": digest(stem.with_suffix(".gguf")), "checkpoint_sha256": digest(stem.with_suffix(".ckpt"))})
    assert records[-1]["gguf_sha256"] == records[-2]["gguf_sha256"], mode
    assert records[-1]["checkpoint_sha256"] == records[-2]["checkpoint_sha256"], mode
    print(mode, "default model and checkpoint byte-identical", flush=True)

# An existing NYARUN v1 single-sequence checkpoint stays readable. Continue
# both binaries from that exact older file and compare the entire result.
for version, executable in executables.items():
    stem = out / f"resume-{version}"
    command = [str(executable), "--data", str(out / "corpus.txt"), "--resume", str(out / "pretrain-before.ckpt"),
               "--steps", "3", "--output", str(stem.with_suffix(".gguf")), "--checkpoint", str(stem.with_suffix(".ckpt")), *dimensions]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records.append({"mode": "resume-v1", "version": version, "command": command, "stdout": result.stdout,
                    "gguf_sha256": digest(stem.with_suffix(".gguf")), "checkpoint_sha256": digest(stem.with_suffix(".ckpt"))})
assert records[-1]["gguf_sha256"] == records[-2]["gguf_sha256"]
assert records[-1]["checkpoint_sha256"] == records[-2]["checkpoint_sha256"]
Path(__file__).with_name("default-compatibility.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print("older v1 checkpoint resumes byte-identically", flush=True)
