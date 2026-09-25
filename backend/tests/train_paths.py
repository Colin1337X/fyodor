"""Native CLI Unicode paths, exact resume, and exclusive-output protection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("trainer", type=Path)
parser.add_argument("--evidence", type=Path)
args = parser.parse_args()
trainer = str(args.trainer.resolve())
records = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


with tempfile.TemporaryDirectory(prefix="fyodor-unicode-") as directory:
    root = Path(directory)
    unicode = root / "\ud55c\uae00 \U0001f43e caf\u00e9"
    unicode.mkdir()
    ascii_data, unicode_data = root / "corpus.txt", unicode / "\uc790\ub8cc \U0001f43e.txt"
    ascii_data.write_text("a\ud55c\U0001f642bc" * 3, encoding="utf-8")
    shutil.copyfile(ascii_data, unicode_data)
    sft = unicode / "\uc9c0\ub3c4.tsv"
    sft.write_text("a\tbc\nabc\tabcabc\nc\tacba\n", encoding="utf-8")
    dpo = unicode / "\uc120\ud638.tsv"
    dpo.write_text("a\tbc\tc\nabc\tabcabc\tcabc\nc\tacba\tabc\n", encoding="utf-8")
    dimensions = ["--dimension", "16", "--ff", "32", "--layers", "1",
                  "--heads", "2", "--kv-heads", "1", "--context", "32"]

    def run(name, steps, settings=(), target=unicode, resume=None, stop=False, expected=0):
        model, checkpoint, metrics = (target / (name + ext) for ext in (".gguf", ".ckpt", ".csv"))
        command = [trainer, "--data", str(unicode_data), "--output", str(model),
                   "--checkpoint", str(checkpoint), "--metrics", str(metrics), "--steps", str(steps),
                   "--accumulate", "3", *dimensions, *settings]
        if resume:
            command += ["--resume", str(resume)]
        if stop:
            command += ["--control-stdin", "1"]
        if stop:
            # Queue before launch: step-zero coverage must not depend on the
            # parent winning a scheduling race against the trainer's startup.
            read_end, write_end = os.pipe()
            os.write(write_end,b"S")
            os.close(write_end)
            with os.fdopen(read_end,"rb") as control:
                result = subprocess.run(command, stdin=control, capture_output=True,
                                        text=True, encoding="utf-8", timeout=30)
        else:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == expected, (command, result.stdout, result.stderr)
        record = {"name": name, "command": command, "exit_code": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}
        if expected == 0:
            assert "exported=" + str(model) in result.stdout
            record.update(model_sha256=sha(model), checkpoint_sha256=sha(checkpoint))
            with metrics.open(newline="", encoding="utf-8") as stream:
                record["losses"] = [r["loss"] for r in csv.DictReader(stream)]
        records.append(record)
        return record, checkpoint, model

    base_record, _, base_ascii = run("base", 2, ["--data", str(ascii_data)], target=root)
    base = unicode / "\ubaa8\ub378 \U0001f43e.gguf"
    shutil.copyfile(base_ascii, base)
    for mode in ("pretrain", "cpt", "sft", "dpo"):
        settings = ["--mode", mode]
        if mode != "pretrain":
            settings += ["--base", str(base), "--rank", "4"]
        if mode in ("sft", "dpo"):
            settings += ["--data", str(sft if mode == "sft" else dpo)]
        first, checkpoint, _ = run(mode + "-\uccab", 2, settings)
        resumed, _, _ = run(mode + "-\uc7ac\uac1c", 3, settings, resume=checkpoint)
        full, _, _ = run(mode + "-\uc804\uccb4", 5, settings)
        for key in ("model_sha256", "checkpoint_sha256"):
            assert resumed[key] == full[key], (mode, key)
        assert first["losses"] + resumed["losses"] == full["losses"]
        # Path spellings are not checkpoint identity: move inputs/base/resume
        # between ASCII and Unicode locations without changing their bytes.
        ascii_settings = settings.copy()
        if mode != "pretrain":
            ascii_settings[ascii_settings.index(str(base))] = str(base_ascii)
        if mode not in ("sft", "dpo"):
            ascii_settings += ["--data", str(ascii_data)]
        ascii_checkpoint = root / (mode + ".ckpt")
        shutil.copyfile(checkpoint, ascii_checkpoint)
        moved, _, _ = run(mode + "-moved", 3, ascii_settings, target=root, resume=ascii_checkpoint)
        for key in ("model_sha256", "checkpoint_sha256", "losses"):
            assert moved[key] == resumed[key], (mode, key)
    _, checkpoint, _ = run("\uc911\uc9c0", 5000, stop=True)
    resumed, _, _ = run("\uc911\uc9c0-\uc7ac\uac1c", 3, resume=checkpoint)
    full, _, _ = run("stop-reference", 3, target=root)
    for key in ("model_sha256", "checkpoint_sha256", "losses"):
        assert resumed[key] == full[key], key
    for flag in ("--output", "--checkpoint", "--metrics"):
        protected = unicode / (flag[2:] + "-\ubcf4\ud638.bin")
        protected.write_bytes(b"keep existing bytes\x00\xff")
        before = sha(protected)
        run("protect-" + flag[2:], 1, [flag, str(protected)], expected=1)
        assert sha(protected) == before
    # Reject corrupt Unicode-named checkpoints before producing a model.
    corrupt = unicode / "\uc190\uc0c1.ckpt"
    corrupt.write_bytes(b"NYARUN\x01\x00")
    _, _, rejected_model = run("corrupt", 1, resume=corrupt, expected=1)
    assert not rejected_model.exists()

if args.evidence:
    args.evidence.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(f"Unicode training paths: {len(records)} native process runs passed.")
