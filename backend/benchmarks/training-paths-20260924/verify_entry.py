"""Recheck all CLI configurations after separating main/wmain entry points.

The full matrix covers the shared file-layer changes. This supplemental pass
rebuilds the final CLI and its private test in every configuration, including
the retained source copies with optional components physically removed.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
changed = ["backend/core/train_main.c", "backend/tests/test_train_runner.c"]
builds = [(name, root / name) for name in ("build-cpu", "build-cuda", "build-cuda-c23", "build-ort", "build-sanitize")]
for kind in ("optcpp", "cuda", "cutlass"):
    name = "verify-without-" + kind
    log = (out / (name + ".log")).read_text(encoding="utf-8")
    matches = re.findall(r"Build files have been written to: ([^\r\n]+)", log)
    assert len(matches) == 1
    build = Path(matches[0]).resolve()
    expected = root / ("build-cuda" if kind == "cutlass" else "build-cpu")
    assert build.is_relative_to(expected.resolve()) and build.parent.name.startswith("without-" + kind + "-")
    for name_in_source in changed:
        destination = build.parent / "source" / name_in_source
        assert destination.is_file()
        shutil.copyfile(root / name_in_source, destination)
    builds.append((name, build))

record = {"source_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in changed}, "runs": []}
for name, build in builds:
    env = dict(os.environ)
    compiler = ".tools/llvm-mingw/llvm-mingw-20260826-ucrt-x86_64/bin" if name == "build-sanitize" else ".tools/w64devkit/bin"
    env["PATH"] = str(root / compiler) + os.pathsep + env["PATH"]
    commands = [["cmake", "--build", str(build), "--target", "fyodor-train", "nya-test-train-runner", "--parallel", "4"],
                ["ctest", "--test-dir", str(build), "-R", "train-accumulation|train-cli-workflows|train-stop-workflows|train-unicode-paths", "--output-on-failure"]]
    with (out / (name + "-entry.log")).open("wb") as log:
        for command in commands:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["runs"].append({"name": name, "command": command, "exit_code": result.returncode})
            (out / "entry-validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            assert result.returncode == 0, name
    print(name, "final entry tests passed", flush=True)
hashes = json.loads((out / "source-hashes.json").read_text(encoding="utf-8"))
hashes.update(record["source_sha256"])
assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == s for n, s in hashes.items())
(out / "source-hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
