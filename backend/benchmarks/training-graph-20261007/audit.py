"""Audit current resident-graph sources against completed validation evidence."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent


def read(name):
    return json.loads((out / name).read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


snapshot = read("snapshot.json")
for name, digest in snapshot["source_sha256"].items():
    assert sha(root / name) == digest, name
for name, digest in snapshot["executables_sha256"].items():
    assert sha(root / "build-cuda" / name) == digest, name
assert sha(root / "build-cpu/fyodor-train.exe") == snapshot["packaged_trainer_sha256"]
matrix = read("matrix/validation.json")
assert len(matrix) == 15 and all(item["exit_code"] == 0 for item in matrix)
skips = {"hardware-rocm-quant", "hardware-rocm-inference", "hardware-mlx-quant", "hardware-mlx-inference"}
for item in matrix:
    text = (out / "matrix" / item["log"]).read_text(errors="replace")
    if "100% tests passed" in text:
        assert set(re.findall(r"- ([\w-]+) \(Skipped\)", text)) == skips
    if item["name"] in ("build-cuda-tests", "build-cuda-c23-tests"):
        for name in ("training-graph-cuda", "training-graph-cuda-reference"):
            assert re.search(re.escape(name) + r"\s+\.+\s+Passed", text)
sanitizers = read("sanitizers/sanitizers.json")
assert len(sanitizers)==8
assert {item["tool"] for item in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
for item in sanitizers:
    assert item["exit_code"] == 0
    assert item["executable_sha256"] == snapshot["executables_sha256"]["nya-test-training-graph.exe"]
    text = (out / "sanitizers" / item["log"]).read_text()
    assert "0 errors" in text
    if item["tool"] == "racecheck":
        assert "0 hazards displayed (0 errors, 0 warnings)" in text
    output = (out / "sanitizers" / item["output"]).read_text()
    if item["suite"]=="lifecycle":
        assert "exact resume/evaluation; exported generation bcabcabc" in output
    else:
        assert output.count("same-representation exported inference maximum_scaled_error=")==2
aligned=read("aligned.json")
assert len(aligned)==4 and all(item["exit_code"]==0 for item in aligned)
real=(out/"tinyllama-aligned.log").read_text()
assert "same-representation exported inference maximum_scaled_error=5.00843671e-06" in real
assert "maximum_scaled_error=4.63128629e-10" in real
assert snapshot["generated_bundles"]["nya_cuda_source"]==read("../training-session-20261006/snapshot.json")["generated_bundles"]["nya_cuda_source"]
package = read("desktop-package.json")
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert sha(Path(package["installer"])) == package["sha256"]
contents = package["verified_contents_sha256"]
assert contents["backend/fyodor-backend.exe"] == snapshot["executables_sha256"]["fyodor-backend.exe"]
assert contents["backend/fyodor-train.exe"] == snapshot["packaged_trainer_sha256"]
assert not package["installed"] and not package["interactive_ui_checked"]
print("Verified current sources/binaries, 15 validation stages, eight GPU sanitizer runs, exact checkpoint continuation and package checks.")
