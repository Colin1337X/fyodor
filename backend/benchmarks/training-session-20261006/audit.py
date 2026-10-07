"""Audit current resident-session sources against completed validation evidence."""
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
        for name in ("training-session-cuda", "training-session-cuda-reference"):
            assert re.search(re.escape(name) + r"\s+\.+\s+Passed", text)
sanitizers = read("sanitizers/sanitizers.json")
assert {item["tool"] for item in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
for item in sanitizers:
    assert item["exit_code"] == 0
    assert item["executable_sha256"] == snapshot["executables_sha256"]["nya-test-training-session.exe"]
    text = (out / "sanitizers" / (item["tool"] + ".log")).read_text()
    assert "0 errors" in text
    if item["tool"] == "racecheck":
        assert "0 hazards displayed (0 errors, 0 warnings)" in text
    output = (out / "sanitizers" / (item["tool"] + ".out")).read_text()
    assert "exact checkpoint continuation passed" in output and "ownership" in output
package = read("desktop-package.json")
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert sha(Path(package["installer"])) == package["sha256"]
contents = package["verified_contents_sha256"]
assert contents["backend/fyodor-backend.exe"] == snapshot["executables_sha256"]["fyodor-backend.exe"]
assert contents["backend/fyodor-train.exe"] == snapshot["packaged_trainer_sha256"]
assert not package["installed"] and not package["interactive_ui_checked"]
print("Verified current sources/binaries, 15 validation stages, four GPU sanitizers, exact checkpoint continuation and package checks.")
