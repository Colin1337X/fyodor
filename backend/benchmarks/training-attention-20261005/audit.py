"""Verify saved attention evidence against the current source and binaries."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent


def read(name):
    return json.loads((out / name).read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


snapshot = read("snapshot.json")
for name, digest in snapshot["source_sha256"].items():
    assert sha(root / name) == digest, name
for name, digest in snapshot["executables_sha256"].items():
    assert sha(root / "build-cuda" / name) == digest, name
matrix = read("matrix/validation.json")
assert len(matrix) == 15 and all(r["exit_code"] == 0 for r in matrix)
for record in matrix:
    assert (out / "matrix" / record["log"]).is_file()
sanitizers = read("final-sanitizers/sanitizers.json")
assert {r["tool"] for r in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
for record in sanitizers:
    assert record["exit_code"] == 0
    assert record["executable_sha256"] == snapshot["executables_sha256"]["nya-test-training-attention.exe"]
    assert "--launch-skip" not in record["command"] and "--launch-count" not in record["command"]
    text = (out / "final-sanitizers" / (record["tool"] + ".log")).read_text()
    assert "0 errors" in text
    if record["tool"] == "racecheck":
        assert "0 hazards displayed (0 errors, 0 warnings)" in text
        assert "--force-synchronization-limit" in record["command"]
assert read("sanitizers.json")[-1]["exit_code"] == 3221226505
assert read("racecheck-trace.json")["exit_code"] == 3221226505
assert read("racecheck-bounded.json")["exit_code"] == 0
assert len(read("windows-faults.json")) == 2
probes = read("probes.json")
assert len(probes) == 4 and all(r["exit_code"] == 0 for r in probes)
for name in ("attention8", "attention64", "attention8-reference"):
    result = read(name + ".out")
    assert result["tolerance"] == 1e-5
    assert result["max_scaled_output_error"] <= 1e-5
    assert result["max_scaled_input_gradient_error"] <= 1e-5
    assert result["uploads"] == 7 and result["downloads"] == 21
    assert len(result["ms"]) == 5 and not result["timing_accepted"]
    n, h = result["tokens"], result["heads"][0]
    assert result["attention_state_bytes"] == 16 * n * h
    assert result["attention_workspace_bytes"] == 16 * n * h * min(n, 16)
assert max(read("qkrope8.out")["max_scaled_errors"]) <= 1e-5
package = read("desktop-package.json")
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert sha(Path(package["installer"])) == package["sha256"]
assert not package["installed"] and not package["interactive_ui_checked"]
assert snapshot["generated_bundles"]["nya_cuda_source"]["lf_normalized_sha256"] == "76313ca8c3b0ee61f4b7d6e805a337ca773617cc7021651d0b5d4d140942a680"
print("Verified sources, binaries, 15 stages, complete sanitizers, real attention probes, preserved failures and package checks.")
