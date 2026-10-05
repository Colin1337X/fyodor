"""Verify saved loss evidence against the current source and binaries."""
from pathlib import Path
import hashlib
import json
import re

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
skips = {"hardware-rocm-quant", "hardware-rocm-inference", "hardware-mlx-quant", "hardware-mlx-inference"}
for record in matrix:
    text = (out / "matrix" / record["log"]).read_text(errors="replace")
    if "100% tests passed" in text:
        assert set(re.findall(r"- ([\w-]+) \(Skipped\)", text)) == skips
    if record["name"] in ("build-cuda-tests", "build-cuda-c23-tests"):
        for name in ("training-loss-cuda", "training-loss-cuda-reference"):
            assert re.search(re.escape(name) + r"\s+\.+\s+Passed", text), name
sanitizers = read("sanitizers/sanitizers.json")
assert {r["tool"] for r in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
for record in sanitizers:
    assert record["exit_code"] == 0
    assert record["executable_sha256"] == snapshot["executables_sha256"]["nya-test-training-loss.exe"]
    assert "--launch-skip" not in record["command"] and "--launch-count" not in record["command"]
    text = (out / "sanitizers" / (record["tool"] + ".log")).read_text()
    assert "0 errors" in text
    if record["tool"] == "racecheck":
        assert "0 hazards displayed (0 errors, 0 warnings)" in text
probes = read("final-probes/probes.json")
assert len(probes) == 5 and all(r["exit_code"] == 0 for r in probes)
for record in probes:
    assert record["executable_sha256"] == snapshot["executables_sha256"]["nya-bench-training-device.exe"]
    result = read("final-probes/" + record["name"] + ".out")
    assert result["tolerance"] == 1e-5
    assert result["max_scaled_loss_error"] == 0
    assert result["max_scaled_logits_error"] <= 1e-5
    assert result["max_scaled_input_gradient_error"] <= 1e-5
    paired = record["name"].startswith("dpo")
    assert result["uploads"] == (6 if paired else 5)
    assert result["downloads"] == (42 if paired else 28)
    assert len(result["ms"]) == 5 and not result["timing_accepted"]
    assert result["loss_state_bytes_per_branch"] == 32 * result["tokens"] + 8
package = read("desktop-package.json")
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert sha(Path(package["installer"])) == package["sha256"]
assert not package["installed"] and not package["interactive_ui_checked"]
assert snapshot["generated_bundles"]["nya_cuda_source"]["sha256"] == "5d307a7bafd596eeb2f8fb886383db315edbc679562664e89bf627c56594ea4d"
print("Verified sources, binaries, 15 stages, complete sanitizers, real loss/DPO probes and package checks.")
