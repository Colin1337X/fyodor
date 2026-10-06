"""Audit optimizer, matrix precision, trajectory and package evidence."""
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
        for name in ("training-optimizer-cuda", "training-optimizer-cuda-reference"):
            assert re.search(re.escape(name) + r"\s+\.+\s+Passed", text), name
for directory, suite in (("optimizer-final-sanitizers", "optimizer"), ("matrix-sanitizers", "device")):
    sanitizers = read(directory + "/sanitizers.json")
    assert {r["tool"] for r in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
    for record in sanitizers:
        assert record["exit_code"] == 0
        assert record["executable_sha256"] == snapshot["executables_sha256"][f"nya-test-training-{suite}.exe"]
        assert "--launch-skip" not in record["command"] and "--launch-count" not in record["command"]
        text = (out / directory / (record["tool"] + ".log")).read_text()
        assert "0 errors" in text
        if record["tool"] == "racecheck":
            assert "0 hazards displayed (0 errors, 0 warnings)" in text
assert read("probes.json")[0]["exit_code"] == 1
assert "error=1.07053262e-05" in (out / "adamw8.log").read_text()
probes = read("ordered-forward-probes/probes.json")
assert len(probes) == 3 and all(r["exit_code"] == 0 for r in probes)
for record in probes:
    assert record["executable_sha256"] == snapshot["executables_sha256"]["nya-bench-training-device.exe"]
    result = read("ordered-forward-probes/" + record["name"] + ".out")
    assert result["tolerance"] == 1e-5 and result["steps"] == 32
    assert result["max_scaled_parameter_error"] == 0
    assert result["max_scaled_gradient_error"] <= 1e-5 and result["max_scaled_loss_error"] <= 1e-5
    assert result["uploads"] == 7 and result["downloads"] == 224
    assert len(result["losses"]) == len(result["cpu_losses"]) == len(result["step_ms"]) == 32
    assert not result["timing_accepted"] and result["losses"][-1] < result["losses"][0] * 0.9
    assert result["optimizer_plan_bytes"] == 1636040
regressions = read("regression-probes/probes.json")
assert len(regressions) == 3 and all(r["exit_code"] == 0 for r in regressions)
for record in regressions:
    assert record["executable_sha256"] == snapshot["executables_sha256"]["nya-bench-training-device.exe"]
    result = read("regression-probes/" + record["name"] + ".out")
    assert not result["timing_accepted"]
    for key, value in result.items():
        if key.startswith("max_scaled"):
            assert (max(value) if isinstance(value, list) else value) <= 1e-5
package = read("desktop-package.json")
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert sha(Path(package["installer"])) == package["sha256"]
assert not package["installed"] and not package["interactive_ui_checked"]
assert snapshot["generated_bundles"]["nya_cuda_source"]["sha256"] == "5d307a7bafd596eeb2f8fb886383db315edbc679562664e89bf627c56594ea4d"
print("Verified current source/binaries, 15 stages, eight sanitizer runs, rejected experiment, LoRA trajectories, regression probes and package checks.")
