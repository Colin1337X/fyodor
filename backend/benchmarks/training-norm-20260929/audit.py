"""Verify resident RMSNorm source, correctness and packaged artifact evidence."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
snapshot = json.loads((out / "snapshot.json").read_text())
for name, digest in snapshot["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
matrix = json.loads((out / "matrix/validation.json").read_text())
assert len(matrix) == 15 and all(stage["exit_code"] == 0 for stage in matrix)
sanitizers = json.loads((out / "sanitizers.json").read_text())
assert {run["tool"] for run in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
for run in sanitizers:
    assert run["exit_code"] == 0
    log = (out / (run["tool"] + ".log")).read_text()
    expected = "0 hazards displayed (0 errors, 0 warnings)" if run["tool"] == "racecheck" else "ERROR SUMMARY: 0 errors"
    assert expected in log
    assert "resident RMSNorm:" in (out / (run["tool"] + ".out")).read_text()
for mode, tokens in (("rmsffn8", 8), ("rmsffn64", 64)):
    run = snapshot["results"][mode]
    assert run["exit_code"] == 0
    result = json.loads((out / (mode + ".out")).read_text())
    assert result == run["result"]
    assert result["tokens"] == tokens and result["microbatches"] == 2
    assert result["tolerance"] == 1e-5
    for field in ("output", "input_gradient", "norm_gradient"):
        assert 0 <= result[f"max_scaled_{field}_error"] <= 1e-5
    assert result["uploads"] == 6 and result["downloads"] == 4
    assert result["live_bytes"] < result["peak_bytes"] < 64 * 1024 * 1024
    assert not result["timing_accepted"] and len(result["ms"]) == 5
legacy = json.loads((out / "legacy-ffn8.out").read_text())
previous = json.loads((out.parent / "training-elementwise-20260928/ffn8.out").read_text())
for field in ("max_scaled_output_error", "max_scaled_input_gradient_error", "peak_bytes", "live_bytes", "uploads", "downloads"):
    assert legacy[field] == previous[field], field
assert not legacy["timing_accepted"]
package = json.loads((out / "desktop-package.json").read_text())
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert not package["installed"] and not package["interactive_ui_checked"]
assert snapshot["inference_bundle_identical_to_module_split"]
record = {
    "source_hashes_match": True, "matrix_stages_passed": 15,
    "complete_norm_sanitizers_passed": 4,
    "legacy_ffn8_correctness_and_memory_unchanged": True,
    "real_ffn_tokens": [8, 64], "real_ffn_scaled_tolerance": 1e-5,
    "inference_bundle_unchanged": True, "timings_accepted": False,
    "complete_gpu_training": False, "package_sha256": package["sha256"],
    "package_bytes": package["bytes"],
}
(out / "audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("RMSNorm evidence audit passed.")
