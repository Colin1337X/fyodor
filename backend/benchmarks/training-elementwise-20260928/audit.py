"""Check elementwise correctness, residency, source and package evidence."""
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
    stdout = (out / (run["tool"] + ".out")).read_text()
    assert "resident elementwise:" in stdout and "gated branch: two microbatches, 68 launches" in stdout
for mode, tokens in (("ffn8", 8), ("ffn64", 64)):
    run = snapshot["results"][mode]
    assert run["exit_code"] == 0
    result = json.loads((out / (mode + ".out")).read_text())
    assert result == run["result"]
    assert result["tokens"] == tokens and result["microbatches"] == 2
    assert result["tolerance"] == 1e-5
    assert 0 <= result["max_scaled_output_error"] <= 1e-5
    assert 0 <= result["max_scaled_input_gradient_error"] <= 1e-5
    assert result["uploads"] == 5 and result["downloads"] == 3
    assert result["live_bytes"] < result["peak_bytes"] < 64 * 1024 * 1024
    assert not result["timing_accepted"] and len(result["ms"]) == 5
package = json.loads((out / "desktop-package.json").read_text())
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert not package["installed"] and not package["interactive_ui_checked"]
assert snapshot["inference_bundle_identical_to_module_split"]
record = {
    "source_hashes_match": True, "matrix_stages_passed": 15,
    "complete_elementwise_sanitizers_passed": 4,
    "real_ffn_tokens": [8, 64], "real_ffn_scaled_tolerance": 1e-5,
    "inference_bundle_unchanged": True, "timings_accepted": False,
    "complete_gpu_training": False, "package_sha256": package["sha256"],
    "package_bytes": package["bytes"],
}
(out / "audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Elementwise evidence audit passed.")
