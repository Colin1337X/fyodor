"""Check completion evidence without treating sanitizer summaries as exit codes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
snapshot = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))
for name, expected in snapshot["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
matrix = json.loads((out / "matrix/validation.json").read_text(encoding="utf-8"))
assert len(matrix) == 15 and all(item["exit_code"] == 0 for item in matrix)
sanitizers = json.loads((out / "sanitizers.json").read_text())
assert len(sanitizers["runs"]) == 3
for run in sanitizers["runs"]:
    tool = run["tool"]
    assert run["exit_code"] == 0
    assert "ERROR SUMMARY: 0 errors" in (out / f"{tool}.log").read_text()
    assert "gradient=60000" in (out / f"{tool}.out").read_text()
race = json.loads((out / "racecheck-bounded.json").read_text())
assert race["exit_code"] == 0
assert "0 hazards displayed (0 errors, 0 warnings)" in (out / "racecheck-bounded.log").read_text()
assert "gradient=60000" in (out / "racecheck-bounded.out").read_text()
interrupted = json.loads((out / "racecheck-interrupted.json").read_text())
assert not interrupted["accepted"] and interrupted["exit_code"] != 0
package = json.loads((out / "desktop-package.json").read_text())
record = {"source_hashes_match": True, "matrix_stages_passed": len(matrix),
          "full_stress_sanitizers": ["memcheck", "initcheck", "synccheck"],
          "racecheck": race, "unrestricted_racecheck": "interrupted and excluded",
          "package_verification_record": package,
          "timings_accepted": False, "gpu_training_integration": "unfinished"}
(out / "audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Evidence audit passed.")
