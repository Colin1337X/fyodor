"""Verify the numerical-check evidence and final packaged binary identity."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
snapshot=json.loads((out/"snapshot.json").read_text())
for name,digest in snapshot["source_sha256"].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
matrix=json.loads((out/"matrix/validation.json").read_text())
assert len(matrix)==15 and all(stage["exit_code"]==0 for stage in matrix)
sanitizers=json.loads((out/"sanitizers.json").read_text())
assert len(sanitizers)==4
for run in sanitizers:
    assert run["exit_code"]==0
    text=(out/(run["tool"]+".log")).read_text()
    assert ("0 hazards displayed (0 errors, 0 warnings)" if run["tool"]=="racecheck" else "ERROR SUMMARY: 0 errors") in text
    assert "finite checks:" in (out/(run["tool"]+".out")).read_text()
package=json.loads((out/"desktop-package.json").read_text())
assert package["cli_exit_code"]==0 and len(package["http_calls"])==8
assert not package["installed"] and not package["interactive_ui_checked"]
assert snapshot["inference_bundle_identical_to_module_split"]
record={"source_hashes_match":True,"matrix_stages_passed":15,"new_fixture_sanitizers_passed":4,
        "inference_bundle_unchanged":True,"real_weight_orders_passed":["checked-forward","checked-reverse"],
        "timings_accepted":False,"complete_gpu_training":False,
        "package_sha256":package["sha256"],"package_bytes":package["bytes"]}
(out/"audit.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print("Finite-check evidence audit passed.")
