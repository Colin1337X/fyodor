"""Audit current public CUDA training sources against completed validation evidence."""
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
assert sha(root / "build-cuda/fyodor-train.exe") == snapshot["packaged_trainer_sha256"]
matrix = read("matrix/validation.json")
assert len(matrix) == 15 and all(item["exit_code"] == 0 for item in matrix)
skips = {"hardware-rocm-quant", "hardware-rocm-inference", "hardware-mlx-quant", "hardware-mlx-inference"}
for item in matrix:
    text = (out / "matrix" / item["log"]).read_text(errors="replace")
    if "100% tests passed" in text:
        assert set(re.findall(r"- ([\w-]+) \(Skipped\)", text)) == skips
    if item["name"] in ("build-cuda-tests", "build-cuda-c23-tests"):
        for name in ("training-graph-cuda", "training-graph-cuda-reference", "train-cuda-eval", "train-cuda-stop"):
            assert re.search(re.escape(name) + r"\s+\.+\s+Passed", text)
sanitizers = read("sanitizers-corrected/sanitizers.json")
assert len(sanitizers)==8
assert {item["tool"] for item in sanitizers} == {"memcheck", "initcheck", "synccheck", "racecheck"}
for item in sanitizers:
    assert item["exit_code"] == 0
    assert item["executable_sha256"] == snapshot["executables_sha256"]["nya-test-training-device.exe" if item["suite"]=="device" else "nya-test-training-graph.exe"]
    text = (out / "sanitizers-corrected" / item["log"]).read_text()
    assert "0 errors" in text
    if item["tool"] == "racecheck":
        assert "0 hazards displayed (0 errors, 0 warnings)" in text
    output = (out / "sanitizers-corrected" / item["output"]).read_text()
    if item["suite"]=="lifecycle":
        assert "exact resume/evaluation; exported generation bcabcabc" in output
    else:
        assert "resident matrix suite:" in output
real=read("real-model.json")
assert real['passed'] and len(real['runs'])==3 and all(item['exit_code']==0 for item in real['runs'])
assert real['maximum_cpt_state_scaled_error']<=1e-5
assert real['trainer_sha256']==snapshot['executables_sha256']['fyodor-train.exe']
probe=read('public-probe.json')
assert probe['exact_gpu_resume'] and probe['maximum_cpu_state_scaled_error']<=1e-5
assert snapshot['generated_bundles']==read('../training-graph-20261007/snapshot.json')['generated_bundles']
visual=read('visual-check.json')
desktop=(root/'frontend/src-tauri/target/release/fyodor-desktop.exe').read_bytes()
assert desktop.count(b'__TAURI_BUNDLE_TYPE_VAR_UNK')==1
assert hashlib.sha256(desktop.replace(b'__TAURI_BUNDLE_TYPE_VAR_UNK',b'__TAURI_BUNDLE_TYPE_VAR_NSS',1)).hexdigest()==visual['final_recheck']['executable_sha256']
frontend=(out/'final-frontend-tests.log').read_text(encoding='utf-8-sig')
assert 'tests 12' in frontend and 'pass 12' in frontend and 'fail 0' in frontend
rust=(out/'rust-tests-4.log').read_text(encoding='utf-8-sig')
assert '3 passed; 0 failed' in rust
package = read("final-package/desktop-package.json")
assert sha(Path(package["reused_execution_evidence"]["path"])) == package["reused_execution_evidence"]["sha256"]
assert package["cli_exit_code"] == 0 and len(package["http_calls"]) == 8
assert sha(Path(package["installer"])) == package["sha256"]
contents = package["verified_contents_sha256"]
assert contents["backend/fyodor-backend.exe"] == snapshot["executables_sha256"]["fyodor-backend.exe"]
assert contents["backend/fyodor-train.exe"] == snapshot["packaged_trainer_sha256"]
assert package['training_capabilities']=={'cpu':True,'cuda':True}
assert len(package['gpu_training_checks'])==2 and all(item['exit_code']==0 for item in package['gpu_training_checks'])
assert not package["installed"] and not package["interactive_ui_checked"]
print("Verified current sources/binaries, 15 validation stages, eight GPU sanitizer runs, exact checkpoint continuation and package checks.")
