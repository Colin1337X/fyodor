"""Audit the saved RoPE milestone evidence and current production/test sources."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
def read(name):return json.loads((out/name).read_text())
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
snapshot=read("snapshot.json")
for name,digest in snapshot["source_sha256"].items():assert sha(root/name)==digest,name
matrix=read("matrix/validation.json")
assert len(matrix)==15 and all(r["exit_code"]==0 for r in matrix)
for record in matrix:assert (out/"matrix"/record["log"]).is_file()
records=read("sanitizers.json")
assert len(records)==4 and all(r["exit_code"]==0 for r in records)
for record in records:
 text=(out/(record["tool"]+".log")).read_text()
 assert "0 errors" in text
 if record["tool"]=="racecheck":assert "0 hazards displayed (0 errors, 0 warnings)" in text
records=read("probes.json")
assert len(records)==4 and all(r["exit_code"]==0 for r in records)
for name in ("qkrope8","qkrope64","qkrope8-reference"):
 r=read(name+".out")
 assert r["tolerance"]==1e-5 and max(r["max_scaled_errors"])<=1e-5
 assert len(r["ms"])==5 and r["uploads"]==6 and r["downloads"]==28 and not r["timing_accepted"]
r=read("rmsffn8.out")
assert r["max_scaled_output_error"]<=1e-5 and r["max_scaled_input_gradient_error"]<=1e-5 and r["max_scaled_norm_gradient_error"]<=1e-5
package=read("desktop-package.json")
assert package["cli_exit_code"]==0 and len(package["http_calls"])==8
assert sha(Path(package["installer"]))==package["sha256"]
assert not package["installed"] and not package["interactive_ui_checked"]
for name in ("focused-tests.log","diagnostic-test.log","trig-gcc.out","trig-gcc-long.out","trig-gcc-import.out","oracle.json","build-cpu-configure.log","installer-build.log"):
 assert (out/name).stat().st_size>0,name
assert snapshot["generated_bundles"]["nya_cuda_source"]["lf_normalized_sha256"]=="76313ca8c3b0ee61f4b7d6e805a337ca773617cc7021651d0b5d4d140942a680"
print("Verified source identities, 15 stages, four CUDA sanitizers, real-model probes, preserved failures and package checks.")
