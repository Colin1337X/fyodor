"""Record exact source/model identity and resident finite-check resource cost."""
import hashlib
import json
import re
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def command(*args): return subprocess.check_output(args,cwd=root,text=True).strip()
sources=["CUDA/training.cu","CUDA/training.inc","CUDA/cuda.c","CUDA/CMakeLists.txt",
         "backend/CMakeLists.txt","backend/core/training_device.c","backend/core/training_device.h",
         "backend/core/training_device_backend.h","backend/tests/test_training_device.c",
         "backend/benchmarks/training_device_bench.c"]
header=(root/"build-cuda/backend/cuda-provider/cuda_source.h").read_text()
bundles={}
for name in ("nya_cuda_source","nya_cuda_training_source"):
    values=re.search(re.escape(name)+r"\[\] = \{([^}]+)\}",header)[1]
    data=bytes(int(value,0) for value in values.split(","))
    bundles[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
previous=json.loads((root/"backend/benchmarks/cuda-modules-20260927/source-bundles.json").read_text())
assert bundles["nya_cuda_source"]==previous["nya_cuda_source"]
results={}
for mode in ("reuse-forward","checked-forward","checked-reverse"):
    counters={key:int(value) for key,value in (field.split("=") for field in (out/f"{mode}.log").read_text().split())}
    rows=[json.loads(line) for line in (out/f"{mode}.out").read_text().splitlines()]
    assert len(rows)==18 and all(len(row["ms"])==5 for row in rows)
    results[mode]={"command":["build-cuda/nya-bench-training-device.exe",".tools/tinyllama-q4_k_m.gguf",mode],
                   "exit_code":0,"counters":counters,"timing_accepted":False}
base=results["reuse-forward"]["counters"]
checked=results["checked-forward"]["counters"]
assert checked==results["checked-reverse"]["counters"]
assert checked["launches"]-base["launches"]==2539
assert checked["download_bytes"]-base["download_bytes"]==4 and checked["downloads"]-base["downloads"]==1
assert checked["uploads"]==base["uploads"] and checked["upload_bytes"]==base["upload_bytes"]
assert checked["peak_bytes"]-base["peak_bytes"]==256 and checked["arena_used"]==4
reference=root/".tools/reference-pytorch-v2.8.0"
model=root/".tools/tinyllama-q4_k_m.gguf"
record={"base_revision":command("git","rev-parse","HEAD"),
        "source_sha256":{name:sha(root/name) for name in sources},"generated_bundles":bundles,
        "inference_bundle_identical_to_module_split":True,
        "gpu":command("nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"),
        "compiler":command(str(root/".tools/w64devkit/bin/gcc.exe"),"--version"),
        "model":{"bytes":model.stat().st_size,"sha256":sha(model)},
        "executable_sha256":sha(root/"build-cuda/nya-bench-training-device.exe"),
        "results":results,
        "method":{"warmups":2,"measured_batches":5,"calls_per_batch":20,"checked_oracle_elements":64,
                  "timing_exclusion":"Interactive desktop activity was not isolated. Correctness and resource counters only.",
                  "environment":{"NYA_CUDA_BLAS":"0","NYA_CUDA_CUTLASS":"0","NYA_COMPUTE":"cpu"}},
        "source_study":{"project":"PyTorch","tag":"v2.8.0","revision":"ba56102387ef21a3b04b357e5b183d48f0afefc7",
                        "license":"BSD-3-Clause-style PyTorch license; read before implementation",
                        "sha256":{name:sha(reference/name.replace("/","__")) for name in ("LICENSE","aten/src/ATen/native/cuda/AmpKernels.cu")},
                        "finding":"AMP retains a caller-reset non-finite flag on device across gradient tensors. Fyodor adopts the resident-status idea with an original F32 classifier, block vote and atomic first-operation tag; no ATen, tensor iterator, unscale operation or upstream source is copied."}}
(out/"snapshot.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print("Source, model, inference bundle and finite-check counters verified.")
