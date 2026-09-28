"""Record resident RMSNorm and real-FFN evidence without timing acceptance."""
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
         "backend/core/training_device_backend.h","backend/tests/test_training_norm.c",
         "backend/tests/test_training_device.c","backend/benchmarks/training_device_bench.c",
         "backend/core/training.c","backend/core/train_executor.c"]
header=(root/"build-cuda/backend/cuda-provider/cuda_source.h").read_text()
bundles={}
for name in ("nya_cuda_source","nya_cuda_training_source"):
    values=re.search(re.escape(name)+r"\[\] = \{([^}]+)\}",header)[1]
    data=bytes(int(value,0) for value in values.split(","))
    bundles[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
previous=json.loads((root/"backend/benchmarks/cuda-modules-20260927/source-bundles.json").read_text())
assert bundles["nya_cuda_source"]==previous["nya_cuda_source"]
results={}
for mode in ("rmsffn8","rmsffn64"):
    result=json.loads((out/f"{mode}.out").read_text())
    assert result["max_scaled_output_error"]<=result["tolerance"]==1e-5
    assert result["max_scaled_input_gradient_error"]<=1e-5
    assert result["max_scaled_norm_gradient_error"]<=1e-5
    assert len(result["ms"])==5 and not result["timing_accepted"]
    results[mode]={"command":["build-cuda/nya-bench-training-device.exe",".tools/tinyllama-q4_k_m.gguf",mode],
                   "exit_code":0,"result":result}
reference=root/".tools/reference-llama-b10809"
model=root/".tools/tinyllama-q4_k_m.gguf"
record={"base_revision":command("git","rev-parse","HEAD"),
        "source_sha256":{name:sha(root/name) for name in sources},"generated_bundles":bundles,
        "inference_bundle_identical_to_module_split":True,
        "gpu":command("nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"),
        "compiler":command(str(root/".tools/w64devkit/bin/gcc.exe"),"--version"),
        "model":{"bytes":model.stat().st_size,"sha256":sha(model)},
        "executable_sha256":sha(root/"build-cuda/nya-bench-training-device.exe"),
        "results":results,
        "method":{"warmups":2,"measured_repetitions":5,"microbatches_per_repetition":2,
                  "checked_elements":"Every final output, accumulated input-gradient and normalization-scale gradient element",
                  "oracle":"Native CPU autograd, six workers, same mapped quantized weights and inputs",
                  "timing_exclusion":"Interactive desktop activity was not isolated; no throughput acceptance.",
                  "environment":{"NYA_CUDA_BLAS":"0","NYA_CUDA_CUTLASS":"0","NYA_COMPUTE":"cpu"}},
        "source_study":{"project":"llama.cpp/ggml","revision":command("git","-C",str(reference),"rev-parse","HEAD"),
                        "license":"MIT",
                        "sha256":{name:sha(reference/name) for name in ("LICENSE","ggml/src/ggml-cuda/norm.cu","ggml/src/ggml-cuda/norm.cuh")},
                        "finding":"Upstream RMSNorm forward/backward uses per-row block reductions. Fyodor uses original double reductions, caller-owned saved inverses, optional trainable scales and ordered accumulating gradients. No upstream source or C++ tensor machinery is copied."}}
(out/"snapshot.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print("Source, inference bundle, real-FFN parity and resource evidence verified.")
