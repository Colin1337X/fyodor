"""Capture sources, toolchains, model and unchanged inference module identity."""
from pathlib import Path
import hashlib,json,re,subprocess
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def command(*args):return subprocess.check_output(args,cwd=root,text=True).strip()
names=["CUDA/training.cu","CUDA/training.inc","CUDA/cuda.c","CUDA/CMakeLists.txt","backend/CMakeLists.txt","backend/core/training_device.c","backend/core/training_device.h","backend/core/training_device_backend.h","backend/core/training.c","backend/tests/test_training_loss.c","backend/benchmarks/training_device_bench.c"]
header=(root/"build-cuda/cuda-provider/cuda_source.h").read_text()
bundles={}
for name in ("nya_cuda_source","nya_cuda_training_source"):
 values=re.search(re.escape(name)+r"\[\] = \{([^}]+)\}",header)[1]
 data=bytes(int(value,0) for value in values.split(","))
 bundles[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"crlf_count":data.count(b"\r\n"),"lf_normalized_sha256":hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()}
previous=json.loads((root/"backend/benchmarks/cuda-modules-20260927/source-bundles.json").read_text())
# The fresh Git worktree checks out CRLF; the historical build mixed LF/CRLF.
reference_header=Path("C:/Users/Colin/fyodor/build-cuda/backend/cuda-provider/cuda_source.h").read_text()
values=re.search(r"nya_cuda_source\[\] = \{([^}]+)\}",reference_header)[1]
reference_data=bytes(int(value,0) for value in values.split(","))
assert {"bytes":len(reference_data),"sha256":hashlib.sha256(reference_data).hexdigest()}==previous["nya_cuda_source"]
assert bundles["nya_cuda_source"]["lf_normalized_sha256"]==hashlib.sha256(reference_data.replace(b"\r\n",b"\n")).hexdigest()
ref=root/".tools/reference-pytorch-v2.8.0";model=root/".tools/tinyllama-q4_k_m.gguf"
configs={}
for build in ("build-cpu","build-cuda","build-cuda-c23","build-ort","build-sanitize"):
 cache=(root/build/"CMakeCache.txt").read_text()
 configs[build]={}
 for line in cache.splitlines():
  if line.startswith(("CMAKE_C_COMPILER:","CMAKE_C_COMPILER_ID:","CMAKE_C_FLAGS:","CMAKE_C_FLAGS_RELEASE:","CMAKE_C_FLAGS_DEBUG:","CMAKE_BUILD_TYPE:","CMAKE_GENERATOR:","NYA_")) and "=" in line:
   key,value=line.split("=",1); configs[build][key]=value
 configs[build]["engine_flags"]=(root/build/"CMakeFiles/nya-engine.dir/flags.make").read_text()
record={"build_configurations":configs,"base_revision":command("git","rev-parse","HEAD"),"source_sha256":{n:sha(root/n) for n in names},"generated_bundles":bundles,"historical_inference_bundle":previous["nya_cuda_source"],"inference_change":"Only checkout CRLF expansion: normalized bytes identical",
 "gpu_activity_observation":command("nvidia-smi","--query-gpu=timestamp,utilization.gpu,memory.used,power.draw","--format=csv,noheader"),
 "gpu":command("nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"),
 "compiler":command(str(root/".tools/w64devkit/bin/gcc.exe"),"--version"),"model":{"bytes":model.stat().st_size,"sha256":sha(model)},
 "executables_sha256":{n:sha(root/"build-cuda"/n) for n in ("nya-test-training-loss.exe","nya-bench-training-device.exe")},
 "source_study":{"project":"PyTorch","revision":"ba56102387ef21a3b04b357e5b183d48f0afefc7","license":"BSD-style (LICENSE retained in source-study cache)","sha256":{n:sha(ref/n) for n in ("LICENSE","aten__src__ATen__native__cuda__SoftMax.cu","aten__src__ATen__native__LossNLL.cpp")},"finding":"Studied per-row stable max/sum reductions and masked hard-label NLL reduction semantics. Original Fyodor kernels retain double loss arithmetic, O(rows) saved state, CPU row-order scalar sum and ordered F32 gradient accumulation. No source copied."},
 "method":{"warmups":2,"measured_repetitions":5,"microbatches_per_repetition":2,"tolerance":1e-5,"oracle":"Six-worker CPU autograd; real quantized vocabulary projection plus masked CE or paired logprob-DPO; full logits, scalar loss and accumulated input gradients after every repetition","timing_accepted":False,"reason":"Interactive desktop and build activity not isolated; component correctness/resource evidence only"}}

(out/"snapshot.json").write_text(json.dumps(record,indent=2))
print("Source and inference-module identity recorded.")
