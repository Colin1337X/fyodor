"""Capture sources, toolchains, model and unchanged inference module identity."""
from pathlib import Path
import hashlib,json,re,subprocess
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def command(*args):return subprocess.check_output(args,cwd=root,text=True).strip()
names=["CUDA/training.cu","CUDA/training.inc","CUDA/cuda.c","CUDA/CMakeLists.txt","backend/CMakeLists.txt","backend/core/training_device.c","backend/core/training_device.h","backend/core/training_device_backend.h","backend/core/training.c","backend/tests/test_training_optimizer.c","backend/tests/test_training_device.c","backend/benchmarks/training_device_bench.c"]
header=(root/"build-cuda/cuda-provider/cuda_source.h").read_text()
bundles={}
for name in ("nya_cuda_source","nya_cuda_training_source"):
 values=re.search(re.escape(name)+r"\[\] = \{([^}]+)\}",header)[1]
 data=bytes(int(value,0) for value in values.split(","))
 bundles[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"crlf_count":data.count(b"\r\n"),"lf_normalized_sha256":hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()}
previous=json.loads((root/"backend/benchmarks/training-loss-20261006/snapshot.json").read_text())["generated_bundles"]["nya_cuda_source"]
assert bundles["nya_cuda_source"]["lf_normalized_sha256"]==previous["lf_normalized_sha256"]
ref=root/".tools/reference-pytorch-v2.8.0";model=root/".tools/tinyllama-q4_k_m.gguf"
configs={}
for build in ("build-cpu","build-cuda","build-cuda-c23","build-ort","build-sanitize"):
 cache=(root/build/"CMakeCache.txt").read_text()
 configs[build]={}
 for line in cache.splitlines():
  if line.startswith(("CMAKE_C_COMPILER:","CMAKE_C_COMPILER_ID:","CMAKE_C_FLAGS:","CMAKE_C_FLAGS_RELEASE:","CMAKE_C_FLAGS_DEBUG:","CMAKE_BUILD_TYPE:","CMAKE_GENERATOR:","NYA_")) and "=" in line:
   key,value=line.split("=",1); configs[build][key]=value
 configs[build]["engine_flags"]=(root/build/"CMakeFiles/nya-engine.dir/flags.make").read_text()
record={"build_configurations":configs,"base_revision":command("git","rev-parse","HEAD"),"source_sha256":{n:sha(root/n) for n in names},"generated_bundles":bundles,"historical_inference_bundle":previous,"inference_change":"Normalized inference source identical to the preceding loss milestone",
 "gpu_activity_observation":command("nvidia-smi","--query-gpu=timestamp,utilization.gpu,memory.used,power.draw","--format=csv,noheader"),
 "gpu":command("nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"),
 "compiler":command(str(root/".tools/w64devkit/bin/gcc.exe"),"--version"),"model":{"bytes":model.stat().st_size,"sha256":sha(model)},
 "executables_sha256":{n:sha(root/"build-cuda"/n) for n in ("nya-test-training-optimizer.exe","nya-test-training-device.exe","nya-bench-training-device.exe")},
 "source_study":{"project":"PyTorch","revision":"ba56102387ef21a3b04b357e5b183d48f0afefc7","license":"BSD-style; LICENSE reviewed before source study","sha256":{n:sha(ref/n) for n in ("LICENSE","aten__src__ATen__native__cuda__fused_adamw_impl.cu","aten__src__ATen__native__cuda__fused_adam_utils.cuh","aten__src__ATen__native__cuda__MultiTensorApply.cuh")},"finding":"Studied chunk-to-tensor scheduling and found_inf guards. Original Fyodor kernels reserve complete staging and separate norm, prepare, stage, commit and step-publication phases, retaining double CPU AdamW semantics and all-or-nothing numerical rejection. No source copied."},
 "method":{"trajectory_steps":32,"workloads":["adamw8","adamw64","adamw8-reference"],"tolerance":1e-5,"oracle":"Six-worker CPU autograd and AdamW; real frozen Q6_K vocabulary weights with trainable rank-four F32 LoRA matrices. Scalar losses and every trainable value/gradient checked after every step.","timing_accepted":False,"reason":"Interactive desktop not isolated; trajectory correctness and resource evidence only, not full-model training throughput."}}

(out/"snapshot.json").write_text(json.dumps(record,indent=2))
print("Source and inference-module identity recorded.")
