"""Capture exact source/build/binary evidence after validation, without timing claims."""
from pathlib import Path
import hashlib,json,re,subprocess,datetime
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def command(args):return subprocess.check_output(args,cwd=root,text=True).strip()
previous=json.loads((out.parent/"training-session-20261006/snapshot.json").read_text(encoding="utf-8"))
files=set(previous["source_sha256"])|{"CUDA/cuda.c","backend/core/training_device_backend.h","backend/core/training_graph_device.h","backend/tests/test_training_graph.c","backend/tests/test_training_device.c"}
record={"base_revision":command(["git","rev-parse","HEAD"]),"captured_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_sha256":{p:sha(root/p) for p in sorted(files)}}
record["build_configurations"]={}
for build,old in previous["build_configurations"].items():
    cache=dict(line.split("=",1) for line in (root/build/"CMakeCache.txt").read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith(("//","#")))
    record["build_configurations"][build]={k:cache[k] for k in old if k in cache}
    record["build_configurations"][build]["engine_flags"]=(root/build/"CMakeFiles/nya-engine.dir/flags.make").read_text(encoding="utf-8")
header=(root/"build-cuda/cuda-provider/cuda_source.h").read_text(encoding="utf-8")
record["generated_bundles"]={}
for name in ("nya_cuda_source","nya_cuda_training_source"):
    match=re.search(re.escape(name)+r"\[\] = \{([^}]+)\}",header)
    data=bytes(int(x.strip(),0) for x in match.group(1).split(",") if x.strip())
    # Include the generated C string terminator, matching earlier evidence.
    record["generated_bundles"][name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
assert record["generated_bundles"]["nya_cuda_source"]==previous["generated_bundles"]["nya_cuda_source"]
record["compiler"]=command([str(root/".tools/w64devkit/bin/gcc.exe"),"--version"])
record["gpu"]=command(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"])
record["gpu_activity_observation"]=command(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,memory.used,power.draw","--format=csv,noheader"])
record["executables_sha256"]={n:sha(root/"build-cuda"/n) for n in ("nya-test-training-graph.exe","fyodor-backend.exe")}
record["packaged_trainer_sha256"]=sha(root/"build-cpu/fyodor-train.exe")
record["model_sha256"]=sha(root/".tools/tinyllama-q4_k_m.gguf")
record["exported_models"]={str(p):{"bytes":p.stat().st_size,"sha256":sha(p)} for p in Path("D:/fyodor-validation/resident-training-graph-20261007").glob("tinyllama-aligned*.gguf")}
record["method"]={"random_steps":100,"fresh_resume_step":50,"cpu_state_scaled_tolerance":1e-5,"aligned_export_scaled_tolerance":1e-5,"tiny_gpu_resume_comparison":"Byte-identical complete portable checkpoint","timing_accepted":False,"public_training":"CPU"}
(out/"snapshot.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print("Captured graph source/build/binary evidence; inference CUDA module remains byte-identical.")
