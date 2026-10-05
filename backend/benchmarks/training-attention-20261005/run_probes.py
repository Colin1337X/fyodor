"""Real-model attention-branch correctness/resource probes; timings are not accepted."""
from pathlib import Path
import subprocess,json,os,datetime
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
env=dict(os.environ,NYA_CUDA_BLAS="0",NYA_CUDA_CUTLASS="0",NYA_COMPUTE="cpu")
records=[]
for name,mode,reference in (("attention8","attention8",False),("attention64","attention64",False),("attention8-reference","attention8",True),("qkrope8","qkrope8",False)):
 runenv=env.copy()
 runenv["NYA_CUDA_REFERENCE"]="1" if reference else "0"
 command=[str(root/"build-cuda/nya-bench-training-device.exe"),str(root/".tools/tinyllama-q4_k_m.gguf"),mode]
 start=datetime.datetime.now(datetime.timezone.utc)
 with (out/(name+".out")).open("xb") as stdout,(out/(name+".log")).open("xb") as stderr:
  result=subprocess.run(command,cwd=root,env=runenv,stdout=stdout,stderr=stderr)
 records.append({"name":name,"command":command,"exit_code":result.returncode,"started_utc":start.isoformat(),"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
 (out/"probes.json").write_text(json.dumps(records,indent=2))
 print(name,result.returncode,flush=True)
 if result.returncode: raise SystemExit(result.returncode)
