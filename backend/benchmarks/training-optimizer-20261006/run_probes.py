"""Real-model LoRA/AdamW trajectory correctness/resource probes; timings are not accepted."""
from pathlib import Path
import subprocess,json,os,datetime,argparse,hashlib
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path)
parser.add_argument('--regressions',action='store_true')
args=parser.parse_args()
if args.output:
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'probes.json').exists()
env=dict(os.environ,NYA_CUDA_BLAS="0",NYA_CUDA_CUTLASS="0",NYA_COMPUTE="cpu")
records=[]
jobs=(("embedffn8","embedffn8",False),("attention64","attention64",False),("dpo64","dpo64",False)) if args.regressions else (("adamw8","adamw8",False),("adamw64","adamw64",False),("adamw8-reference","adamw8",True))
for name,mode,reference in jobs:
 runenv=env.copy()
 runenv["NYA_CUDA_REFERENCE"]="1" if reference else "0"
 command=[str(root/"build-cuda/nya-bench-training-device.exe"),str(root/".tools/tinyllama-q4_k_m.gguf"),mode]
 start=datetime.datetime.now(datetime.timezone.utc)
 with (out/(name+".out")).open("xb") as stdout,(out/(name+".log")).open("xb") as stderr:
  result=subprocess.run(command,cwd=root,env=runenv,stdout=stdout,stderr=stderr)
 records.append({"name":name,"command":command,"executable_sha256":hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),"exit_code":result.returncode,"started_utc":start.isoformat(),"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
 (out/"probes.json").write_text(json.dumps(records,indent=2))
 print(name,result.returncode,flush=True)
 if result.returncode: raise SystemExit(result.returncode)
