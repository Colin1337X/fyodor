"""Validate resident device timing and graph evaluation recovery under each CUDA sanitizer."""
import hashlib
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,help='Use a fresh evidence directory; existing logs are never overwritten')
parser.add_argument('--suite',choices=('graph',),default='graph')
args=parser.parse_args()
if args.output:
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
tool=root/".tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe"
env=dict(os.environ,NYA_CUDA_BLAS="0",NYA_CUDA_CUTLASS="0")
records=[]
for tool_name,mode in ((t,m) for t in ("memcheck","initcheck","synccheck","racecheck") for m in ("device","lifecycle")):
    name=tool_name+"-"+mode
    assert not (out/f"{name}.log").exists()
    command=[str(tool),"--tool",tool_name,"--error-exitcode","9","--log-file",str(out/f"{name}.log"),
             str(root/f"build-cuda/nya-test-training-{args.suite}.exe"),"--cuda"]
    if tool_name=="racecheck": command[3:3]=["--force-synchronization-limit","4","--racecheck-num-workers","2"]
    executable=root/("build-cuda/nya-test-training-device.exe" if mode=="device" else "build-cuda/nya-test-training-graph.exe")
    command=command[:-2]+[str(executable)]+(["--cuda"] if mode=="device" else ["--cuda",str(out/name)])
    started=datetime.datetime.now(datetime.timezone.utc)
    with (out/f"{name}.out").open("xb") as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append({"tool":tool_name,"suite":mode,"log":name+".log","output":name+".out","command":command,"exit_code":result.returncode,
                    "executable_sha256":hashlib.sha256(executable.read_bytes()).hexdigest(),
                    "started_utc":started.isoformat(),"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
    (out/"sanitizers.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    print(name,result.returncode,flush=True)
    if result.returncode: raise SystemExit(result.returncode)
