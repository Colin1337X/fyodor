"""Run the complete new RMSNorm suite under each CUDA sanitizer."""
import datetime
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
tool=root/".tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe"
env=dict(os.environ,NYA_CUDA_BLAS="0",NYA_CUDA_CUTLASS="0")
records=[]
for name in ("memcheck","initcheck","synccheck","racecheck"):
    assert not (out/f"{name}.log").exists()
    command=[str(tool),"--tool",name,"--error-exitcode","9","--log-file",str(out/f"{name}.log"),
             str(root/"build-cuda/nya-test-training-norm.exe"),"--cuda"]
    started=datetime.datetime.now(datetime.timezone.utc)
    with (out/f"{name}.out").open("xb") as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append({"tool":name,"command":command,"exit_code":result.returncode,
                    "started_utc":started.isoformat(),"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
    (out/"sanitizers.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    print(name,result.returncode,flush=True)
    if result.returncode: raise SystemExit(result.returncode)
