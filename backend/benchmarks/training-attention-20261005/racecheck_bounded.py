import os,subprocess,json,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
tool=root/".tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe"
command=[str(tool),"--tool","racecheck","--error-exitcode","9","--force-synchronization-limit","4","--racecheck-num-workers","2","--log-file",str(out/"racecheck-bounded.log"),str(root/"build-cuda/nya-test-training-attention.exe"),"--cuda"]
start=datetime.datetime.now(datetime.timezone.utc)
with (out/"racecheck-bounded.out").open("xb") as f:
 p=subprocess.run(command,cwd=root,env=dict(os.environ,NYA_TRAIN_TEST_TRACE="1",NYA_CUDA_BLAS="0",NYA_CUDA_CUTLASS="0"),stdout=f,stderr=subprocess.STDOUT)
(out/"racecheck-bounded.json").write_text(json.dumps({"command":command,"exit_code":p.returncode,"started_utc":start.isoformat(),"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print("racecheck traced",p.returncode,flush=True)
