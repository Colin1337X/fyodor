from pathlib import Path
import os,json,subprocess,hashlib
root=Path.cwd();out=root/'backend/benchmarks/gemma4-partial-gate-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))};env.update(json.loads((out/'results.json').read_text())['runs'][0]['environment']);env['NYA_CUDA_BYTE_DOT']='1'
exe=root/'build-cuda-c23/gemma-partial-allocation-oracle.exe';runs=[]
for mode in ['partial','fallback']:
    cmd=[str(exe),'3',mode];p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out/('c23-'+mode+'.log')).write_bytes(p.stdout+p.stderr)
    runs.append(dict(command=cmd,exit_code=p.returncode));print(mode,p.returncode,p.stdout.decode(errors='replace'),flush=True)
    (out/'allocation-results.json').write_text(json.dumps(dict(runs=runs,environment={k:v for k,v in env.items() if k.startswith('NYA_')},binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((out/'allocation_oracle.c').read_bytes()).hexdigest(),injection='fallback mode sets NYA_CUDA_FAIL_ALLOCATION_AFTER=1 only for the second session, after weights were cached by the baseline'),indent=2)+'\n')
    if p.returncode:raise SystemExit(p.returncode)
