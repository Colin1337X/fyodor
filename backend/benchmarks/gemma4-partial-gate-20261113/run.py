from pathlib import Path
import os,subprocess,json,hashlib
root=Path.cwd();out=root/'backend/benchmarks/gemma4-partial-gate-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((root/'backend/benchmarks/gemma4-gate-20261113/model-oracle-exit.json').read_text())['environment'])
env.update(NYA_CUDA_BYTE_DOT='0',NYA_CUDA_GEMMA_GATE_PARTIAL='1')
exe=root/'build-cuda/gemma-partial-gate-oracle.exe';runs=[]
for name,count,expected,budget in [('partial-short',3,'partial','9000'),('no-pair-short',3,'fallback','7050'),('partial-8k',8194,'partial',None)]:
    active=dict(env)
    if budget:active['NYA_CUDA_MEMORY_MIB']=budget
    cmd=[str(exe),str(count),expected]
    p=subprocess.run(cmd,env=active,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out/(name+'.log')).write_bytes(p.stdout+p.stderr)
    runs.append(dict(name=name,command=cmd,exit_code=p.returncode,environment={k:v for k,v in active.items() if k.startswith('NYA_')}))
    (out/'results.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'runs':runs},indent=2)+'\n')
    print(name,p.returncode,p.stdout.decode(errors='replace'),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
