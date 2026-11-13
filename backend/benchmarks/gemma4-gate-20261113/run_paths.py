from pathlib import Path
import os,subprocess,json,hashlib
root=Path.cwd(); out=root/'backend/benchmarks/gemma4-gate-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((out/'model-oracle-exit.json').read_text())['environment'])
env['NYA_CUDA_BYTE_DOT']='0'
exe=root/'build-cuda/gemma-gate-path-oracle.exe'
runs=[]
for name,count,expected,budget in [('gate-only-short',3,'fused',None),('gate-only-1k',1026,'fused',None),('budget-fallback-short',3,'fallback','9000'),('gate-only-8k',8194,'fallback',None)]:
    active=dict(env)
    if budget:active['NYA_CUDA_MEMORY_MIB']=budget
    cmd=[str(exe),str(count),expected]
    p=subprocess.run(cmd,env=active,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out/(name+'.log')).write_bytes(p.stdout+p.stderr)
    runs.append(dict(name=name,command=cmd,exit_code=p.returncode,environment={k:v for k,v in active.items() if k.startswith('NYA_')}))
    (out/'path-oracle-results.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'runs':runs},indent=2)+'\n')
    print(name,p.returncode,p.stdout.decode(errors='replace'),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
