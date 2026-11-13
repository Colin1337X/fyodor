"""Report planner admission under the prefix-control settings; no timing claim."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
target=out/'plan-capacity.json';log_path=out/'plan-capacity.log'
if target.exists() or log_path.exists():raise SystemExit('Refusing to overwrite evidence')
source=out/'plan_capacity.c';library=root/'build-cuda/libnya-engine.a'
compiler=root/'.tools/w64devkit/bin/gcc.exe';exe=root/'build-cuda/plan-capacity.exe'
env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
settings=json.loads((out/'prefix-kv-control.json').read_text())['environment']
env.update(settings)
build=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Wconversion','-Wshadow','-Werror',
       '-I','backend/include','-I','backend/core',str(source),str(library),'-lbcrypt',
       'C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(build,env=env,check=True)
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,library,exe)}
command=[str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf']
with log_path.open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
rows=[json.loads(line) for line in log_path.read_text().splitlines() if line.startswith('{')]
with target.open('x') as output:json.dump({'accepted_benchmark':False,'build':build,'command':command,
    'environment':settings,'sha256':hashes,'exit_code':result.returncode,'rows':rows},output,indent=2)
if result.returncode:raise SystemExit(result.returncode)
assert [r['prefix'] for r in rows]==[1024,4096,8192]
assert all(r['prefill_batch']>0 and r['weights_bytes']>0 and r['kv_bytes']>0 for r in rows)
print(json.dumps(rows,indent=2))
