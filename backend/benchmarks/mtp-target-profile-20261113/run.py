"""Profile decode after cleanup; graph-disabled timings are not benchmark scores."""
import hashlib,json,os,subprocess
from pathlib import Path

root=Path.cwd()
out=Path(__file__).resolve().parent
if (out/'profile.json').exists():
    raise SystemExit('Refusing to overwrite profiling evidence')
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=os.environ.copy()
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
exe=root/'build-cuda/mtp-target-profile.exe'
command=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror',
         '-I','backend/include','-I','backend/core',str(out/'model.c'),
         'build-cuda/mtp-cost-candidate-core.o','build-cuda/libnya-engine.a',
         '-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(command,env=env,check=True)
settings=json.loads((root/'backend/benchmarks/small-batch-reuse-20261113/runtime-on/timing.json').read_text())['environment']
settings.update(NYA_CUDA_PROFILE='1',NYA_CUDA_MTP_WIDE_VALUE='0')
meta={'accepted_llama_score':False,'command':command,'environment':settings,
      'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'runs':[]}
for context,threshold in [('short','25'),('long','8205')]:
    env.update(settings)
    env['NYA_CUDA_PROFILE_FROM']=threshold
    with (out/(context+'.stdout')).open('w') as stdout,(out/(context+'.stderr')).open('w') as stderr:
        result=subprocess.run([str(exe),context],env=env,stdout=stdout,stderr=stderr)
    meta['runs'].append({'context':context,'threshold':threshold,'exit_code':result.returncode})
    (out/'profile.json').write_text(json.dumps(meta,indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
    rows=[json.loads(line) for line in (out/(context+'.stdout')).read_text().splitlines() if line.startswith('{"round":')]
    assert len(rows)==4 and all(row['parity'] and row['generated']==32 for row in rows)
    assert 'path=small-batch-weight-reuse' in (out/(context+'.stderr')).read_text()
    print(context,'parity and resident reuse dispatch verified',flush=True)
