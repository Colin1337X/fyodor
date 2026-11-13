"""Production-core real Gemma generation parity with assisted FFN enabled."""
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'production-short.json').exists():raise SystemExit('Refusing to overwrite evidence')
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
model=out/'generation.c';model.write_bytes((out.parent/'mtp-cost-20261113/model.c').read_bytes())
exe=root/'build-cuda/assisted-ffn-generation.exe'
command=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(model),'build-cuda/libnya-engine.a','-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(command,env=env,check=True)
settings=json.loads((out.parent/'batch-logits-20261114/runtime-on/timing.json').read_text())['environment']
settings.update(NYA_CUDA_ASSISTED_FFN='1',NYA_CUDA_PROFILE='0');env.update(settings)
with (out/'short.stdout').open('w') as stdout,(out/'short.stderr').open('w') as stderr:
 result=subprocess.run([str(exe),'short'],env=env,stdout=stdout,stderr=stderr)
meta={'command':command,'environment':settings,'exit_code':result.returncode,'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (exe,model,root/'build-cuda/libnya-engine.a',root/'CUDA/assisted_ffn.inc',root/'backend/core/llm_cpu.c')}}
(out/'production-short.json').write_text(json.dumps(meta,indent=2)+'\n')
assert result.returncode==0
rows=[json.loads(line) for line in (out/'short.stdout').read_text().splitlines() if line.startswith('{"round":')]
assert len(rows)==12 and all(r['parity'] and r['generated']==32 for r in rows)
trace=(out/'short.stderr').read_text()
calls=re.findall(r'CUDA assisted feed-forward calls: (\d+)',trace)
assert len(calls)==1 and int(calls[0])>0
print('12 real generation calls passed; assisted FFN calls:',calls[0],flush=True)
