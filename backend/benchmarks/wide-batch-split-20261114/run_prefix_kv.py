"""Locate batched/serial prefix drift without enabling experimental attention."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or sys.argv[1] not in ('control','native-matmul','unsplit-attention'):
 raise SystemExit('Choose control, native-matmul or unsplit-attention')
mode=sys.argv[1];meta_path=out/f'prefix-kv-{mode}.json'
if meta_path.exists():raise SystemExit('Refusing to overwrite evidence')
compiler=root/'.tools/w64devkit/bin/gcc.exe';exe=root/f'build-cuda/prefix-kv-{mode}.exe'
source=out/'prefix_kv.c';env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
build=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(source),'build-cuda/libnya-engine.a','-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(build,env=env,check=True)
settings=json.loads((out/'replay-control.json').read_text())['environment'].copy()
settings={k:v for k,v in settings.items() if not k.startswith('NYA_TEST_')}
if mode=='native-matmul':settings['NYA_CUDA_BLAS']='0'
elif mode=='unsplit-attention':settings.update(NYA_CUDA_WIDE_SPLIT='0',NYA_CUDA_FLASH_DECODING_PP='0')
env.update(settings)
command=[str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf']
path=out/f'prefix-kv-{mode}.log'
with path.open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
rows=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
meta={'build':build,'command':command,'environment':settings,'exit_code':result.returncode,
      'diagnostic_only':True,'rows':rows,'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,exe,root/'build-cuda/libnya-engine.a')}}
meta_path.write_text(json.dumps(meta,indent=2)+'\n')
if result.returncode:raise SystemExit(result.returncode)
assert len([r for r in rows if r['kind']=='kv'])==144 and len([r for r in rows if r['kind']=='logits'])==1
print(json.dumps(rows[0]),flush=True)
