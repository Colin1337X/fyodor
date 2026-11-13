"""Run unchanged lifecycle assertions against the isolated aligned library."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=4:raise SystemExit('Usage: verify_aligned.py PREFIX RING COUNT')
prefix,ring,count=map(int,sys.argv[1:])
if prefix not in (257,8192) or ring not in (0,1) or count not in (2,4):raise SystemExit('Unsupported case')
baseline=json.loads((out/'aligned-prefix-ctx8192.json').read_text())
assert baseline['exit_code']==0 and baseline['parse_error'] is None
assert [r for r in baseline['rows'] if r['kind']=='logits'][0]['max_scaled']<=.0005
library=root/'build-cuda/aligned-prefix-ctx8192/libnya-engine.a'
assert hashlib.sha256(library.read_bytes()).hexdigest()==baseline['sha256'][str(library.relative_to(root))]
tag=f'aligned-verify-prefix{prefix}-ring{ring}-count{count}'
paths=[out/f'{tag}.{ext}' for ext in ('json','log','stderr.log','c')]
if any(p.exists() for p in paths):raise SystemExit('Refusing to overwrite evidence')
source=(root/'backend/tests/test_verify.c').read_text()
assert source.count('size_t count=17,')==1
source=source.replace('size_t count=17,',f'size_t count={count},')
paths[3].write_text(source)
compiler=root/'.tools/w64devkit/bin/gcc.exe';exe=root/f'build-cuda/{tag}.exe'
env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
settings={k:v for k,v in baseline['environment'].items() if not k.startswith('NYA_TEST_')}
settings.update(NYA_CUDA_RING_KV=str(ring),NYA_TEST_VERIFY_RESIDENT='1')
env.update(settings)
build=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Wconversion','-Wshadow','-Werror',
       '-I','backend/include','-I','backend/core',str(paths[3]),str(library),'-lbcrypt',
       'C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(build,env=env,check=True)
command=[str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf',str(prefix)]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (paths[3],library,exe)}
with paths[1].open('x') as stdout,paths[2].open('x') as stderr:
 result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
with paths[0].open('x') as output:json.dump({'accepted_benchmark':False,'prefix':prefix,'ring':ring,
 'count':count,'build':build,'command':command,'environment':settings,'sha256':hashes,
 'exit_code':result.returncode,'note':'All original test_verify lifecycle assertions and the 0.0005 bound are retained; only verification token count is specialized.'},output,indent=2)
if result.returncode:raise SystemExit(result.returncode)
print(tag,'passed',flush=True)
