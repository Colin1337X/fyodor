"""Preserve the 8K replay failure with the new split path explicitly disabled."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if (out/'replay-control.json').exists():raise SystemExit('Refusing to overwrite evidence')
source=(out/'model.c').read_text()
needle='        CHECK(!nya_llm_session_verify(s,&replacement,1,actual+1,v));'
assert source.count(needle)==1
source=source.replace(needle,needle+'''
        double replay_max=0;size_t replay_index=0;
        for(size_t i=0;i<v;++i) {
            float a=actual[i+1],b=reference[(count+1)*v+i];
            if(a==-INFINITY && b==-INFINITY)continue;
            double delta=fabs((double)a-b)/(1+fabs(b));
            if(delta>replay_max) {replay_max=delta;replay_index=i;}
        }
        fprintf(stderr,"replacement replay max_scaled=%.9g index=%zu actual=%.9g serial=%.9g\\n",
            replay_max,replay_index,(double)actual[replay_index+1],(double)reference[(count+1)*v+replay_index]);
''')
model=out/'replay_control.c';model.write_text(source)
exe=root/'build-cuda/wide-batch-split-replay-control.exe';compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
build=[str(compiler),'-std=c17','-O3','-Wall','-Wextra','-Werror','-I','backend/include','-I','backend/core',str(model),'build-cuda/libnya-engine.a','-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
subprocess.run(build,env=env,check=True)
settings=json.loads((out/'validation.json').read_text())['environment']
settings.pop('NYA_TEST_VERIFY_BATCH_SPLIT',None)
settings.update(NYA_CUDA_WIDE_BATCH_SPLIT='0',NYA_CUDA_RING_KV='0',NYA_TEST_VERIFY_TWO='1')
env.update(settings);env.pop('NYA_TEST_VERIFY_BATCH_SPLIT',None)
command=[str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','8192']
with (out/'replay-control.log').open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
meta={'build_command':build,'command':command,'environment':settings,'exit_code':result.returncode,
      'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,exe,root/'build-cuda/libnya-engine.a')}}
(out/'replay-control.json').write_text(json.dumps(meta,indent=2)+'\n')
print((out/'replay-control.log').read_text(),flush=True)
raise SystemExit(result.returncode)
