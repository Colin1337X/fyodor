from pathlib import Path
import os,subprocess,json
r=Path(__file__).resolve().parents[3];p=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env['PATH']=str(r/'.tools/w64devkit/bin')+';'+env['PATH']
results={}
for standard in ('c17','c23'):
 exe=r/'build-cuda'/('pack-test-'+standard+'.exe')
 cmd=[str(r/'.tools/w64devkit/bin/gcc.exe'),'-std='+standard,'-O3','-Wall','-Wextra','-Werror',str(p/'pack_test.c'),'-o',str(exe)]
 with (p/('pack-'+standard+'.log')).open('wb') as f:
  for command in (cmd,[str(exe)]):
   run=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60);assert run.returncode==0
 results[standard]='passed'
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0',NYA_CUDA_FLASH_DECODING_PP='short',NYA_CUDA_INTEGER_GATE='1',NYA_CUDA_INTEGER_DOT='1',NYA_CUDA_INTEGER_Q6='1',NYA_CUDA_DEBUG='1')
cmd=[str(r/'build-cuda/fyodor-bench.exe'),'-m',str(r/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','0','-n','128','--context','0','-r','1','--warmup','0','--json']
base=None
for label in ('baseline','packed','allocation-failure','budget-fallback'):
 case=env.copy();case['NYA_CUDA_BYTE_GATE']='0' if label=='baseline' else '1'
 if label=='allocation-failure':case['NYA_CUDA_FAIL_ALLOCATION_AFTER']=str(1+sum(base['quant_tensor_counts'].values()))
 if label=='budget-fallback':case['NYA_CUDA_MEMORY_MIB']='900'
 run=subprocess.run(cmd,env=case,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
 (p/(label+'.json')).write_bytes(run.stdout);(p/(label+'.log')).write_bytes(run.stderr)
 assert run.returncode==0,(label,run.returncode)
 data=json.loads(run.stdout);assert data['execution']=='resident'
 if label=='baseline':base=data
 if label=='packed':assert data['device_weights_bytes']>base['device_weights_bytes'] and b'aligned Q4 gate copy' in run.stderr
 if label in ('allocation-failure','budget-fallback'):assert data['device_weights_bytes']==base['device_weights_bytes'] and b'aligned Q4 gate copy' not in run.stderr
 results[label]={'device_weights_bytes':data['device_weights_bytes'],'execution':data['execution'],'environment':{k:v for k,v in case.items() if k.startswith('NYA_')}}
 (p/'extra-results.json').write_text(json.dumps(results,indent=2)+'\n');print(label,'passed',flush=True)
