import json,os,subprocess,hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root/'backend/benchmarks/score-refresh-20261011'))
import environment
environment.configure(out)
@environment.capture('benchmark')
def run(cmd,env):
 p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 try:
  environment.set_active_pid(p.pid)
  stdout,stderr=p.communicate(timeout=180)
 except BaseException:
  p.kill();p.communicate();raise
 return subprocess.CompletedProcess(cmd,p.returncode,stdout,stderr)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='512',NYA_CUDA_BLAS='1',NYA_CUDA_CUTLASS='0',NYA_CUDA_BLAS_LIBRARY=str(root/'.tools/llama-b10809-cuda/cublas64_13.dll'))
exes={'before':Path('D:/fyodor-validation/decode-before-residual-f5e61a2.exe'),'after':root/'build-cuda/fyodor-bench.exe'}
m={'caveat':'Monitored desktop, same strict idle gate as scorecard; graphs enabled; serial runs; no concurrent builds/tests','hashes':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in exes.items()},'runs':[]}
for prefix in (0,):
 for phase,order in [('a',('before','after')),('b',('after','before'))]:
  for name in order:
   tag=f'{prefix}-{name}-{phase}';cmd=[str(exes[name]),'-m',str(root/'.tools/tinyllama-q4_k_m.gguf'),'-b','cuda','-p','0','-n','128','--context',str(prefix),'-r','5','--warmup','2','--json']
   p=run(cmd,env)
   (out/(tag+'.json')).write_bytes(p.stdout);(out/(tag+'.log')).write_bytes(p.stderr)
   m['runs'].append({'name':tag,'command':cmd,'exit_code':p.returncode});(out/'metadata.json').write_text(json.dumps(m,indent=2))
   assert p.returncode==0
   d=json.loads(p.stdout);assert d['execution']=='resident'; print(tag,d['results'][0]['tokens_per_second'],flush=True)
