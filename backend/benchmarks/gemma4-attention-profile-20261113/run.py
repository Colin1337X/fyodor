from pathlib import Path
import os,sys,subprocess,json,hashlib,re
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'backend/benchmarks/gemma4-qkv-20261029'))
import environment
exe=root/'build-cuda/fyodor-bench.exe'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((root/'backend/benchmarks/gemma4-ring-timing-20261113/raw/metadata.json').read_text())['environment'])
env['NYA_CUDA_PROFILE']='1'
model='E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'
meta={'accepted_performance':False,'reason':'CUDA event profiling disables graphs; desktop GPU activity retained. Diagnostic attribution only.','base_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((root/'CUDA/resident.inc').read_bytes()).hexdigest(),'runs':[]}
for prefix in (8192,):
 for variant in ('linear','ring'):
  name=f'{prefix}-{variant}';active=dict(env);active.update(NYA_CUDA_RING_KV='1' if variant=='ring' else '0',NYA_CUDA_PROFILE_FROM=str(prefix))
  cmd=[str(exe),'-m',model,'-b','cuda','-p','0','-n','16','--context',str(prefix),'-r','1','--warmup','0','--json']
  before=environment.snapshot();print('starting',name,flush=True)
  p=subprocess.run(cmd,env=active,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
  (out/(name+'.json')).write_bytes(p.stdout);(out/(name+'.log')).write_bytes(p.stderr)
  meta['runs'].append({'name':name,'command':cmd,'environment':{k:v for k,v in active.items() if k.startswith('NYA_')},'before':before,'after':environment.snapshot(),'exit_code':p.returncode})
  (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
  assert p.returncode==0,p.stderr.decode(errors='replace')
  d=json.loads(p.stdout);assert d['execution']=='resident' and d['kv_type']=='f32'
  scopes=re.findall(rb'CUDA profile scope: from_position=(\d+) batches=(\d+) tokens=(\d+)',p.stderr)
  assert scopes==[(str(prefix).encode(),b'16',b'16')],scopes
  line=re.search(rb'CUDA kernel GPU milliseconds: ([^\r\n]+)',p.stderr)
  print(name,line[1].decode(),flush=True)
