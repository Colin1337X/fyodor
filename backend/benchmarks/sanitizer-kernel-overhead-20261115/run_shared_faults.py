import hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;san=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
if (out/'shared-fault-checks.json').exists():raise SystemExit('Refusing overwrite')
env=os.environ.copy();env.pop('NV_COMPUTE_SANITIZER_BINARY_PATCHING',None);records=[]
for mode,file in [('Runtime','shared-oob.cubin'),('CompileTime','shared-oob.memcheck.cubin')]:
 env['FYODOR_SHARED_FIXTURE']=file
 for kind,kernel in [('Static','nya_attention_cached_query_probe'),('Dynamic','unrelated_probe')]:
  label=mode+kind+'Fault';command=[str(san),'--tool','memcheck','--error-exitcode','99',sys.executable,'-B',str(out/'probe_shared_oob.py'),kernel]
  r=subprocess.run(command,env=env,capture_output=True,text=True)
  for suffix,value in (('.stdout',r.stdout),('.stderr',r.stderr)):
   with (out/(label+suffix)).open('x') as f:f.write(value)
  summaries=re.findall(r'^========= ERROR SUMMARY: (\d+) errors$',r.stdout+r.stderr,re.M)
  detected=r.returncode==99 and len(summaries)==1 and int(summaries[0])>0 and 'Invalid __shared__ write' in r.stdout+r.stderr
  records.append(dict(label=label,command=command,environment_override={'FYODOR_SHARED_FIXTURE':file},exit_code=r.returncode,detected=detected,summaries=summaries));print(label,detected,r.returncode,summaries,flush=True)
paths=[Path(__file__),out/'shared-oob.ptx',out/'probe_shared_oob.py',san,*[root/'build-cuda/sanitizer-kernel-overhead-20261115'/f for f in ('shared-oob.cubin','shared-oob.memcheck.cubin')],*[out/(r['label']+s) for r in records for s in ('.stdout','.stderr')]]
result=dict(coverage_equivalent=all(r['detected'] for r in records),records=records,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with (out/'shared-fault-checks.json').open('x') as f:json.dump(result,f,indent=2)
