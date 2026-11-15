"""Progress-visible diagnostics; never a throughput acceptance runner."""
import argparse,hashlib,json,os,queue,re,subprocess,threading,time
from pathlib import Path

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('standard',choices=['c17','c23']);parser.add_argument('prefix',type=int)
 parser.add_argument('label');parser.add_argument('--memcheck',action='store_true')
 args=parser.parse_args()
 if not 5<=args.prefix<=8192 or not args.label.isascii() or not args.label.isalnum():parser.error('prefix 5..8192 and alphanumeric label required')
 root=Path.cwd();out=Path(__file__).resolve().parent;folder=out/args.label
 if folder.exists():raise SystemExit('Refusing to overwrite evidence')
 proof=json.loads((out/'build.json').read_text())
 for name,h in proof['sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
 source=root/'backend/benchmarks/model-attention-query-cache-20261115/lifecycle-c17-8192/result.json'
 settings=json.loads(source.read_text())['environment']
 env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
 command=[str(root/f'build-cuda/lifecycle-progress-20261115/lifecycle-{args.standard}.exe'),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf',str(args.prefix)]
 if args.memcheck:
  command=[str(root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'),'--tool','memcheck','--kernel-name','regex=nya_attention_cached_query','--error-exitcode','99']+command
 folder.mkdir();started=time.monotonic();events=queue.Queue();last=started;latest='starting';closed=0
 def reader(pipe,name):
  for line in pipe:events.put((name,line))
  pipe.close();events.put((name,None))
 child=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
 (folder/'running.json').write_text(json.dumps(dict(pid=child.pid,command=command,environment=settings),indent=2))
 for name in ('stdout','stderr'):threading.Thread(target=reader,args=(getattr(child,name),name),daemon=True).start()
 streams={name:(folder/name).open('x') for name in ('stdout','stderr')}
 next_heartbeat=started+10
 try:
  while closed<2:
   try:name,line=events.get(timeout=max(.01,next_heartbeat-time.monotonic()))
   except queue.Empty:line=None;name=None
   now=time.monotonic()
   if name is not None:
    if line is None:closed+=1
    else:
     streams[name].write(line);streams[name].flush();print(line,end='',flush=True)
     if line.startswith(('PROGRESS ','LIFECYCLE ')):last=now;latest=line.strip()
   if now>=next_heartbeat:
    print(f'HEARTBEAT elapsed_s={now-started:.1f} since_last_progress_s={now-last:.1f} pid={child.pid} last={latest}',flush=True)
    next_heartbeat=now+10
  code=child.wait()
 finally:
  for stream in streams.values():stream.close()
  if child.poll() is None:
   child.terminate();child.wait()
 stdout=(folder/'stdout').read_text();stderr=(folder/'stderr').read_text()
 rows=re.findall(r'^reset=(\d+) prefix=(\d+) tokens=4 max_scaled=([\deE.+-]+) execution=resident$',stdout,re.M)
 numerical=len(rows)==2 and [r[0] for r in rows]==['0','1'] and all(int(p)==args.prefix and 0<=float(e)<=.0005 for _,p,e in rows)
 progress=[dict(item.split('=',1) for item in line.split()[1:]) for line in stderr.splitlines() if line.startswith('PROGRESS ')]
 completed=[r for r in progress if r['event']=='prefill_complete']
 logs_valid=len(completed)==4 and all(r['completed']==r['total'] and float(r['percent'])==100 and float(r['tokens_per_s'])>0 for r in completed)
 logs_valid=logs_valid and any(r['event']=='layer_begin' for r in progress) and any(r['event']=='layer_complete' for r in progress)
 summary=re.findall(r'^========= ERROR SUMMARY:.*$',stdout+stderr,re.M)
 sanitizer_ok=not args.memcheck or summary==['========= ERROR SUMMARY: 0 errors']
 changed=[n for n,h in proof['sha256'].items() if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h]
 result=dict(passed=code==0 and numerical and logs_valid and sanitizer_ok and not changed,exit_code=code,numerical_checks_passed=numerical,progress_checks_passed=logs_valid,memcheck=args.memcheck,accepted_benchmark=False,elapsed_s=time.monotonic()-started,progress_records=len(progress),prefill_summaries=completed,changed_inputs=changed,command=command,environment=settings)
 result['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),out/'build.json',source,*folder.iterdir()] if p.is_file()}
 (folder/'result.json').write_text(json.dumps(result,indent=2))
 print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('sha256','environment','command')}),flush=True)
 return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
