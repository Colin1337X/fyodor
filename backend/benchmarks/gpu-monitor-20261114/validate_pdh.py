"""Validate attribution against a known correctness-test PID; never accept timing."""
import datetime,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
identity_source=out.parent/'gpu-activity-20261114/cuda_identity.py'
spec=importlib.util.spec_from_file_location('identity',identity_source)
identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
if (out/'pdh-live.jsonl').exists() or (out/'pdh-validation.json').exists():raise SystemExit('Refusing to overwrite live evidence')
devices=identity.identities()
(out/'pdh-identity.json').write_text(json.dumps(devices,indent=2)+'\n')
adapter=next(d for d in devices['devices'] if d['ordinal']==0)
stop=out/'pdh-live.stop';trace=out/'pdh-live.jsonl'
worker_command=[sys.executable,str(out/'capture_pdh.py'),'--output',str(trace),'--stop',str(stop),'--luid',adapter['counter_luid'],'--samples','60']
worker=subprocess.Popen(worker_command,creationflags=subprocess.CREATE_NO_WINDOW)
work=None;meta={'accepted_benchmark':False,'adapter':adapter,'monitor_command':worker_command}
try:
 deadline=time.monotonic()+20
 while True:
  if worker.poll() is not None:raise RuntimeError('Monitor exited before test')
  if trace.exists() and len(trace.read_text().splitlines())>=3:break
  if time.monotonic()>deadline:raise RuntimeError('Monitor readiness timeout')
  time.sleep(.1)
 command=[sys.executable,str(out/'workload.py')]
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 with (out/'pdh-workload.log').open('x') as log:
  meta['workload_start_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  meta['workload_start_qpc_seconds']=time.perf_counter()
  work=subprocess.Popen(command,env=env,stdout=log,stderr=log)
  meta.update(workload_command=command,workload_pid=work.pid)
  meta['workload_exit_code']=work.wait(timeout=120)
  meta['workload_end_qpc_seconds']=time.perf_counter()
  meta['workload_end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 # Preserve a complete sampling interval after the test exits.
 time.sleep(2)
finally:
 if work is not None and work.poll() is None:
  work.terminate();work.wait(timeout=15);meta['workload_terminated']=True
 stop.touch(exist_ok=False)
 try:meta['monitor_exit_code']=worker.wait(timeout=15)
 except subprocess.TimeoutExpired:
  worker.terminate();worker.wait(timeout=15);meta['monitor_terminated']=True
 (out/'pdh-validation.json').write_text(json.dumps(meta,indent=2)+'\n')
rows=[json.loads(line) for line in trace.read_text().splitlines()]
assert meta.get('workload_exit_code')==0 and meta.get('monitor_exit_code')==0
assert rows[-1]['kind']=='end' and rows[-1]['requested_stop'] and not rows[-1]['failed']
start=datetime.datetime.fromisoformat(meta['workload_start_utc']);end=datetime.datetime.fromisoformat(meta['workload_end_utc'])
activity=[counter for row in rows if row['kind']=='sample' and start<=datetime.datetime.fromisoformat(row['timestamp'])<=end for counter in row['target_counters'] if counter['process_id']==meta['workload_pid'] and counter['status'] in (0,1) and counter['engine_percent']>0]
assert activity,'Known CUDA test PID was not observed on target adapter'
meta['observed_workload_positive_counters']=len(activity);meta['passed']=True
(out/'pdh-validation.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Live CUDA PID attribution passed:',len(activity),'positive counter observations',flush=True)
