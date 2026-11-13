"""Retain dynamically enumerated PDH engine instances in flushed JSONL."""
import argparse,datetime,json,math,re,time
from pathlib import Path
from pdh_counter import Counter
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop',type=Path,required=True);parser.add_argument('--luid',required=True);parser.add_argument('--samples',type=int,default=900)
a=parser.parse_args();luid=a.luid.lower()
if not re.fullmatch(r'luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}',luid) or not 2<=a.samples<=3600 or a.stop.exists():raise SystemExit('Invalid capture parameters')
with a.output.open('x',encoding='utf-8') as output:
 def emit(row):output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
 query=None;failed=False;count=0;started=time.perf_counter()
 try:
  query=Counter()
  emit({'kind':'header','schema':1,'adapter_luid':luid,'provider':'PDH wildcard array','accepted_benchmark':False})
  for count in range(1,a.samples+1):
   time.sleep(1)
   before=time.perf_counter();rows=query.collect();after=time.perf_counter()
   target=[];other=[]
   for row in rows:
    match=re.match(r'^pid_(\d+)_',row['instance']);row['process_id']=int(match[1]) if match else None
    if row['engine_percent'] is not None and not math.isfinite(row['engine_percent']):
     row['nonfinite_value']=True;row['engine_percent']=None
    if luid in row['instance']:target.append(row)
    elif row['engine_percent'] is not None and row['engine_percent']>0:other.append(row)
   emit({'kind':'sample','index':count,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'qpc_before_collect':before,'qpc_observed_seconds':after,'collection_seconds':after-before,
         'observer_cpu_seconds':time.process_time(),'all_counter_count':len(rows),
         'target_counters':target,'other_positive_engines':other})
   if a.stop.exists():break
 except BaseException as error:
  failed=True;emit({'kind':'error','message':str(error)})
 finally:
  if query is not None:
   try:query.close()
   except BaseException as error:failed=True;emit({'kind':'error','message':str(error)})
  emit({'kind':'end','samples':count,'requested_stop':a.stop.exists(),'failed':failed,'elapsed_seconds':time.perf_counter()-started})
 if failed:raise SystemExit(1)
