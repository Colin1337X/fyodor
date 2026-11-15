"""Read-only progress interpretation. Never performance or safety acceptance."""
import argparse,hashlib,json,math
from pathlib import Path

def summarize(data):
 state=dict(diagnostic_only=True,accepted_benchmark=False,process_liveness='not_checked',overall_percent=None,prefills_started=0,prefills_completed=0,active_prefill=None,last_completed_prefill=None,current_operation=None,last_sampled_token=None,malformed_records=0)
 def number(fields,key,integer=False):
  value=int(fields[key]) if integer else float(fields[key])
  if not math.isfinite(value) or value<0:raise ValueError(key)
  return value
 for raw in data.splitlines(keepends=True):
  if not raw.endswith(('\n','\r')):continue # A writer may still be emitting this record.
  line=raw.strip()
  if line.startswith('INSTRUMENTED_DIAGNOSTIC '):line=line.removeprefix('INSTRUMENTED_DIAGNOSTIC ')
  if not line.startswith(('PROGRESS ','TOKEN ','LIFECYCLE ')):continue
  try:
   kind,rest=line.split(' ',1);pairs=[item.split('=',1) for item in rest.split()];fields=dict(pairs)
   if len(fields)!=len(pairs):raise ValueError('duplicate field')
   if kind=='LIFECYCLE':
    if fields['event']=='begin':state['current_operation']=dict(operation=fields['operation'],source_line=number(fields,'source_line',True))
    elif fields['event']=='end':state['current_operation']=None
    else:raise ValueError('operation event')
   elif kind=='TOKEN':
    state['last_sampled_token']=dict(position=number(fields,'completed_position',True),session_capacity=number(fields,'session_capacity',True),session_capacity_is_not_work_total=True,instrumented_token_seconds=number(fields,'token_elapsed_s'),instrumented_instantaneous_tokens_per_s=number(fields,'instantaneous_tokens_per_s'))
   else:
    event=fields['event'];done=number(fields,'completed',True);total=number(fields,'total',True);position=number(fields,'position',True);elapsed=number(fields,'elapsed_s')
    if done>total:raise ValueError('completed exceeds total')
    if event=='prefill_begin':
     if state['active_prefill'] is not None:raise ValueError('nested prefill')
     state['prefills_started']+=1
     state['active_prefill']=dict(index=state['prefills_started'],completed_tokens=done,total_tokens=total,percent=100*done/total if total else None,completed_batch_rate_tokens_per_s=None,batch=None)
    elif event in ('batch_begin','batch_complete','prefill_complete'):
     phase=state['active_prefill']
     if phase is None or total!=phase['total_tokens'] or done<phase['completed_tokens']:raise ValueError('phase progress mismatch')
     if event=='prefill_complete' and done!=total:raise ValueError('incomplete prefill')
     rate=number(fields,'tokens_per_s')
     phase.update(completed_tokens=done,percent=100*done/total if total else None)
     if event=='batch_begin':phase['batch']=dict(position=position,token_count=None,layers_completed=0,current_layer=None,layer_count=None,instrumented_elapsed_s=0)
     else:
      phase['completed_batch_rate_tokens_per_s']=rate;phase['batch']=None
      if event=='prefill_complete':
       state['prefills_completed']+=1;state['last_completed_prefill']=phase;state['active_prefill']=None
    elif event in ('layer_begin','layer_complete'):
     phase=state['active_prefill'];layer=number(fields,'layer',True);layers=number(fields,'layers',True)
     if phase is None or phase['batch'] is None or layer<1 or layer>layers or position!=phase['batch']['position']:raise ValueError('layer outside batch')
     batch=phase['batch']
     if batch['token_count'] not in (None,total) or layer<batch['layers_completed']:raise ValueError('batch mismatch')
     batch.update(token_count=total,current_layer=layer,layer_count=layers,instrumented_elapsed_s=elapsed)
     if event=='layer_complete':batch['layers_completed']=layer
    else:raise ValueError('unknown progress event')
  except (ValueError,KeyError,TypeError,OverflowError):state['malformed_records']+=1
 return state

def main():
 parser=argparse.ArgumentParser();parser.add_argument('log',type=Path);args=parser.parse_args()
 data=args.log.read_bytes();result=summarize(data.decode('utf-8',errors='replace'))
 result['snapshot']=dict(path=str(args.log.resolve()),bytes_read=len(data),sha256=hashlib.sha256(data).hexdigest())
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
