"""Strict trace contract for the pending 8K cached-query safety check."""
import math,re

def validate(stdout,stderr):
 def inventory(prefix,pattern,count=1):
  lines=[line for line in stderr.splitlines() if prefix in line]
  matches=[re.fullmatch(pattern,line) for line in lines]
  return matches,len(lines)==count and all(matches)
 counts={};valid=True
 for name,prefix in [('integer','CUDA Q4 integer batch-four calls:'),('qkv','CUDA QKV input reuse capture/direct hits:'),('pair','CUDA paired rows capture/direct calls:'),('q6','CUDA Q6 paired rows capture/direct calls:'),('cached','CUDA cached query capture/direct calls:')]:
  matches,ok=inventory(prefix,re.escape(prefix)+r' (\d+)')
  counts[name]=[int(m[1]) for m in matches if m]
  valid=valid and ok and all(v>0 for v in counts[name])
 matches,ok=inventory('CUDA batched graph captures:',r'CUDA batched graph captures: (\d+) replays: (\d+)')
 graphs=[(int(m[1]),int(m[2])) for m in matches if m]
 graph_valid=ok and all(c>0 and r>c for c,r in graphs)
 matches,ok=inventory('CUDA native metadata Q4:',r'CUDA native metadata Q4: selected')
 metadata_valid=ok
 matches,ok=inventory('SANITIZER_MODULE',r'SANITIZER_MODULE compile_time_memcheck=1 exact_ptx_match=1 bytes=([1-9]\d*)')
 module_valid=ok
 lines=[line for line in (stdout+'\n'+stderr).splitlines() if 'ERROR SUMMARY:' in line]
 memcheck_valid=lines==['========= ERROR SUMMARY: 0 errors']
 lines=[line for line in stdout.splitlines() if 'reset=' in line]
 matches=[re.fullmatch(r'reset=([01]) prefix=8192 tokens=4 max_scaled=([\deE.+-]+) execution=resident',line) for line in lines]
 numerical_valid=len(lines)==2 and all(matches)
 if numerical_valid:
  try:values=[float(m[2]) for m in matches]
  except ValueError:numerical_valid=False
  else:numerical_valid=[m[1] for m in matches]==['0','1'] and all(math.isfinite(v) and 0<=v<=.0005 for v in values)
 return dict(passed=bool(valid and graph_valid and metadata_valid and module_valid and memcheck_valid and numerical_valid),counter_inventory_valid=bool(valid),counts=counts,graph_inventory_valid=bool(graph_valid),graphs=graphs,metadata_valid=bool(metadata_valid),module_valid=bool(module_valid),memcheck_valid=memcheck_valid,numerical_inventory_valid=bool(numerical_valid))
