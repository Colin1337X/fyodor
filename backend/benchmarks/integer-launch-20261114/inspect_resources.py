"""Record driver-reported resource usage; this does not measure occupancy."""
import json
from pathlib import Path
from common import *
output=Path(__file__).resolve().parent/'resources.json'
if output.exists(): raise SystemExit('Refusing to overwrite resource evidence')
get_attribute=bind(cu,'cuFuncGetAttribute',[C.POINTER(I),I,P])
results={}
try:
 for name in ['nya_interleaved_integer_12_4']+[f'nya_integer_launch_{r}_{b}' for r,b in ((4,4),(4,8),(8,3),(8,4))]:
  values={}
  for label,attribute_id in [('max_threads',0),('static_shared_bytes',1),('local_bytes',3),('registers_per_thread',4)]:
   value=I();check(get_attribute(C.byref(value),attribute_id,funcs[name]));values[label]=value.value
  results[name]=values
finally:
 check(unload(module));check(release(device))
result=dict(accepted_benchmark=False,source_sha256=hashlib.sha256(source).hexdigest(),
 harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),functions=results)
output.write_text(json.dumps(result,indent=2))
print(json.dumps(results,indent=2))
