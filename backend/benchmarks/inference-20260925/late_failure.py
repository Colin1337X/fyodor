"""Force failure after both decode topologies have executed; require CPU replay."""
import json
import os
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_CUDA_FAIL_AFTER='269',NYA_TEST_DECODE='1')
records=[]
for graphs in ('1','0'):
    command=[str(root/'build-cuda/nya-test-inference.exe'),str(root/'build-cuda/backend/test_models/attention-shape-1.gguf'),'resident','273']
    p=subprocess.run(command,env={**env,'NYA_CUDA_GRAPHS':graphs},capture_output=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    name='late-failure-graphs-'+graphs
    (out/(name+'.log')).write_bytes(p.stdout+p.stderr)
    # Every resulting logit must match a full scalar recomputation exactly,
    # rather than retaining any partially computed accelerator prefix.
    values=re.findall(rb'max_scaled=([0-9.e+-]+)',p.stderr)
    assert p.returncode==0 and len(values)==3 and all(float(v)==0 for v in values)
    records.append({'command':command,'graphs':graphs,'environment':{k:v for k,v in env.items() if k.startswith('NYA_')},'exit_code':p.returncode,'all_three_prefixes_exact':True})
(out/'late-failure.json').write_text(json.dumps(records,indent=2))
print('Late failure after token 269: exact CPU replay, graphs enabled and disabled.',flush=True)
