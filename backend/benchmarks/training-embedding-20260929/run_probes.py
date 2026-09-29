"""Run component correctness probes; timings are not accepted performance evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,help='Use a fresh evidence directory; existing logs are never overwritten')
args=parser.parse_args()
if args.output:
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_CUDA_BLAS='0',NYA_CUDA_CUTLASS='0',NYA_COMPUTE='cpu')
records=[]
for mode,reference in (('embedffn8',False),('embedffn64',False),('embedffn8',True),('rmsffn8',False),('ffn8',False)):
    label='final-'+mode+('-reference' if reference else '')
    command=[str(root/'build-cuda/nya-bench-training-device.exe'),str(root/'.tools/tinyllama-q4_k_m.gguf'),mode]
    runenv=dict(env)
    if reference: runenv['NYA_CUDA_REFERENCE']='1'
    with (out/(label+'.out')).open('xb') as stdout, (out/(label+'.log')).open('xb') as stderr:
        result=subprocess.run(command,cwd=root,env=runenv,stdout=stdout,stderr=stderr)
    records.append({'label':label,'command':command,'reference':reference,'exit_code':result.returncode})
    (out/'probes.json').write_text(json.dumps(records,indent=2)+'\n')
    print(label,result.returncode,flush=True)
    if result.returncode: raise SystemExit(result.returncode)
