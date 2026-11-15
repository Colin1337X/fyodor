"""Compare offline SASS for paired-row unroll candidates without using CUDA."""
from collections import Counter
import hashlib,json,os,re,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
proof_path=out/'assembly.json';proof=json.loads(proof_path.read_text());assert proof['assembled']
inputs={root/name for name in proof['sha256']}
for name,h in proof['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
dump=root/'.tools/cuda-sdk/cuda_cuobjdump-windows-x86_64-13.1.115-archive/bin/cuobjdump.exe'
disasm=root/'.tools/cuda-sdk/cuda_nvdisasm-windows-x86_64-13.1.115-archive/bin/nvdisasm.exe'
cubin=root/'build-cuda/q4-pair-unroll-20261115/module.cubin';records=[]
for name in ('nya_q4_pair_rows','nya_q4_pair_unroll2','nya_q4_pair_unroll3'):
 command=[str(dump),'--function',name,'--dump-sass',str(cubin)]
 r=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PATH=str(disasm.parent)+os.pathsep+os.environ['PATH']))
 stdout=out/(name+'.sass');stderr=out/(name+'.stderr')
 for p,s in ((stdout,r.stdout),(stderr,r.stderr)):
  with p.open('x') as f:f.write(s)
 assert r.returncode==0
 match=re.search(r'Function : '+name+r'\s+(.*?)(?=Function :|\Z)',r.stdout,re.S);assert match
 counts=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]*)',match[1]));assert counts
 records.append(dict(kernel=name,command=command,static_instruction_counts=dict(sorted(counts.items())),total_static_instructions=sum(counts.values())))
 inputs.update((stdout,stderr));print(name,'static_instructions',sum(counts.values()),'FFMA',counts['FFMA'],'local_loads',sum(v for k,v in counts.items() if k.startswith('LDL')))
inputs.update((Path(__file__),proof_path,dump,disasm))
with (out/'sass-audit.json').open('x') as f:json.dump(dict(passed=True,gpu_execution=False,accepted_benchmark=False,records=records,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),f,indent=2)
