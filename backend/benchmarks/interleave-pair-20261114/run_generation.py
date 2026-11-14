"""Whole-model joint-base output parity; timings are never accepted here."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=3 or sys.argv[1] not in ('short','long') or sys.argv[2] not in ('c17','c23'):
 raise SystemExit('Choose short|long c17|c23')
context,standard=sys.argv[1:];folder=out/f'generation-{context}-{standard}'
if folder.exists(): raise SystemExit('Refusing to overwrite generation evidence')
proofs=[out/'executables.json',out/'build.json',out/'lifecycle-c17-257/result.json',out/'lifecycle-c23-257/result.json']
inputs=set(proofs)
for path in proofs:
 proof=json.loads(path.read_text())
 if 'passed' in proof: assert proof['passed']
 if 'link_passed' in proof: assert proof['link_passed']
 if 'compile_passed' in proof: assert proof['compile_passed']
 if 'exit_code' in proof: assert proof['exit_code']==0 and min(proof['integer_calls'])>0
 for n,h in proof['sha256'].items():
  assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
  inputs.add(root/n)
sys.path.insert(0,str(out.parent/'mtp-joint-base-v2-20261114'))
from model_identity import verify_gemma
identity=verify_gemma()
sys.path.insert(0,str(out.parent/'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
 raise SystemExit('Another GPU workload remains')
settings_path=out.parent/'joint-decode-profile-20261114/short-integer/result.json'
settings=json.loads(settings_path.read_text())['environment']
settings['NYA_CUDA_PROFILE']='0';settings.pop('NYA_CUDA_PROFILE_FROM',None)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
exe=root/f'build-cuda/interleave-pair-20261114/generation-{standard}.exe'
command=[str(exe),context];folder.mkdir()
inputs.update([Path(__file__),settings_path,exe])
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
result=dict(passed=False,accepted_benchmark=False,model_identity=identity,command=command,environment=settings,
 limitation='Output correctness only; no continuous contamination acceptance or accepted timing.')
try:
 with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
  result['exit_code']=subprocess.run(command,env=env,stdout=stdout,stderr=stderr).returncode
 assert result['exit_code']==0
 rows=[json.loads(l) for l in (folder/'stdout').read_text().splitlines() if l.startswith('{"round":')]
 modes=('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')
 order=[(r,m) for r in range(3) for m in (reversed(modes) if r==1 else modes)]
 assert [(r['round'],r['mode']) for r in rows]==order
 assert all(r['parity'] is True and r['generated']==32 and r['prompt']==(25 if context=='short' else 8205) for r in rows)
 trace=(folder/'stderr').read_text()
 assert trace.count('MTP joint base: enabled; proposal window=3')==3
 calls=[int(v) for v in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)',trace)]
 assert len(calls)==3 and min(calls)>0
 for p in inputs: assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(root))],p
 result.update(passed=True,cases=rows,integer_calls=calls)
finally:
 result['sha256']={**hashes,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'stdout',folder/'stderr')}}
 (folder/'result.json').write_text(json.dumps(result,indent=2))
print(context,standard,'twelve matching cases; no accepted timing')
