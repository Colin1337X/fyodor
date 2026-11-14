"""Whole-model joint-base output parity; timings are never accepted here."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
def key(path):return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
if not __debug__: raise SystemExit('Generation validation requires Python assertions')
if len(sys.argv)!=3 or sys.argv[1] not in ('short','long') or sys.argv[2] not in ('c17','c23'):
 raise SystemExit('Choose short|long c17|c23')
context,standard=sys.argv[1:];folder=out/f'generation-{context}-{standard}'
if folder.exists(): raise SystemExit('Refusing to overwrite generation evidence')
proofs=[(out/'build.json','build_passed')]+[(out/f'lifecycle-{standard}-{prefix}/result.json','passed') for standard,prefix in (('c17','257'),('c23','257'),('c17','8192'))]
for path,required in proofs:
 if not path.exists():raise SystemExit('Wait for completed prerequisite: '+str(path))
inputs={path for path,required in proofs}
for path,required in proofs:
 proof=json.loads(path.read_text())
 if proof.get(required) is not True: raise SystemExit(f'Failed prerequisite: {path}')
 if 'exit_code' in proof: assert proof['exit_code']==0 and proof['single_download_assertions_passed'] is True and min(proof['integer_calls'])>0
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
settings['NYA_CUDA_BATCH_GRAPHS']='1'
settings['NYA_CUDA_QKV_INPUT_REUSE']='1'
settings['NYA_CUDA_PAIR_ROWS']='1'
settings['NYA_CUDA_Q6_PAIR_ROWS']='1'
settings['NYA_CUDA_PROFILE']='0';settings.pop('NYA_CUDA_PROFILE_FROM',None)
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
exe=root/f'build-cuda/model-q6-pair-20261114/generation-{standard}.exe'
command=[str(exe),context];folder.mkdir()
inputs.update([Path(__file__),settings_path,exe,
 out.parent/'mtp-joint-base-v2-20261114/model_identity.py',
 out.parent/'gemma4-qkv-20261029/environment.py'])
hashes={key(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
result=dict(passed=False,accepted_benchmark=False,model_identity=identity,command=command,environment=settings,
 limitation='Output correctness only; no continuous contamination acceptance or accepted timing.')
try:
 with (folder/'stdout').open('xb') as stdout,(folder/'stderr').open('xb') as stderr:
  child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
  result['process_id']=child.pid
  (folder/'running.json').write_text(json.dumps(dict(process_id=child.pid,command=command),indent=2))
  try:result['exit_code']=child.wait()
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=15)
 assert result['exit_code']==0
 rows=[json.loads(l) for l in (folder/'stdout').read_text().splitlines() if l.startswith('{"round":')]
 modes=('ordinary','mtp-serial2','mtp-batch2','mtp-joint3')
 order=[(r,m) for r in range(3) for m in (reversed(modes) if r==1 else modes)]
 assert [(r['round'],r['mode']) for r in rows]==order
 assert all(r['parity'] is True and r['generated']==32 and r['prompt']==(25 if context=='short' else 8205) for r in rows)
 trace=(folder/'stderr').read_text()
 assert 'CUDA native metadata Q4: selected' in trace
 assert trace.count('MTP joint base: enabled; proposal window=3')==3
 calls=[int(v) for v in re.findall(r'CUDA Q4 integer batch-four calls: (\d+)',trace)]
 assert len(calls)==3 and min(calls)>0
 hits=[int(v) for v in re.findall(r'CUDA QKV input reuse capture/direct hits: (\d+)',trace)]
 assert len(hits)>=3 and min(hits)>0
 pair_lines=[line for line in trace.splitlines() if 'CUDA paired rows capture/direct calls:' in line]
 pair_matches=[re.fullmatch(r'CUDA paired rows capture/direct calls: (\d+)',line) for line in pair_lines]
 assert len(pair_matches)==3 and all(pair_matches)
 pair_calls=[int(match[1]) for match in pair_matches]
 assert min(pair_calls)>0
 q6_lines=[line for line in trace.splitlines() if 'CUDA Q6 paired rows capture/direct calls:' in line]
 q6_matches=[re.fullmatch(r'CUDA Q6 paired rows capture/direct calls: (\d+)',line) for line in q6_lines]
 assert len(q6_matches)==3 and all(q6_matches)
 q6_calls=[int(match[1]) for match in q6_matches]
 assert min(q6_calls)>0
 for p in inputs: assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[key(p)],p
 graphs=[(int(a),int(b)) for a,b in re.findall(r'CUDA batched graph captures: (\d+) replays: (\d+)',trace)]
 assert len([1 for captures,replays in graphs if captures>0 and replays>captures])==6
 result.update(passed=True,metadata_selected=True,cases=rows,integer_calls=calls,qkv_reuse_hits=hits,pair_rows_calls=pair_calls,q6_pair_rows_calls=q6_calls,graph_counts=graphs)
finally:
 result['sha256']={**hashes,**{key(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'stdout',folder/'stderr')}}
 (folder/'result.json').write_text(json.dumps(result,indent=2))
print(context,standard,'twelve matching cases; no accepted timing')
