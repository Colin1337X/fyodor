"""Summarize a complete pair, then validate boundary cases without timing overlap."""
import hashlib,json,os,re,statistics,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
results={}
for label in ('long-off','long-on'):
 folder=out/label;metadata=json.loads((folder/'timing.json').read_text())
 assert len(metadata['runs'])==1 and metadata['runs'][0]['context']=='long' and metadata['runs'][0]['exit_code']==0
 stdout=(folder/'long.stdout').read_text()
 rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{"round":')]
 assert len(rows)==12 and all(row['parity'] and row['generated']==32 for row in rows)
 assert {(row['round'],row['mode']) for row in rows}=={(r,m) for r in range(3) for m in ('ordinary','mtp-serial2','mtp-batch2','mtp-batch4')}
 cells={};case=None
 for line in (folder/'long.stderr').read_text().splitlines():
  match=re.fullmatch(r'CASE round=(\d+) mode=(\S+)',line)
  if match:case=(int(match[1]),match[2])
  match=re.search(r' decode_loop_ms=([\d.]+)',line)
  if match and case[0]>0:cells.setdefault(case[1],[]).append(float(match[1]))
 assert len(cells)==4 and all(len(ms)==2 for ms in cells.values())
 results[label]={'cells':{mode:{'milliseconds':ms,'mean_ms':statistics.mean(ms),'stdev_ms':statistics.stdev(ms)} for mode,ms in cells.items()},'rows':rows,'reference_text':stdout.split('TEXT ',1)[1].split('\n{"round":',1)[0]}
assert results['long-off']['reference_text']==results['long-on']['reference_text']
ratios={mode:100*results['long-off']['cells'][mode]['mean_ms']/results['long-on']['cells'][mode]['mean_ms'] for mode in results['long-off']['cells']}
(out/'long-summary.json').write_text(json.dumps({'accepted_llama_score':False,'contract':'Same binary, QPC, 8205 prompt tokens, 32 greedy output tokens; round zero excluded; two warmed observations per mode.','results':results,'candidate_speed_percent_of_baseline':ratios},indent=2)+'\n')
print(json.dumps(ratios,indent=2),flush=True)
if (out/'boundaries.json').exists():raise SystemExit('Refusing to overwrite boundary evidence')
compiler=root/'.tools/w64devkit/bin/gcc.exe';env=os.environ.copy()
env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
env.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='8',NYA_CUDA_RING_KV='1',NYA_CUDA_SMALL_BATCH='1',NYA_SPECULATIVE_BATCH='1')
report={'test_source_sha256':hashlib.sha256((root/'backend/tests/test_generation.c').read_bytes()).hexdigest(),'builds':[],'runs':[]}
for standard,build in [('c17','build-cuda'),('c23','build-cuda-c23')]:
 exe=root/build/'mtp-early-boundaries.exe'
 command=[str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Werror','-I','backend/include','-I','backend/core','backend/tests/test_generation.c',str(root/build/'mtp-early-reject-core.o'),str(root/build/'libnya-engine.a'),'-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',str(exe)]
 subprocess.run(command,env=env,check=True)
 report['builds'].append({'command':command,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()})
 for provider in ('cpu','cuda'):
  env['NYA_COMPUTE']=provider
  if provider=='cuda':env['NYA_TEST_SPECULATIVE_BATCH']='1'
  else:env.pop('NYA_TEST_SPECULATIVE_BATCH',None)
  for enabled in ('0','1'):
   env['NYA_SPECULATIVE_EARLY_REJECT']=enabled;label=f'boundaries-{standard}-{provider}-{enabled}'
   command=[str(exe),str(root/build/f'test_models/{label}.gguf')]
   with (out/(label+'.log')).open('w') as log:result=subprocess.run(command,env=env,stdout=log,stderr=log)
   report['runs'].append({'command':command,'standard':standard,'provider':provider,'enabled':enabled,'exit_code':result.returncode})
   (out/'boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
   if result.returncode:raise SystemExit(result.returncode)
   assert (out/(label+'.log')).read_text().count('speculative boundary cases=48 plus output-limit failure passed')==2
   print(label,'passed',flush=True)
