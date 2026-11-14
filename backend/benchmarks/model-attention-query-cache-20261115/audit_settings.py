"""Check effective inherited runner settings without launching a workload."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
root=Path.cwd();out=Path(__file__).resolve().parent
outer=out/'run_monitored.py';inner=out.parent/'pdh-recovery-20261115/run_cost.py'
outer_text=outer.read_text();inner_text=inner.read_text()
a=outer_text.index('    env = {');b=outer_text.index('    runner = ',a)
import textwrap
outer_code=compile(textwrap.dedent(outer_text[a:b]),str(outer),'exec')
a=inner_text.index('settings=json.loads(');b=inner_text.index('exe=root/',a)
inner_code=compile(inner_text[a:b],str(inner),'exec')
proofs=[out/f'generation-short-{s}/result.json' for s in ('c17','c23')]
validated=[]
for path in proofs:
 p=json.loads(path.read_text())
 if p.get('passed') is not True:raise ValueError('Generation did not pass')
 validated.append(p['environment'])
if validated[0]!=validated[1]:raise ValueError('C standards used different settings')
results=[]
for enabled in (False,True):
 # A hostile inherited NYA flag must not survive the outer runner's filter.
 inherited={'PATH':'unchanged','NYA_UNKNOWN_TEST_FLAG':'1','NYA_CUDA_BATCH':'7'}
 scope={'os':SimpleNamespace(environ=inherited),'enabled':enabled}
 exec(outer_code,scope)
 merged={'os':SimpleNamespace(environ=scope['env']),'json':json,'root':root}
 exec(inner_code,merged)
 actual={k:v for k,v in merged['env'].items() if k.startswith('NYA_')}
 controls={'NYA_BENCH_GPU_MONITOR','NYA_BENCH_MODE_SET'}
 expected=dict(validated[0],NYA_CUDA_CACHED_QUERY='1' if enabled else '0')
 execution={k:v for k,v in actual.items() if k not in controls}
 if execution!=expected:raise ValueError(dict(enabled=enabled,actual=execution,expected=expected))
 if actual.get('NYA_BENCH_GPU_MONITOR')!='1' or actual.get('NYA_BENCH_MODE_SET')!='joint-base':
  raise ValueError('Missing observation controls')
 results.append(dict(enabled=enabled,effective_environment=actual,validated_setting_count=len(expected)))
inputs=[Path(__file__).resolve(),outer,inner,*proofs,
 out.parent/'gemma4-gate-20261113/model-oracle-exit.json']
(out/'settings-audit.json').write_text(json.dumps(dict(passed=True,gpu_execution=False,variants=results,
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('Both effective environments match validated generation; only cached-query selection changes')
