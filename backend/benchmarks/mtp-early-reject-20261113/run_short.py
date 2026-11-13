"""Screen the already-validated prototype at short context before longer timing."""
import json,os,re,statistics,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
tests=json.loads((out/'validation.json').read_text())
assert len(tests)==8 and all(row['exit_code']==0 for row in tests)
env=os.environ.copy()
env.update(NYA_CUDA_SMALL_BATCH_REUSE='1',NYA_CUDA_WIDE_BATCH='1',NYA_CUDA_MTP_WIDE_VALUE='0')
results={}
for label,enabled in [('short-off','0'),('short-on','1')]:
 env['NYA_SPECULATIVE_EARLY_REJECT']=enabled
 subprocess.run([sys.executable,'backend/benchmarks/mtp-cost-20261113/run.py','../mtp-early-reject-20261113/'+label,'build-cuda/mtp-early-reject.exe','short'],env=env,check=True)
 metadata=json.loads((out/label/'timing.json').read_text())
 assert len(metadata['runs'])==1 and metadata['runs'][0]['exit_code']==0
 stdout=(out/label/'short.stdout').read_text()
 rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{"round":')]
 assert len(rows)==12 and all(row['parity'] and row['generated']==32 for row in rows)
 observations={};case=None
 for line in (out/label/'short.stderr').read_text().splitlines():
  match=re.fullmatch(r'CASE round=(\d+) mode=(\S+)',line)
  if match:case=(int(match[1]),match[2])
  match=re.search(r' decode_loop_ms=([\d.]+)',line)
  if match and case[0]>0:observations.setdefault(case[1],[]).append(float(match[1]))
 assert len(observations)==4 and all(len(ms)==2 for ms in observations.values())
 results[label]={'cells':{mode:{'milliseconds':ms,'mean_ms':statistics.mean(ms),'stdev_ms':statistics.stdev(ms)} for mode,ms in observations.items()},'rows':rows,'reference_text':stdout.split('TEXT ',1)[1].split('\n{"round":',1)[0]}
assert results['short-off']['reference_text']==results['short-on']['reference_text']
ratios={mode:100*results['short-off']['cells'][mode]['mean_ms']/results['short-on']['cells'][mode]['mean_ms'] for mode in results['short-off']['cells']}
(out/'short-summary.json').write_text(json.dumps({'accepted_llama_score':False,'contract':'Same binary, QPC decode-loop wall time, 25 prompt tokens, 32 greedy output tokens including first prefill-logit sample; first round excluded; two warmed observations per mode.','results':results,'candidate_speed_percent_of_baseline':ratios},indent=2)+'\n')
print(json.dumps(ratios,indent=2))
