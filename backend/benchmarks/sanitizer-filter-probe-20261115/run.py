import hashlib,json,re,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;folder=out/'results'
if folder.exists():raise SystemExit('Refusing to overwrite evidence')
folder.mkdir()
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
cases=[('selected','nya_attention_cached_query_probe','regex=nya_attention_cached_query',True),('excluded','unrelated_probe','regex=nya_attention_cached_query',False),('unfiltered','unrelated_probe',None,True),('no_match','nya_attention_cached_query_probe','regex=never_matches_fixture',False),('substring','unrelated_probe','kns=nya_attention_cached_query',False)]
records=[]
for label,kernel,filter_value,expected_error in cases:
 command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99']
 if filter_value:command+=['--kernel-name',filter_value]
 command += [sys.executable,'-B',str(out/'probe.py'),kernel]
 start=time.monotonic();r=subprocess.run(command,capture_output=True,text=True)
 (folder/(label+'.stdout')).write_text(r.stdout);(folder/(label+'.stderr')).write_text(r.stderr)
 log=r.stdout+r.stderr
 summaries=re.findall(r'^========= ERROR SUMMARY: (\d+) errors?$',log,re.M)
 expected_code=99 if expected_error else 0
 passed=(r.returncode==expected_code and len(summaries)==1 and (int(summaries[0])>0 if expected_error else int(summaries[0])==0) and ('Invalid __global__ write' in log)==expected_error and ('FIXTURE kernel='+kernel) in log)
 records.append(dict(label=label,command=command,exit_code=r.returncode,expected_error=expected_error,passed=passed,elapsed_s=time.monotonic()-start,summaries=summaries))
 print(json.dumps(records[-1]),flush=True)
paths=[Path(__file__).resolve(),out/'probe.py',out/'probe.ptx',sanitizer,*folder.iterdir()]
(folder/'result.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in records),accepted_benchmark=False,records=records,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
raise SystemExit(0 if all(r['passed'] for r in records) else 1)
