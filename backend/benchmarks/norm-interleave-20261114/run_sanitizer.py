"""Repeat the complete RMSNorm oracle under a selected CUDA sanitizer."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or sys.argv[1] not in ('memcheck','racecheck','synccheck'):raise SystemExit('Choose sanitizer')
tool=sys.argv[1];raw_path=out/'oracle.json';raw=json.loads(raw_path.read_text())
if raw.get('passed') is not True or len(raw['cases'])!=1248 or len(raw['rejections'])!=3:raise ValueError('Incomplete raw oracle')
inputs=dict(raw['input_sha256'])
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
for p in (raw_path,Path(__file__).resolve(),sanitizer):inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in inputs.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
stdout_path=out/(tool+'.log');stderr_path=out/(tool+'.stderr.log');oracle_path=out/(tool+'-oracle.json');result_path=out/(tool+'-run.json')
if any(p.exists() for p in (stdout_path,stderr_path,oracle_path,result_path)):raise SystemExit('Refusing evidence overwrite')
command=[str(sanitizer),'--tool',tool,'--error-exitcode','99',sys.executable,str(out/'check_kernel.py'),tool]
result=dict(passed=False,accepted_benchmark=False,command=command,input_sha256=inputs)
try:
    with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        result['process_id']=child.pid
        (out/(tool+'-running.json')).write_text(json.dumps(result,indent=2))
        try:result['exit_code']=child.wait()
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=15)
    if result['exit_code']!=0:raise ValueError('Sanitizer failed')
    text=stdout_path.read_text()+'\n'+stderr_path.read_text()
    pattern=r'^========= RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)\s*$' if tool=='racecheck' else r'^========= ERROR SUMMARY: (\d+) errors\s*$'
    if re.findall(pattern,text,re.M)!=([('0','0','0')] if tool=='racecheck' else ['0']):raise ValueError('Missing or nonzero sanitizer summary')
    oracle=json.loads(oracle_path.read_text())
    if oracle.get('passed') is not True or oracle['cases']!=raw['cases'] or oracle['rejections']!=raw['rejections'] or oracle['input_sha256']!=raw['input_sha256']:raise ValueError('Sanitized oracle mismatch')
    for name,h in inputs.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed: '+name)
    result['passed']=True
finally:
    result['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (stdout_path,stderr_path,oracle_path) if p.exists()}
    result_path.write_text(json.dumps(result,indent=2))
print(tool,'1248 parity and three rejection cases passed')
