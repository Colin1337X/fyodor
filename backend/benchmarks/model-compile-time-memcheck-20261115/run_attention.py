"""Check all cached-query shapes using the exact instrumented model module."""
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=2 or not sys.argv[1].isascii() or not sys.argv[1].isalnum():raise SystemExit('Unique alphanumeric label required')
label=sys.argv[1];folder=out/label
if folder.exists() or (out/(label+'-oracle.json')).exists():raise SystemExit('Refusing overwrite')
proof_path=out/'v2-build.json';proof=json.loads(proof_path.read_text());assert proof['build_passed']
inputs=dict(proof['sha256'])
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
for p in (Path(__file__),proof_path,out/'attention_common.py',out/'check_attention.py',sanitizer):inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
def changed():return [name for name,h in inputs.items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
assert not changed()
env=os.environ.copy();env.pop('NV_COMPUTE_SANITIZER_BINARY_PATCHING',None)
command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99',sys.executable,'-B',str(out/'check_attention.py'),label+'-oracle.json']
folder.mkdir();start=time.monotonic()
with (folder/'stdout').open('x') as stdout,(folder/'stderr').open('x') as stderr:
 child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
 (folder/'running.json').write_text(json.dumps(dict(pid=child.pid,command=command),indent=2))
 try:
  while child.poll() is None:
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:
    lines=(folder/'stdout').read_text().splitlines();completed=sum(line.endswith(' passed') for line in lines)
    print(f'HEARTBEAT elapsed_s={time.monotonic()-start:.1f} cases={completed}/24 pid={child.pid}',flush=True)
 finally:
  if child.poll() is None:child.terminate();child.wait()
logs=(folder/'stdout').read_text()+(folder/'stderr').read_text();summary=re.findall(r'^========= ERROR SUMMARY:.*$',logs,re.M)
oracle_path=out/(label+'-oracle.json');oracle=json.loads(oracle_path.read_text()) if oracle_path.exists() else {};cases=oracle.get('cases',[])
expected=[(w,p,b,r) for w in (256,512) for p in (256,8192) for b in (1,2,4) for r in (False,True)]
inventory=[(c['width'],c['position'],c['batch'],c['ring']) for c in cases]==expected
numerics=inventory and all(c['serial_split_exact'] and 0<=c['unsplit_scaled_error']<=.0005 and 0<=c['float64_scaled_error']<=.0005 for c in cases)
changes=changed();passed=child.returncode==0 and summary==['========= ERROR SUMMARY: 0 errors'] and numerics and not changes
result=dict(passed=passed,accepted_benchmark=False,model_lifecycle_gate_satisfied=False,compile_time_instrumentation=True,filtered=False,exit_code=child.returncode,elapsed_s=time.monotonic()-start,case_inventory_valid=inventory,numerical_checks_passed=numerics,changed_inputs=changes,command=command)
paths=[*folder.iterdir()]+([oracle_path] if oracle_path.exists() else [])
result['sha256']={**inputs,**{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}}
(folder/'result.json').write_text(json.dumps(result,indent=2));print('RESULT',passed,'cases',len(cases),flush=True)
raise SystemExit(0 if passed else 1)
