"""Run explicitly reduced-coverage diagnostics, never safety acceptance."""
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
if len(sys.argv)!=3 or sys.argv[1] not in ('filtered','unfiltered') or not sys.argv[2].isascii() or not sys.argv[2].isalnum():raise SystemExit('mode label')
mode,label=sys.argv[1:];dest=out/(label+'-command.json')
if dest.exists() or (out/(label+'.json')).exists():raise SystemExit('Refusing overwrite')
env=os.environ.copy();env.pop('NV_COMPUTE_SANITIZER_BINARY_PATCHING',None)
# Keep default binary instrumentation; compiler instrumentation is in the CUBIN.
sanitizer=root/'.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99']
if mode=='filtered':command+=['--kernel-name','regex=fyodor_deliberately_no_matching_kernel_20261115']
command+=[sys.executable,'-B',str(out/'probe_instrumented.py'),label]
start=time.monotonic()
with (out/(label+'.stdout')).open('x') as stdout,(out/(label+'.stderr')).open('x') as stderr:r=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
text=(out/(label+'.stdout')).read_text()+(out/(label+'.stderr')).read_text()
summaries=re.findall(r'^========= ERROR SUMMARY:.*$',text,re.M)
result=dict(passed=r.returncode==0 and summaries==['========= ERROR SUMMARY: 0 errors'] and json.loads((out/(label+'.json')).read_text())['passed'],exit_code=r.returncode,accepted_benchmark=False,safety_evidence=False,mode=mode,command=command,environment_override={'NV_COMPUTE_SANITIZER_BINARY_PATCHING':env.get('NV_COMPUTE_SANITIZER_BINARY_PATCHING')},elapsed_s=time.monotonic()-start)
result['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),sanitizer,out/'probe_instrumented.py',out/(label+'.json'),out/(label+'.stdout'),out/(label+'.stderr'))}
dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
