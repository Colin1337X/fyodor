import hashlib,importlib.util,json,os,subprocess,time,sys
from pathlib import Path
root=Path.cwd(); out=Path(__file__).resolve().parent
contexts=sys.argv[3:] or ['short','long']
if len(set(contexts))!=len(contexts) or any(context not in ('short','long') for context in contexts):
    raise SystemExit('Contexts must be distinct short/long names')
if len(sys.argv)>1:out=out/sys.argv[1]
out.mkdir(exist_ok=True)
if (out/'timing.json').exists():raise SystemExit('Refusing to overwrite existing timing evidence')
spec=importlib.util.spec_from_file_location('gate',root/'backend/benchmarks/gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
gate.configure(out)
try:gate.idle();accepted=True
except RuntimeError:accepted=False
settings=json.loads((root/'backend/benchmarks/gemma4-gate-20261113/model-oracle-exit.json').read_text())['environment']
settings.update(NYA_CUDA_BATCH='512',NYA_CUDA_RING_KV='1',NYA_CUDA_SMALL_BATCH='1',NYA_CUDA_WIDE_VALUE='1',NYA_CUDA_GEMMA_GATE='1',NYA_CUDA_GEMMA_GATE_PARTIAL='1',NYA_CUDA_BLAS='1',NYA_MTP_RESIDENT_TARGET='1',NYA_MTP_TRACE='1',NYA_GENERATION_PROFILE='1')
settings.update(NYA_CUDA_SMALL_BATCH_REUSE=os.environ.get('NYA_CUDA_SMALL_BATCH_REUSE','0'),NYA_CUDA_MTP_WIDE_VALUE=os.environ.get('NYA_CUDA_MTP_WIDE_VALUE','0'))
settings['NYA_CUDA_WIDE_BATCH']=os.environ.get('NYA_CUDA_WIDE_BATCH','0')
settings['NYA_SPECULATIVE_EARLY_REJECT']=os.environ.get('NYA_SPECULATIVE_EARLY_REJECT','0')
settings['NYA_CUDA_BATCH_LOGITS']=os.environ.get('NYA_CUDA_BATCH_LOGITS','0')
settings['NYA_CUDA_ASSISTED_FFN']=os.environ.get('NYA_CUDA_ASSISTED_FFN','0')
env=os.environ.copy();env.update(settings)
exe=root/(sys.argv[2] if len(sys.argv)>2 else 'build-cuda/mtp-cost.exe')
meta={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'environment':settings,'gate_passed':accepted,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'backend/core/llm_cpu.c',Path(__file__),Path(__file__).with_name('model.c')]},'runs':[]}
for context in contexts:
    command=[str(exe),context]
    row={'context':context,'command':command,'samples':[]};meta['runs'].append(row)
    with (out/(context+'.stdout')).open('w') as stdout,(out/(context+'.stderr')).open('w') as stderr:
        process=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
        while process.poll() is None:
            row['samples'].append(gate.snapshot())
            (out/'timing.json').write_text(json.dumps(meta,indent=2)+'\n')
            time.sleep(1)
        row['exit_code']=process.returncode
    (out/'timing.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(context,row['exit_code'],flush=True)
    if process.returncode:raise SystemExit(process.returncode)
