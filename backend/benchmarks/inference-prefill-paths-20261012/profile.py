"""Compare actual prefill dispatch paths; diagnostic timing, not acceptance."""
import argparse, datetime, hashlib, json, os, re, subprocess
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--cutlass-library',type=Path)
parser.add_argument('--prompt',type=int,choices=(128,512),default=512)
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
exe=root/'build-cuda/fyodor-bench.exe'
model=root/'.tools/tinyllama-q4_k_m.gguf'
blas=root/'.tools/llama-b10809-cuda/cublas64_13.dll'
cutlass=(args.cutlass_library or root/'build-cutlass/fyodor-cutlass.dll').resolve()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gpu():
    return subprocess.check_output(['nvidia-smi'],text=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
base={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
base.update(NYA_CPU_THREADS='6',NYA_CUDA_BATCH='512',NYA_CUDA_BLAS_LIBRARY=str(blas),NYA_CUDA_CUTLASS_LIBRARY=str(cutlass))
meta={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
      'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'caveat':'Diagnostic interactive desktop runs only; no performance acceptance. Event profiles disable graphs and include host submission gaps.',
      'sha256':{str(p):sha(p) for p in (exe,model,blas,cutlass,Path(__file__),root/'CUDA/blas.inc',root/'CUDA/resident.inc',root/'CUDA/cutlass/gemm.cu')},'runs':[]}
(out/'processes-before.txt').write_text(gpu())
pattern=re.compile(r'CUDA matrix profile: path=(\S+) type=(\d+) rows=(\d+) columns=(\d+) batch=(\d+) calls=(\d+) milliseconds=([\d.]+)')
for phase,providers in [('a',('native','cutlass','cublas')),('b',('cublas','cutlass','native'))]:
    for provider in providers:
        name=f'{provider}-{phase}'
        env=base|{'NYA_CUDA_BLAS':str(int(provider=='cublas')),'NYA_CUDA_CUTLASS':str(int(provider=='cutlass')),'NYA_CUDA_PROFILE':'1'}
        cmd=[str(exe),'-m',str(model),'-b','cuda','-p',str(args.prompt),'-n','0','-r','2','--warmup','1','--json']
        row={'name':name,'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith('NYA_')}}
        meta['runs'].append(row)
        proc=subprocess.run(cmd,env=env,capture_output=True,timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
        row['exit_code']=proc.returncode
        (out/(name+'.json')).write_bytes(proc.stdout);(out/(name+'.log')).write_bytes(proc.stderr)
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        assert proc.returncode==0,row
        data=json.loads(proc.stdout);assert data['execution']=='resident' and data['kv_type']=='f32'
        result=data['results'][0];assert result['test']==f'pp{args.prompt}'
        assert result['external_matmul_calls']==(0 if provider=='native' else 308)
        assert result['cutlass_matmul_calls']==(308 if provider=='cutlass' else 0)
        rows=[dict(zip(('path','type','rows','columns','batch','calls','milliseconds'),(m[0],*map(int,m[1:6]),float(m[6])))) for m in pattern.findall(proc.stderr.decode())]
        assert rows and not any(r['path']=='overflow' for r in rows)
        assert sum(r['calls'] for r in rows if r['path']=='dot')==3
        expected='native-gemm' if provider=='native' else 'vendor-gemm'
        assert sum(r['calls'] for r in rows if r['path']==expected)==462
        aggregate=re.search(r'matmul=([\d.]+).*attention=([\d.]+)',proc.stderr.decode())
        row.update(matmul_ms=float(aggregate[1]),attention_ms=float(aggregate[2]),shapes=sorted(rows,key=lambda r:-r['milliseconds']),dispatch_verified=True)
        (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(name,'matrix ms',row['matmul_ms'],'attention ms',row['attention_ms'],'top',row['shapes'][0],flush=True)
(out/'processes-after.txt').write_text(gpu())
