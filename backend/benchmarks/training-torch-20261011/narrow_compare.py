"""Before/after LoRA scheduling comparison; alternate executable order."""
from pathlib import Path
import argparse,csv,hashlib,json,os,statistics,subprocess,sys
root=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('D:/fyodor-validation/narrow-compare-20261011'))
parser.add_argument('--before',type=Path,default=Path('D:/fyodor-validation/fyodor-train-before-narrow-20261011.exe'))
parser.add_argument('--contexts',type=int,nargs='+',default=[8,32,128])
parser.add_argument('--steps',type=int,default=3)
parser.add_argument('--warmup',type=int,default=1)
parser.add_argument('--repeats',type=int,default=2)
parser.add_argument('--rank',type=int,default=2)
parser.add_argument('--monitor-environment',action='store_true',help='Require monitored desktop idle and reject competing GPU identities')
parser.add_argument('--desktop-window',action='store_true',help='Use recorded ten-sample desktop gate; requires environment monitoring')
args=parser.parse_args()
assert args.steps>args.warmup>=1 and args.repeats>=2 and args.rank>0 and all(c>0 for c in args.contexts)
assert not args.desktop_window or args.monitor_environment
work=args.output.resolve();work.mkdir(parents=True,exist_ok=False)
if args.monitor_environment:
    sys.path.insert(0,str(root/'backend/benchmarks/score-refresh-20261011'))
    import environment
    environment.configure(work,window=args.desktop_window)

def execute(command):
    process=subprocess.Popen(command,env=dict(os.environ,NYA_TRAIN_PROFILE='0'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        if args.monitor_environment:environment.set_active_pid(process.pid)
        stdout,stderr=process.communicate(timeout=1800)
    except BaseException:
        process.kill();process.communicate();raise
    return subprocess.CompletedProcess(command,process.returncode,stdout,stderr)

if args.monitor_environment:execute=environment.capture('whole-step-lora')(execute)

executables={'before':args.before.resolve(),
             'after':root/'build-cuda/fyodor-train.exe'}
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
corpus=work/'corpus.txt';corpus.write_text('The small cat sat beside the quiet library. '*300)
record={'executables':{key:{'path':str(p),'sha256':sha(p)} for key,p in executables.items()},
        'method':'Whole update; warmups discarded; alternating executable order; profiler disabled',
        'configuration':vars(args)|{'output':str(work),'before':str(args.before)},
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'cuda_source_sha256':sha(root/'CUDA/training.cu'),
        'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'),'runs':[]}
for repeat in range(args.repeats):
    for context in args.contexts:
        for version in (('before','after') if repeat%2==0 else ('after','before')):
            name=f'{repeat}-{context}-{version}'; export=work/(name+'.gguf')
            command=[str(executables[version]),'--mode','cpt','--base',str(root/'.tools/tinyllama-q4_k_m.gguf'),
                '--rank',str(args.rank),'--context',str(context),'--steps',str(args.steps),'--threads','6','--compute','cuda',
                '--memory-mib','4096','--device-memory-mib','6144','--data',str(corpus),
                '--output',str(export),'--checkpoint',str(work/(name+'.ckpt')),'--metrics',str(work/(name+'.csv'))]
            if args.monitor_environment:
                p=execute(command)
            else:
                with (work/(name+'-gpu.csv')).open('wb') as gpu:
                    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm','--format=csv,nounits','-lms','500'],stdout=gpu,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    try:p=execute(command)
                    finally:monitor.terminate();monitor.wait(timeout=10)
            (work/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8');assert p.returncode==0,p.stderr
            with (work/(name+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
            assert len(rows)==args.steps
            row={'name':name,'repeat':repeat,'context':context,'version':version,'command':command,
                 'metrics':rows,'mean_step_ms':statistics.mean(float(r['step_ms']) for r in rows[args.warmup:]),
                 'mean_forward_ms':statistics.mean(float(r['forward_ms']) for r in rows[args.warmup:]),
                 'checkpoint_sha256':sha(work/(name+'.ckpt')),'export_sha256':sha(export)}
            record['runs'].append(row)
            (work/'results.json').write_text(json.dumps(record,indent=2)+'\n')
            print(name,row['mean_step_ms'],row['mean_forward_ms'],flush=True)
            # Exact generated export only; hashes retained before reclaiming disk.
            assert export.parent.resolve()==work.resolve() and export.is_file();export.unlink()
for context in args.contexts:
    runs=[r for r in record['runs'] if r['context']==context]
    assert len({r['checkpoint_sha256'] for r in runs})==1
    assert len({r['export_sha256'] for r in runs})==1
record['all_outputs_exact']=True
(work/'results.json').write_text(json.dumps(record,indent=2)+'\n')
