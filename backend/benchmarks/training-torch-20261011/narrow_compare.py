"""Before/after LoRA scheduling comparison; alternate executable order."""
from pathlib import Path
import csv,hashlib,json,os,statistics,subprocess
root=Path(__file__).resolve().parents[3]
work=Path('D:/fyodor-validation/narrow-compare-20261011');work.mkdir(parents=True,exist_ok=False)
executables={'before':Path('D:/fyodor-validation/fyodor-train-before-narrow-20261011.exe'),
             'after':root/'build-cuda/fyodor-train.exe'}
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
corpus=work/'corpus.txt';corpus.write_text('The small cat sat beside the quiet library. '*300)
record={'executables':{key:{'path':str(p),'sha256':sha(p)} for key,p in executables.items()},
        'method':'Three updates, first discarded; two repetitions in forward/reverse order; profiler disabled',
        'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'),'runs':[]}
for repeat in range(2):
    for context in (8,32,128):
        for version in (('before','after') if repeat==0 else ('after','before')):
            name=f'{repeat}-{context}-{version}'; export=work/(name+'.gguf')
            command=[str(executables[version]),'--mode','cpt','--base',str(root/'.tools/tinyllama-q4_k_m.gguf'),
                '--rank','2','--context',str(context),'--steps','3','--threads','6','--compute','cuda',
                '--memory-mib','4096','--device-memory-mib','6144','--data',str(corpus),
                '--output',str(export),'--checkpoint',str(work/(name+'.ckpt')),'--metrics',str(work/(name+'.csv'))]
            with (work/(name+'-gpu.csv')).open('wb') as gpu:
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm','--format=csv,nounits','-lms','500'],stdout=gpu,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    p=subprocess.run(command,env=dict(os.environ,NYA_TRAIN_PROFILE='0'),capture_output=True,text=True,timeout=1800)
                finally:monitor.terminate();monitor.wait(timeout=10)
            (work/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8');assert p.returncode==0,p.stderr
            with (work/(name+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
            assert len(rows)==3
            row={'name':name,'repeat':repeat,'context':context,'version':version,'command':command,
                 'metrics':rows,'mean_step_ms':statistics.mean(float(r['step_ms']) for r in rows[1:]),
                 'mean_forward_ms':statistics.mean(float(r['forward_ms']) for r in rows[1:]),
                 'checkpoint_sha256':sha(work/(name+'.ckpt')),'export_sha256':sha(export)}
            record['runs'].append(row)
            (work/'results.json').write_text(json.dumps(record,indent=2)+'\n')
            print(name,row['mean_step_ms'],row['mean_forward_ms'],flush=True)
            # Exact generated export only; hashes retained before reclaiming disk.
            assert export.parent.resolve()==work.resolve() and export.is_file();export.unlink()
for context in (8,32,128):
    runs=[r for r in record['runs'] if r['context']==context]
    assert len({r['checkpoint_sha256'] for r in runs})==1
    assert len({r['export_sha256'] for r in runs})==1
record['all_outputs_exact']=True
(work/'results.json').write_text(json.dumps(record,indent=2)+'\n')
