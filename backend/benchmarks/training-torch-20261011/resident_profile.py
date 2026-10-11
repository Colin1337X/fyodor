"""Check profiling transparency, then collect diagnostic operation timings."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess
root=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('D:/fyodor-validation/resident-operation-profile-20261011'))
parser.add_argument('--context',type=int,default=32)
parser.add_argument('--rank',type=int,default=2)
args=parser.parse_args()
assert args.context>0 and args.rank>0
work=args.output.resolve()
work.mkdir(parents=True,exist_ok=False)
trainer=root/'build-cuda/fyodor-train.exe'
corpus=work/'corpus.txt';corpus.write_text('The small cat sat beside the quiet library. '*100)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
record={'trainer_sha256':sha(trainer),'training_source_sha256':sha(root/'backend/core/training.c'),
        'cuda_training_source_sha256':sha(root/'CUDA/training.cu'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'model_sha256':sha(root/'.tools/tinyllama-q4_k_m.gguf'),
        'method':'Synchronized host diagnostic; unsuitable for competitive throughput acceptance','runs':[]}
def run(name,extra,profile):
    command=[str(trainer),'--data',str(corpus),'--steps','3','--threads','6','--compute','cuda',
             '--memory-mib','4096','--device-memory-mib','6144','--output',str(work/(name+'.gguf')),
             '--checkpoint',str(work/(name+'.ckpt')),'--metrics',str(work/(name+'.csv'))]+extra
    env=dict(os.environ,NYA_TRAIN_PROFILE='1' if profile else '0')
    p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=1800)
    (work/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
    assert p.returncode==0,p.stderr
    lines=[line for line in p.stderr.splitlines() if line.startswith('train_profile_resident ')]
    assert bool(lines)==profile
    summaries=[]
    for line in lines:
        fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
        assert fields['timing']=='synchronized_host_diagnostic'
        assert int(fields['forward_linear_calls'])>0 and int(fields['backward_linear_calls'])>0
        summaries.append(fields)
    matrices=[]
    for line in p.stderr.splitlines():
        if line.startswith('train_profile_resident '):matrices.append([])
        elif line.startswith('train_profile_matrix_table '):
            fields=dict(re.findall(r'(\w+)=([^ ]+)',line));assert fields['available']=='1'
        elif line.startswith('train_profile_matrix '):
            assert matrices
            matrices[-1].append(dict(re.findall(r'(\w+)=([^ ]+)',line)))
    assert len(matrices)==len(summaries)
    for aggregate,entries in zip(summaries,matrices):
        assert entries and len(entries)<=64
        for phase in ('forward','backward'):
            expected_calls=sum(int(aggregate.get(f'{phase}_{kind}_calls',0)) for kind in ('linear','mapped_linear'))
            assert sum(int(e[f'{phase}_calls']) for e in entries)==expected_calls
            expected_ms=sum(float(aggregate.get(f'{phase}_{kind}_ms',0)) for kind in ('linear','mapped_linear'))
            actual_ms=sum(float(e[f'{phase}_ms']) for e in entries)
            assert abs(expected_ms-actual_ms)<(len(entries)+2)*1e-6
    row={'name':name,'command':command,'profile':profile,'checkpoint_sha256':sha(work/(name+'.ckpt')),
         'export_sha256':sha(work/(name+'.gguf')),'operation_summaries':summaries,'matrix_summaries':matrices}
    record['runs'].append(row)
    (work/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    return row
shape=['--dimension','256','--ff','704','--layers','2','--heads','8','--kv-heads','2','--context','128']
off=run('random-off',shape,False);on=run('random-on',shape,True)
assert off['checkpoint_sha256']==on['checkpoint_sha256']
assert off['export_sha256']==on['export_sha256']
run('tinyllama-on',['--mode','cpt','--base',str(root/'.tools/tinyllama-q4_k_m.gguf'),
                   '--rank',str(args.rank),'--context',str(args.context)],True)
record['profile_on_off_exact']=True
(work/'results.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
