import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
base=root/'build-cuda/model-attention-query-cache-v2-20261115'
work=root/'build-cuda/lifecycle-progress-20261115'
proof_path=root/'backend/benchmarks/model-attention-query-cache-20261115/build.json'
proof=json.loads(proof_path.read_text())
if work.exists():raise SystemExit('Refusing to overwrite build')
for name,h in proof['sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
work.mkdir()
for name in ('cuda.c','cuda_source.h'):(work/name).write_bytes((base/name).read_bytes())
(work/'lifecycle.c').write_bytes((root/'build-cuda/lifecycle-verbose-v2-20261115/lifecycle.c').read_bytes())
s=(base/'resident.inc').read_text()
helper=r'''
/* Diagnostic build only: timestamps measure this instrumented run. */
static double progress_seconds(void)
{
    LARGE_INTEGER ticks, frequency;
    if (!QueryPerformanceCounter(&ticks) || !QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) return 0;
    return (double)ticks.QuadPart / (double)frequency.QuadPart;
}
static void progress_emit(const char *event, size_t position, size_t done, size_t total, size_t layer, size_t layers, double started)
{
    double elapsed=progress_seconds()-started;
    fprintf(stderr,"PROGRESS event=%s position=%zu completed=%zu total=%zu percent=%.3f layer=%zu layers=%zu elapsed_s=%.3f tokens_per_s=%.6f\n",
        event,position,done,total,total?100.0*(double)done/(double)total:0.0,layer,layers,elapsed,elapsed>0?(double)done/elapsed:0.0);
    fflush(stderr);
}
'''
s=helper+s
start=s.index('static int cuda_plan_batch(');end=s.index('static int cuda_plan_token(',start)
part=s[start:end]
part=part.replace('    p->batch = count;','    double progress_batch_started=progress_seconds();\n    p->batch = count;',1)
needle='    for (size_t layer = 0; layer < m->block_count; ++layer) {'
assert part.count(needle)==1
part=part.replace(needle,needle+'\n        if(count>4) progress_emit("layer_begin",position,0,count,layer+1,m->block_count,progress_batch_started);')
needle='        if (w->output_scale) RUN(plan_vector(p, p->x, p->x, 0, d, 0, nya_llm_tensor_value(w->output_scale, 0)));'
assert part.count(needle)==1
part=part.replace(needle,needle+'\n        if(count>4) { RUN(c->stream_sync(c->stream)!=CUDA_SUCCESS); progress_emit("layer_complete",position,0,count,layer+1,m->block_count,progress_batch_started); }')
part=part.replace('batch_done:\n','batch_done:\n    if(count>4) RUN(c->stream_sync(c->stream)!=CUDA_SUCCESS);\n',1)
s=s[:start]+part+s[end:]
start=s.index('static int cuda_plan_prefill(');end=s.index('static int cuda_plan_verify(',start)
part=s[start:end]
part=part.replace('    for (size_t i = 0;', '    double progress_started=progress_seconds();\n    progress_emit("prefill_begin",position,0,count,0,0,progress_started);\n    for (size_t i = 0;',1)
part=part.replace('        if (cuda_plan_batch', '        progress_emit("batch_begin",position+i,i,count,0,0,progress_started);\n        if (cuda_plan_batch',1)
part=part.replace('        i += chunk;', '        i += chunk;\n        progress_emit("batch_complete",position+i,i,count,0,0,progress_started);',1)
part=part.replace('    return 0;', '    progress_emit("prefill_complete",position,count,count,0,0,progress_started);\n    return 0;',1)
s=s[:start]+part+s[end:]
(work/'resident.inc').write_text(s)
env=dict(os.environ,PATH=str(root/'.tools/w64devkit/bin')+os.pathsep+os.environ['PATH'])
commands=[]
for original in proof['commands']:
 if any('generation' in arg for arg in original):continue
 command=[arg.replace(str(base),str(work)) for arg in original]
 subprocess.run(command,env=env,check=True);commands.append(command)
assert len(commands)==4
for name,h in proof['sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
paths=[Path(__file__).resolve(),proof_path,*work.iterdir()]
(out/'build.json').write_text(json.dumps(dict(build_passed=True,accepted_benchmark=False,commands=commands,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2))
print('C17/C23 progress builds passed')
