"""Isolated batch-512 reduction-order experiment; production library is untouched."""
import difflib,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/ordered-prefix-diagnostic'
if work.exists():raise SystemExit('Refusing to overwrite experiment')
work.mkdir()
resident=root/'CUDA/resident.inc';original=resident.read_text()
anchor='    if (p->batch > 1) {\n        if (p->blas_storage && nya_blas_bytes'
replacement='''    if (p->batch > 1) {
        /* Diagnostic only: retain large batches, use the serial reduction order. */
        if ((type==12 || type==14) && p->cuda->small_reuse[type][1]) {
            unsigned r=(unsigned)rows,batch=(unsigned)p->batch;
            void *ordered[]={&matrix,&in,&out,&cols,&stride,&r,&batch};
            return plan_grid(p,p->cuda->small_reuse[type][1],(rows+7)/8,
                (p->batch+3)/4,256,1,ordered);
        }
        if (p->blas_storage && nya_blas_bytes'''
assert original.count(anchor)==1
modified=original.replace(anchor,replacement)
(work/'resident_ordered.inc').write_text(modified)
patch=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='CUDA/resident.inc',tofile='CUDA/resident.inc'))
with (out/'ordered-prefix.patch').open('x') as f:f.write(patch)
cuda=(root/'CUDA/cuda.c').read_text();assert cuda.count('#include "resident.inc"')==1
(work/'cuda_ordered.c').write_text(cuda.replace('#include "resident.inc"','#include "resident_ordered.inc"'))
probe=(out/'prefix_kv.c').read_text()
anchor='    CHECK(!strcmp(nya_llm_session_execution(s),"resident"));'
assert probe.count(anchor)==1
probe=probe.replace(anchor,anchor+'''
    nya_compute_stats stats={0};nya_llm_session_stats(s,&stats);
    printf("{\\"kind\\":\\"plan\\",\\"prefill_batch\\":%zu}\\n",stats.prefill_batch);
    CHECK(stats.prefill_batch==512);''')
(work/'probe.c').write_text(probe)
flags=(root/'build-cuda/CMakeFiles/nya-engine.dir/flags.make').read_text().splitlines()
defines=next(l.split(' = ',1)[1].split() for l in flags if l.startswith('C_DEFINES = '))
cflags=next(l.split(' = ',1)[1].split() for l in flags if l.startswith('C_FLAGS = '))
compiler=root/'.tools/w64devkit/bin/gcc.exe';ar=compiler.with_name('ar.exe')
env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env['PATH']
obj=work/'cuda.c.obj';library=work/'libnya-engine.a';exe=work/'probe.exe'
base=root/'build-cuda/libnya-engine.a';commands=[]
def run(command):
 commands.append([str(a) for a in command]);subprocess.run(commands[-1],env=env,check=True)
run([compiler,*defines,*cflags,'@'+str(root/'build-cuda/CMakeFiles/nya-engine.dir/includes_C.rsp'),
     '-I',root/'CUDA','-c',work/'cuda_ordered.c','-o',obj])
run([compiler,*defines,*[('-std=c23' if a=='-std=c17' else a) for a in cflags],
     '@'+str(root/'build-cuda/CMakeFiles/nya-engine.dir/includes_C.rsp'),'-I',root/'CUDA',
     '-c',work/'cuda_ordered.c','-o',work/'cuda-c23.obj'])
members=subprocess.check_output([str(ar),'t',str(base)],text=True).splitlines()
assert members.count('cuda.c.obj')==1
shutil.copyfile(base,library);run([ar,'r',library,obj])
run([compiler,'-std=c17','-O3','-Wall','-Wextra','-Wconversion','-Wshadow','-Werror',
     '-I',root/'backend/include','-I',root/'backend/core',work/'probe.c',library,
     '-lbcrypt','C:/Windows/System32/vulkan-1.dll','-o',exe])
settings=json.loads((out/'prefix-kv-control.json').read_text())['environment'].copy()
settings.update(NYA_CUDA_WIDE_SPLIT='0',NYA_CUDA_FLASH_DECODING_PP='0')
env.update(settings)
command=[str(exe),'E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','2048']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (base,library,exe,work/'cuda_ordered.c',work/'resident_ordered.inc',work/'probe.c')}
started=time.perf_counter()
with (out/'ordered-prefix.log').open('x') as stdout,(out/'ordered-prefix.stderr.log').open('x') as stderr:
 result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
elapsed=time.perf_counter()-started
rows=[];parse_error=None
try:rows=[json.loads(l) for l in (out/'ordered-prefix.log').read_text().splitlines() if l.startswith('{')]
except ValueError as error:parse_error=str(error)
with (out/'ordered-prefix.json').open('x') as f:json.dump({'accepted_benchmark':False,
 'build_commands':commands,'command':command,'environment':settings,'sha256':hashes,
 'exit_code':result.returncode,'parse_error':parse_error,'process_elapsed_seconds':elapsed,
 'rows':rows,'note':'Process duration includes load, prefix preparation, full serial replay and diagnostics; not a throughput score. Production library and dispatch remain unchanged.'},f,indent=2)
assert hashlib.sha256(base.read_bytes()).hexdigest()==hashes[str(base.relative_to(root))]
if result.returncode:raise SystemExit(result.returncode)
if parse_error:raise SystemExit(parse_error)
assert len(rows)==146 and rows[0]=={'kind':'plan','prefill_batch':512}
print(json.dumps([r for r in rows if r['kind']=='logits'][0]),flush=True)
