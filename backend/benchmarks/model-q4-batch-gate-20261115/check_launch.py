"""Exercise actual fused host launch wiring and error paths without a GPU."""
from pathlib import Path
import hashlib,json,os,subprocess
root=Path.cwd();out=Path(__file__).resolve().parent
work=root/'build-cuda/model-q4-batch-gate-20261115'
source=(work/'resident.inc').read_text()
a=source.index('static int plan_batch_gate(');b=source.index('static int plan_gate_up(',a)
fixture=r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
typedef uint64_t CUdeviceptr;
typedef struct { CUdeviceptr address; } nya_llm_tensor;
typedef struct { int interleave_four, batch_gate; } fake_cuda;
typedef struct {
 fake_cuda *cuda;
 CUdeviceptr norm, interleaved_input, gate;
 unsigned matrix_profile;
 unsigned long long batch_gate_calls;
} nya_cuda_plan;
static int failure,converts,kernels;
static CUdeviceptr plan_tensor(nya_cuda_plan *p,const nya_llm_tensor *t)
{ (void)p;return t->address; }
static unsigned plan_matrix_slot(nya_cuda_plan *p,size_t r,size_t c,unsigned t,unsigned path)
{ (void)p;assert(r==15360 && c==3840 && t==12 && path==5);return 7; }
static int plan_launch(nya_cuda_plan *p,int f,size_t blocks,void **args)
{
 assert(f==11 && blocks==15);
 assert(*(CUdeviceptr *)args[0]==p->norm);
 assert(*(CUdeviceptr *)args[1]==p->interleaved_input);
 assert(*(unsigned long long *)args[2]==3840);
 ++converts;return failure==1?-1:0;
}
static int plan_grid(nya_cuda_plan *p,int f,size_t gx,size_t gy,unsigned bx,unsigned by,void **args)
{
 assert(converts==1 && f==22 && gx==1920 && gy==1 && bx==256 && by==1);
 assert(*(CUdeviceptr *)args[0]==101 && *(CUdeviceptr *)args[1]==202);
 assert(*(CUdeviceptr *)args[2]==p->interleaved_input);
 assert(*(CUdeviceptr *)args[3]==p->gate);
 assert(*(unsigned long long *)args[4]==3840);
 assert(*(unsigned long long *)args[5]==2160);
 assert(*(unsigned *)args[6]==15360 && *(unsigned *)args[7]==4);
 assert(p->matrix_profile==7);
 ++kernels;return failure==2?-1:0;
}
'''+source[a:b]+r'''
int main(void)
{
 fake_cuda cuda={11,22};nya_llm_tensor gate={101},up={202};
 for(failure=0;failure<3;++failure) {
  nya_cuda_plan p={&cuda,303,404,505,0,9};converts=kernels=0;
  int result=plan_batch_gate(&p,&gate,&up,15360,3840);
  assert(result==(failure?-1:0));
  assert(converts==1 && kernels==(failure==1?0:1));
  assert(p.batch_gate_calls==(failure?9:10));
  assert(p.matrix_profile==0);
  assert(p.norm==303 && p.interleaved_input==404 && p.gate==505);
 }
 return 0;
}
'''
path=work/'launch-control.c';path.write_text(fixture)
compiler=root/'.tools/w64devkit/bin/gcc.exe'
env=dict(os.environ,PATH=str(compiler.parent)+os.pathsep+os.environ['PATH'])
inputs=[Path(__file__).resolve(),work/'resident.inc',path,compiler];commands=[]
for standard in ('c17','c23'):
 exe=work/f'launch-control-{standard}.exe'
 command=[str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic','-Wconversion',
          '-Wshadow','-Werror',str(path),'-o',str(exe)]
 subprocess.run(command,env=env,check=True);subprocess.run([str(exe)],check=True)
 commands.append(command);inputs.append(exe)
(out/'launch-control.json').write_text(json.dumps(dict(passed=True,gpu_execution=False,
 cases_per_standard=3,commands=commands,sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))
print('C17/C23 launch arguments, fresh interleave and error propagation passed')
