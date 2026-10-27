"""Build a fresh source copy with optional non-C components physically absent."""
from pathlib import Path
import shutil,uuid,subprocess,os,json
r=Path(__file__).resolve().parents[3];p=Path(__file__).resolve().parent
old=json.loads((r/'backend/benchmarks/fdpp-pipeline-20261016/removal-proof.json').read_text())
temp=r/'build-cuda'/('without-fixed-gate-'+uuid.uuid4().hex[:12]);source=temp/'source'
assert source.is_relative_to(r/'build-cuda')
source.mkdir(parents=True)
for name in ('CMakeLists.txt','config.yaml'):shutil.copy2(r/name,source/name)
for name in ('backend','CUDA','rocm','mlx'):
 if (r/name).exists():shutil.copytree(r/name,source/name,ignore=shutil.ignore_patterns('optcpp','cutlass','cute','vendor','benchmarks','build*'))
(source/'backend/benchmarks').mkdir();shutil.copy2(r/'backend/benchmarks/training_device_bench.c',source/'backend/benchmarks/training_device_bench.c')
cmd=old['configure'];cmd[cmd.index('-S')+1]=str(source);cmd[cmd.index('-B')+1]=str(temp/'build')
cmd+=['-DNYA_ENABLE_FIXED_GATE=ON']
env=os.environ.copy();env['PATH']=str(r/'.tools/w64devkit/bin')+';'+env['PATH']
with (p/'removal-build.log').open('wb') as f:
 for c in (cmd,[cmd[0],'--build',str(temp/'build'),'--target','fyodor-bench','-j','8']):
  a=subprocess.run(c,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180);assert a.returncode==0
absent={name:not (source/name).exists() for name in old['absent']};assert all(absent.values())
assert 'CMAKE_CXX_COMPILER:' not in (temp/'build/CMakeCache.txt').read_text()
smoke=old['smoke'];smoke[0]=str(temp/'build/fyodor-bench.exe')
env['NYA_CUDA_FIXED_GATE']='1';env['NYA_CUDA_FLASH_DECODING_PP']='short';env['NYA_CPU_THREADS']='6'
with (p/'removal-smoke.json').open('wb') as f,(p/'removal-smoke.log').open('wb') as g:a=subprocess.run(smoke,cwd=r,env=env,stdout=f,stderr=g,timeout=120)
assert a.returncode==0
(p/'removal-proof.json').write_text(json.dumps({'source':str(source),'configure':cmd,'smoke':smoke,'absent':absent,'smoke_exit':a.returncode,'CXX_compiler_present':False},indent=2))
print('Removal build and real-model smoke passed')
