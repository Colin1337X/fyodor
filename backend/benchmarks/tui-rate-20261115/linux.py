"""Build current TUI in fresh Linux trees physically lacking optional C++ code."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root=Path.cwd()
out=Path(__file__).resolve().parent
standard=sys.argv[1]
if standard not in ('17','23'): raise SystemExit('Choose 17 or 23')
work=Path.home()/'.cache'/('fyodor-tui-rate-20261115-c'+standard)
work.mkdir(exist_ok=False)
source=work/'source';source.mkdir()
for name in ('CMakeLists.txt','config.yaml'):
    shutil.copy2(root/name,source/name)
for name in ('backend','CUDA','rocm','mlx','themes'):
    shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('optcpp','cutlass','cute','build*','benchmarks','__pycache__'))
for name in ('scripts/generate-themes.py','frontend/public/themes.js','frontend/public/theme-licenses.txt','frontend/src/themes.css','backend/benchmarks/training_device_bench.c'):
    target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
absent={name:not (source/name).exists() for name in ('optcpp','backend/optcpp','CUDA/cutlass','CUDA/cute')}
assert all(absent.values())
build=work/'build'
commands=[['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
           '-DNYA_C_STANDARD='+standard,'-DNYA_WARNINGS_AS_ERRORS=ON','-DBUILD_TESTING=ON',
           '-DNYA_ENABLE_OPTCPP=ON','-DNYA_ENABLE_CUDA=OFF','-DNYA_ENABLE_VULKAN=OFF',
           '-DNYA_ENABLE_ROCM=OFF','-DNYA_ENABLE_MLX=OFF','-DNYA_ENABLE_ONNXRUNTIME=OFF'],
          ['cmake','--build',str(build),'--target','nya-test-tui','fyodor-tui','fyodor-train','--parallel','2'],
          ['ctest','--test-dir',str(build),'--output-on-failure','-R','^(tui-|train-learning-rate-metrics$|train-evaluation$)']]
log=out/('c'+standard+'.log')
with log.open('x') as stream:
    for command in commands:
        stream.write(json.dumps(command)+'\n');stream.flush()
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
cache=(build/'CMakeCache.txt').read_text()
assert 'CMAKE_CXX_COMPILER' not in cache
links=subprocess.check_output(['ldd',str(build/'fyodor-tui')],text=True)
assert 'libstdc++' not in links and 'libc++' not in links
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(passed=True,accepted_benchmark=False,standard=standard,commands=commands,
    absent=absent,dependencies=links,source=str(source),build=str(build),
    source_sha256={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()},
    sha256={str(p):digest(p) for p in (Path(__file__),log,build/'CMakeCache.txt',build/'nya-test-tui',build/'fyodor-tui',build/'fyodor-train')},
    limitation='TUI, learning-rate and evaluation tests only; no full-suite, GPU or performance claim.')
(out/('c'+standard+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print('C'+standard+' TUI/rate/evaluation tests passed without optional C++ source/runtime',flush=True)
