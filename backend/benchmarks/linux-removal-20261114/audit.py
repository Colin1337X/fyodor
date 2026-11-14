"""Audit completed physical-removal Linux builds without running GPU work."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
out=Path(__file__).resolve().parent
standard=sys.argv[1]
if standard not in ('c17','c23'): raise SystemExit('Choose c17 or c23')
dest=out/f'{standard}.json'
if dest.exists(): raise SystemExit('Refusing to overwrite evidence')
log=out/f'{standard}.log';text=log.read_text()
roots=re.findall(r'Verified without non-c: (.+); benchmark JSON retained',text)
assert len(roots)==1 and '100% tests passed, 0 tests failed out of 82' in text
root=Path(roots[0]);source=root/'source';build=root/'build'
absent={n:not (source/n).exists() for n in ('optcpp','backend/optcpp','CUDA/cutlass','CUDA/cute')}
assert all(absent.values())
cache=(build/'CMakeCache.txt').read_text()
assert not re.search(r'^CMAKE_CXX_COMPILER[:_]',cache,re.M)
assert f'NYA_C_STANDARD:STRING={standard[1:]}' in cache
assert 'NYA_ENABLE_OPTCPP:BOOL=ON' in cache
assert 'NYA_WARNINGS_AS_ERRORS:BOOL=ON' in cache
links={}
for name in ('fyodor-backend','fyodor-bench','fyodor-train','fyodor-tui'):
 links[name]=subprocess.check_output(['ldd',str(build/name)],text=True)
 assert 'libstdc++' not in links[name] and 'libc++' not in links[name]
skips=re.findall(r'\d+ - (\S+) \(Skipped\)',text)
assert sorted(skips)==sorted(['hardware-rocm-quant','hardware-rocm-inference','hardware-mlx-quant','hardware-mlx-inference'])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(passed=True,accepted_benchmark=False,standard=standard,source=str(source),build=str(build),
 absent=absent,cxx_compiler_present=False,tests=dict(total=82,passed=78,skipped=skips,failed=0),
 dependencies=links,source_sha256={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()},
 artifact_sha256={str(p):digest(p) for p in (log,Path(__file__),build/'CMakeCache.txt',root/'before.json',root/'after.json')},
 limitation='Synthetic smoke timing during a separate GPU sanitizer run; not performance acceptance or a real-model comparison.')
dest.write_text(json.dumps(result,indent=2))
print(standard,'physical absence, C-only compiler/linkage and 78 passes / 4 expected skips verified')
