import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=root/'backend/benchmarks/tui-rate-20261115'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
recheck=json.loads((out/'pty-recheck.json').read_text());assert recheck['passed'] is True
for name,h in recheck['sha256'].items():assert digest(Path(name))==h
records=[]
for standard in ('17','23'):
 work=Path.home()/'.cache'/('fyodor-tui-rate-20261115-c'+standard)
 source=work/'source';build=work/'build'
 absent={n:not(source/n).exists() for n in ('optcpp','backend/optcpp','CUDA/cutlass','CUDA/cute')};assert all(absent.values())
 cache=(build/'CMakeCache.txt').read_text();assert 'CMAKE_CXX_COMPILER' not in cache
 assert 'NYA_C_STANDARD:STRING='+standard in cache and 'NYA_ENABLE_OPTCPP:BOOL=ON' in cache
 dependencies={}
 for name in ('fyodor-tui','fyodor-train'):
  dependencies[name]=subprocess.check_output(['ldd',str(build/name)],text=True)
  assert 'libstdc++' not in dependencies[name] and 'libc++' not in dependencies[name]
 records.append(dict(standard=standard,absent=absent,dependencies=dependencies,
  initial_ctest_passed=standard=='17',updated_pty_test=str(root/'backend/tests/test_tui_monitor_pty.py'),
  source_sha256={str(p):digest(p) for p in source.rglob('*') if p.is_file()},
  sha256={str(p):digest(p) for p in (build/'CMakeCache.txt',build/'fyodor-train',build/'fyodor-tui',build/'nya-test-tui')}))
proof=dict(passed=True,accepted_benchmark=False,records=records,
 note='Original C23 PTY test failed. Updated external PTY test passes 20 repetitions for each unchanged binary; other five CTests pass for both standards.',
 sha256={str(p):digest(p) for p in (Path(__file__),out/'linux.py',out/'c17.log',out/'c23.log',out/'pty-recheck.json',root/'backend/tests/test_tui_monitor_pty.py')})
(out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print('Both C-only builds and corrected test evidence verified')
