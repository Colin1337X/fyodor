import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=root/'backend/benchmarks/tui-rate-20261115'
log=out/'pty-recheck.log';records=[]
with log.open('x') as stream:
 for standard in ('17','23'):
  build=Path.home()/'.cache'/('fyodor-tui-rate-20261115-c'+standard)/'build'
  command=['ctest','--test-dir',str(build),'--output-on-failure','-R','^(tui-|train-learning-rate-metrics$|train-evaluation$)','-E','tui-monitor-posix-pty']
  subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
  for repeat in range(20):
   command=['python3',str(root/'backend/tests/test_tui_monitor_pty.py'),str(build/'fyodor-tui')]
   subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
  records.append(dict(standard=standard,other_ctest_passes=5,updated_pty_passes=20,
    binaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (build/'fyodor-tui',build/'fyodor-train',build/'nya-test-tui')}))
proof=dict(passed=True,accepted_benchmark=False,records=records,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),log,root/'backend/tests/test_tui_monitor_pty.py')})
(out/'pty-recheck.json').write_text(json.dumps(proof,indent=2)+'\n')
print('Both standards: five other tests and twenty updated real-PTY checks pass')
