from pathlib import Path
import os,json,subprocess,hashlib
root=Path.cwd();out=root/'backend/benchmarks/gemma4-ring-runtime-20261113'
env={k:v for k,v in os.environ.items() if not k.startswith(('NYA_','GGML_'))}
env.update(json.loads((out/'results.json').read_text())['environment'])
exe=root/'build-cuda/gemma-ring-rewind-oracle.exe'
p=subprocess.run([str(exe),'2051'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(out/'rewind.log').write_bytes(p.stdout+p.stderr)
(out/'rewind-results.json').write_text(json.dumps(dict(exit_code=p.returncode,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),environment={k:v for k,v in env.items() if k.startswith('NYA_')}),indent=2)+'\n')
print(p.returncode,p.stdout.decode(errors='replace'),flush=True)
raise SystemExit(p.returncode)
