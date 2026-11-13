from pathlib import Path
import os,subprocess,re,json,sys,hashlib
root=Path.cwd();out=root/'backend/benchmarks/gemma4-decode-profile-20261113'
build=sys.argv[1] if len(sys.argv)>1 else 'build-cuda'
tag='' if build=='build-cuda' else build+'-'
exe=root/build/'profile-scope-oracle.exe'
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')}
env.update(NYA_COMPUTE='cuda',NYA_CUDA_BATCH='8')
p=subprocess.run([str(exe),str(root/'build-cuda/test_models/pretraining.trained.gguf')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(out/(tag+'scope.log')).write_bytes(p.stdout+p.stderr)
assert p.returncode==0,p.returncode
scopes=re.findall(rb'CUDA profile scope: from_position=(\d+) batches=(\d+) tokens=(\d+)',p.stderr)
expected=[(b'0',b'10',b'38'),(b'10',b'6',b'6'),(b'16',b'6',b'6'),(b'17',b'4',b'4'),(b'33',b'0',b'0'),(b'0',b'10',b'38')]
assert scopes==expected,(scopes,expected)
# Windows empty env removes the variable, restoring the default full scope.
assert p.stderr.count(b'CUDA profiling disabled: invalid NYA_CUDA_PROFILE_FROM')==2
(out/(tag+'scope-results.json')).write_text(json.dumps({'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'exit_code':p.returncode,'exact_logits':True,'scopes':[[int(v) for v in row] for row in scopes],'invalid_settings_disabled':2},indent=2)+'\n')
print('Exact logits, reset-aware thresholds, excluded batches, and invalid-option fallback passed')
