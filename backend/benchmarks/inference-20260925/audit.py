"""Check the recorded acceptance, correctness and package evidence together."""
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
matrix=json.loads((out/'matrix/validation.json').read_text())
assert len(matrix)==15 and all(r['exit_code']==0 for r in matrix)
for name,digest in json.loads((out/'matrix/source-hashes.json').read_text()).items():
    assert sha(root/name)==digest,name
sanitizers=json.loads((out/'cuda-sanitizers.json').read_text())
assert len(sanitizers)==4 and all(r['exit_code']==0 for r in sanitizers)
for row in sanitizers:
    text=(out/row['log']).read_text()
    assert '2 graph captures; 795 checked decodes' in text
    assert '0 errors' in text
for name in ('real-273.log','real-1153-cpu-kernels.log'):
    values=re.findall(r'max_scaled=([0-9.e+-]+)',(out/name).read_text())
    assert len(values)==3 and max(map(float,values))<0.001
bench=json.loads((out/'accepted-clean3/metadata.json').read_text())
assert len(bench['runs'])==24
assert all(r['exit_code']==0 and 'gpu_idle_after' in r and not r.get('contaminated') for r in bench['runs'])
for name,digest in bench['source_sha256'].items():assert sha(root/name)==digest,name
assert sha(root/'build-cuda/fyodor-bench.exe')==bench['binary_sha256']['after']
package=json.loads((out/'desktop-package.json').read_text())
assert sha(package['installer'])==package['sha256']
assert package['verified_contents_sha256']['backend/fyodor-backend.exe']==sha(root/'build-cuda/fyodor-backend.exe')
assert package['cli_exit_code']==0 and len(package['http_calls'])==8
late=json.loads((out/'late-failure.json').read_text())
assert len(late)==2 and all(r['exit_code']==0 and r['all_three_prefixes_exact'] for r in late)
hashes={name.replace('\\','/'):digest for name,digest in bench['source_sha256'].items()}
for name in ('CUDA/resident.cu','CUDA/resident.inc','CUDA/cuda.c','backend/CMakeLists.txt','backend/tests/test_inference.c','backend/README.md'):
    hashes[name]=sha(root/name)
record={'validation_stages':len(matrix),'cuda_sanitizers':len(sanitizers),'accepted_process_runs':len(bench['runs']),'installer_sha256':package['sha256'],'source_sha256':hashes,'baseline_revision':bench['revision'],'passed':True}
(out/'audit.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
