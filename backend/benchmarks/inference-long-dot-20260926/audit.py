"""Audit final evidence against current sources and the packaged backend."""
import hashlib
import json
from pathlib import Path
import re

out = Path(__file__).resolve().parent
root = out.parents[2]
def read(name):
    return json.loads((out/name).read_text())
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

matrix = read('matrix/validation.json')
assert len(matrix) == 15 and all(r['exit_code'] == 0 for r in matrix)
for name, digest in read('matrix/source-hashes.json').items():
    assert sha(root/name) == digest, name
assert '2076.98633 expected 2077' in (out/'regression-before.log').read_text()
assert '100% tests passed' in (out/'extended-after.log').read_text()
real = read('real-model.json')
assert len(real) == 2 and all(r['exit_code'] == 0 for r in real)
for row in real:
    errors = list(map(float,re.findall(r'max_scaled=([\deE.+-]+)',(out/row['log']).read_text())))
    assert len(errors) == 3 and max(errors) <= 0.001
sanitizers = read('sanitizers.json')
assert len(sanitizers) == 4 and all(r['exit_code'] == 0 for r in sanitizers)
for row in sanitizers:
    assert 'ERROR SUMMARY: 0 errors' in (out/row['log']).read_text()
bench = read('accepted-remote/metadata.json')
assert len(bench['runs']) == 20
for row in bench['runs']:
    assert row['exit_code'] == 0 and 'gpu_idle_after' in row and not row.get('contaminated')
    samples = read('accepted-remote/'+row['name']+'-resources.json')
    assert samples and all(not s.get('competing_compute_processes') for s in samples)
for name,digest in bench['source_sha256'].items():
    assert sha(root/name) == digest, name
assert sha(root/'build-cuda/fyodor-bench.exe') == bench['binary_sha256']['after']
package = read('desktop-package.json')
assert sha(Path(package['installer'])) == package['sha256']
assert package['verified_contents_sha256']['backend/fyodor-backend.exe'] == sha(root/'build-cuda/fyodor-backend.exe')
assert package['cli_exit_code'] == 0 and len(package['http_calls']) == 8
assert package['environment']['NYA_CUDA_BLAS'] == '0' and package['environment']['NYA_CUDA_CUTLASS'] == '0'
sources = ('CUDA/gemm.cu','CUDA/resident.inc','backend/CMakeLists.txt','backend/tests/test_quant_kernels.c','backend/README.md')
record = {'passed':True,'validation_stages':len(matrix),'sanitizer_runs':len(sanitizers),
          'accepted_process_runs':len(bench['runs']),'installer_sha256':package['sha256'],
          'source_sha256':{name:sha(root/name) for name in sources}}
(out/'audit.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
