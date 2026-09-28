"""Audit the current evaluation change against preserved build/process evidence."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

sources = json.loads((out/'source-hashes.json').read_text())
assert len(sources) == 27
for name, digest in sources.items():
    assert sha(root/name) == digest,name
matrix = json.loads((out/'validation.json').read_text())
assert len(matrix) == 15 and all(r['exit_code'] == 0 for r in matrix)
package = json.loads((out/'desktop-package.json').read_text())
assert sha(Path(package['installer'])) == package['sha256']
assert not package['installed'] and package['cli_exit_code'] == 0
counts = {}
for name, expected in [('process.json',36),('packaged-evaluation.json',36),
                       ('packaged-paths.json',24),('packaged-stop-workflows.json',54)]:
    records = json.loads((out/name).read_text())
    assert len(records) == expected,name
    counts[name] = len(records)
real = json.loads((out/'real-evaluation.json').read_text())
prior = json.loads((out.parent/'training-execution-20260925/real-resume.json').read_text())
for key in ('model_sha256','checkpoint_sha256'):
    assert real[key] == prior[key]
record = {'source_hashes_verified':len(sources),'validation_stages_passed':len(matrix),
          'process_counts':counts,'real_resume_exact':True,'installer_sha256':package['sha256'],
          'installer_bytes':package['bytes']}
(out/'audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
