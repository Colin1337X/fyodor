"""Verify preserved measurements and final working-tree validation evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

validation = json.loads((out / 'validation.json').read_text())
assert len(validation) == 15 and all(r['exit_code'] == 0 for r in validation)
sources = json.loads((out / 'source-hashes.json').read_text())
assert len(sources) == 25
for name, digest in sources.items():
    assert sha(root / name) == digest, name
counts = {}
for name in ('dense-final', 'real-final', 'small-sustained'):
    data = json.loads((out / name / 'metadata.json').read_text())
    contracts = {}
    for run in data['runs']:
        with (out / name / (run['name'] + '.csv')).open(newline='') as stream:
            losses = [row['loss'] for row in csv.DictReader(stream)]
        contract = (run['model_sha256'], run['checkpoint_sha256'], losses)
        previous = contracts.setdefault(run['workload'], contract)
        assert previous == contract, run['name']
    counts[name] = len(data['runs'])
package = json.loads((out / 'desktop-package.json').read_text())
assert sha(Path(package['installer'])) == package['sha256']
assert package['cli_exit_code'] == 0 and not package['installed']
staged_binary = root / 'build-staged-training/build/fyodor-train.exe'
index_binary = subprocess.check_output(['git', 'show', ':frontend/src-tauri/resources/backend/fyodor-train.exe'], cwd=root)
assert hashlib.sha256(index_binary).hexdigest() == sha(staged_binary)
record = {'validation_stages_passed': len(validation), 'current_source_hashes_verified': len(sources),
          'exact_trajectory_process_counts': counts, 'installer_sha256': package['sha256'],
          'staged_cpu_trainer_sha256': sha(staged_binary),
          'scope': 'Working-tree matrix and package include pre-existing provider work. Isolated staged-source tests and CPU trainer exclude those edits.'}
(out / 'audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))
