"""Check logging transformation and early loader-failure output without GPU work."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
work = root / 'build-cuda/lifecycle-verbose-v2-20261115'
original = (root / 'build-cuda/model-attention-query-cache-v2-20261115/lifecycle.c').read_text()
source = (work / 'lifecycle.c').read_text()
stripped = '\n'.join(line for line in source.splitlines() if not line.startswith('    lifecycle_phase(')) + '\n'
start = stripped.index('#include <time.h>')
end = stripped.index('int main(', start)
assert stripped[:start] + stripped[end:] == original
fixture = out / 'empty.gguf'
fixture.write_bytes(b'')
records = []
for standard in ('c17', 'c23'):
    run = subprocess.run([str(work / f'lifecycle-{standard}.exe'), str(fixture), '8192'], capture_output=True, text=True)
    assert run.returncode == 1, run
    assert 'operation=nya_llm_load event=begin' in run.stderr
    assert 'operation=nya_llm_load event=end' not in run.stderr
    assert 'operation=nya_llm_session_free event=begin' in run.stderr
    (out / f'loader-failure-{standard}.stderr').write_text(run.stderr)
    records.append(dict(standard=standard, exit_code=run.returncode, stderr=run.stderr))
paths = [Path(__file__).resolve(), out / 'build.json', fixture]
(out / 'check.json').write_text(json.dumps(dict(passed=True, numerical_statements_unchanged=True,
    scope='Transformation equivalence and early loader failure only; no model/GPU validation',
    records=records, sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), indent=2))
print('C17/C23 phase logging and unchanged lifecycle statements verified')
