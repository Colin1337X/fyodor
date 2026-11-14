"""Validate retained runs using each sanitizer's actual summary format."""
import hashlib
import json
from pathlib import Path
import re

def summary_valid(tool, logs):
    summaries = re.findall(r'^========= (?:ERROR|RACECHECK) SUMMARY:.*$', logs, re.MULTILINE)
    expected = ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                if tool == 'racecheck' else '========= ERROR SUMMARY: 0 errors')
    return summaries == [expected]

def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    inputs = {str(Path(__file__).resolve().relative_to(root)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    checks = []
    expected_source = json.loads((out / 'gemma-oracle.json').read_text())['source_sha256']
    for tool in ('memcheck', 'racecheck', 'synccheck'):
        path = out / ('gemma-' + tool + '-run.json')
        proof = json.loads(path.read_text())
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs.update(proof['input_sha256'])
        for name, digest in proof['evidence_sha256'].items():
            inputs[str((out / name).relative_to(root))] = digest
        logs = (out / ('gemma-' + tool + '.log')).read_text() + (out / ('gemma-' + tool + '.stderr')).read_text()
        oracle = json.loads((out / ('gemma-' + tool + '-oracle.json')).read_text())
        expected = {(w, p, b, r) for w in (256,512) for p in (256,8192) for b in (1,2,4) for r in (False,True)}
        actual = [(c['width'],c['position'],c['batch'],c['ring']) for c in oracle['cases']]
        passed = (proof['exit_code'] == 0 and not proof['changed_inputs'] and summary_valid(tool, logs)
                  and len(actual) == 24 and set(actual) == expected and oracle['source_sha256'] == expected_source
                  and all(c['serial_split_exact'] and c['float64_scaled_error'] <= .0005
                          and c['unsplit_scaled_error'] <= .0005 for c in oracle['cases']))
        if not passed:
            raise SystemExit('Invalid evidence: ' + tool)
        checks.append(dict(tool=tool, passed=True, original_wrapper_passed=proof['passed']))
    for name, digest in inputs.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise SystemExit('Changed evidence: ' + name)
    with (out / 'sanitizers-validated.json').open('x') as output:
        json.dump(dict(passed=True, accepted_benchmark=False, checks=checks, input_sha256=inputs), output, indent=2)
    print('All three retained sanitizer runs validated')

if __name__ == '__main__':
    main()
