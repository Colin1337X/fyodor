"""Validate the complete boundary sanitizer matrix before new GPU work."""
import hashlib
import itertools
import json
from pathlib import Path
from run_boundary_memcheck import validate


def audit(root, out):
    hashes = {}
    cases = []
    for standard, width, ring in itertools.product(('c17', 'c23'), ('256', '512'), ('0', '1')):
        label = f'{standard}-{width}-ring{ring}'
        folder = out / f'boundary-memcheck-{label}'
        path = folder / 'result.json'
        if not path.exists():
            raise ValueError('Pending boundary sanitizer: ' + label)
        result = json.loads(path.read_text())
        if result.get('passed') is not True or result.get('accepted_benchmark') is not False or result.get('errors') != []:
            raise ValueError('Failed boundary terminal result: ' + label)
        command = result['command']
        if command[-2:] != [width, ring] or Path(command[5]).name != f'boundaries-{standard}.exe':
            raise ValueError('Wrong boundary executable/configuration: ' + label)
        validate(result['exit_code'], (folder / 'stdout').read_text(), (folder / 'stderr').read_text())
        for required in (folder / 'stdout', folder / 'stderr', out / 'run_boundary_memcheck.py'):
            if str(required.relative_to(root)) not in result['sha256']:
                raise ValueError('Missing evidence hash: ' + str(required))
        for name, expected in result['sha256'].items():
            if name not in hashes:
                hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if hashes[name] != expected:
                raise ValueError('Changed boundary evidence: ' + name)
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases.append(label)
    for path in (Path(__file__).resolve(), out / 'run_boundary_memcheck.py'):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(passed=True, accepted_benchmark=False, configurations=cases,
                parity_cases=42 * len(cases), sha256=hashes)


if __name__ == '__main__':
    out = Path(__file__).resolve().parent
    destination = out / 'boundary-matrix-audit.json'
    if destination.exists():
        raise SystemExit('Refusing to overwrite boundary audit')
    result = audit(Path.cwd(), out)
    with destination.open('x') as output:
        json.dump(result, output, indent=2)
    print(f"{result['parity_cases']} sanitized boundary cases; {len(result['sha256'])} hashes verified")
