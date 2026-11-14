"""Independently audit the complete extended recovery matrix without a GPU."""
import hashlib
import itertools
import json
from pathlib import Path
import re

COLD = ('begin', 'end', 'instantiate', 'launch', 'request', 'lowering', 'sync', 'readback')
CACHED = ('request', 'launch', 'sync', 'readback')


def cases():
    for standard, width, ring in itertools.product(('c17', 'c23'), ('256', '512'), ('0', '1')):
        for cache, sites in (('cold', COLD), ('cached', CACHED)):
            for site in sites:
                yield standard, width, ring, site, cache


def validate(result, stdout, stderr, width, ring, site, cache):
    if result.get('passed') is not True or type(result.get('exit_code')) is not int or result['exit_code'] != 0 or result.get('errors') != []:
        raise ValueError('Failed or incomplete terminal result')
    if result.get('accepted_benchmark') is not False:
        raise ValueError('Correctness run cannot accept timing')
    if re.findall(r'^========= ERROR SUMMARY: (\d+) errors\s*$', stdout + '\n' + stderr, re.M) != ['0']:
        raise ValueError('Missing, duplicate or nonzero sanitizer summary')
    if re.findall(r'^INJECTED graph failure: (\w+)$', stderr, re.M) != [site]:
        raise ValueError('Missing, duplicate or wrong injection')
    markers = re.findall(r'^recovery=.*$', stdout, re.M)
    if markers != [f'recovery=1 site={site} cache={cache} width={width} ring={ring}']:
        raise ValueError('Wrong recovery configuration')
    resources = re.findall(r'^GRAPH resources: graphs=(\d+)/(\d+) execs=(\d+)/(\d+) live=(\d+) errors=(\d+)$', stderr, re.M)
    graphs = 2 if cache == 'cached' or site in ('request', 'begin') else 3
    executables = 3 if cache == 'cold' and site in ('launch', 'sync', 'readback') else 2
    expected = tuple(map(str, (graphs, graphs, executables, executables, 0, 0)))
    if resources != [expected]:
        raise ValueError('Unexpected ownership history, leak or accounting error')


def main():
    root = Path.cwd()
    out = Path(__file__).resolve().parent
    report = out / 'recovery-v2-audit.json'
    if report.exists():
        raise SystemExit('Refusing to overwrite audit evidence')
    hashes = {}
    verified = []
    for standard, width, ring, site, cache in cases():
        label = f'{standard}-{width}-ring{ring}-{site}-{cache}'
        folder = out / f'recovery-v2-{label}'
        result_path = folder / 'result.json'
        result = json.loads(result_path.read_text())
        validate(result, (folder / 'stdout').read_text(), (folder / 'stderr').read_text(), width, ring, site, cache)
        if result['command'][-4:] != [width, ring, site, cache]:
            raise ValueError('Command configuration mismatch: ' + label)
        if Path(result['command'][5]).name != f'recovery-{standard}.exe':
            raise ValueError('Wrong language-standard executable: ' + label)
        required = [folder / 'stdout', folder / 'stderr', out / 'run_recovery_v2.py', out / 'recovery-build-v2.json']
        for path in required:
            if str(path.relative_to(root)) not in result['sha256']:
                raise ValueError('Missing evidence hash: ' + str(path))
        for name, expected in result['sha256'].items():
            if name not in hashes:
                hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if hashes[name] != expected:
                raise ValueError('Changed evidence: ' + name)
        hashes[str(result_path.relative_to(root))] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        verified.append(label)
    hashes[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report.write_text(json.dumps(dict(passed=True, accepted_benchmark=False, cases=verified, sha256=hashes), indent=2))
    print(f'{len(verified)} recovery cases and {len(hashes)} hashes verified; no timing acceptance')


if __name__ == '__main__':
    main()
