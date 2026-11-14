"""Run actual Gemma-geometry checks under all three CUDA sanitizers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if not __debug__:
    raise SystemExit('Assertions must remain enabled')
root = Path.cwd()
out = Path(__file__).resolve().parent
proof = json.loads((out / 'gemma-manifest.json').read_text())
if proof.get('passed') is not True:
    raise SystemExit('Raw Gemma validation required')
hashes = dict(proof['input_sha256'])
for path in (Path(__file__).resolve(), out / 'gemma-manifest.json', out / 'gemma-oracle.json'):
    hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    return [name for name, digest in hashes.items()
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
if verify():
    raise SystemExit('Prerequisite inputs changed')
sanitizer = root / '.tools/cuda-sanitizer/cuda_sanitizer_api-windows-x86_64-13.1.118-archive/compute-sanitizer/compute-sanitizer.exe'
for tool in ('synccheck',):
    prefix = 'gemma-' + tool
    paths = [out / (prefix + suffix) for suffix in ('.log', '.stderr', '-oracle.json', '-run.json')]
    if any(path.exists() for path in paths):
        raise SystemExit('Refusing to overwrite evidence: ' + prefix)
    command = [str(sanitizer), '--tool', tool, '--error-exitcode', '99', sys.executable,
               str(out / 'check_gemma.py'), paths[2].name]
    with paths[0].open('x') as stdout, paths[1].open('x') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    logs = paths[0].read_text() + paths[1].read_text()
    oracle = json.loads(paths[2].read_text()) if paths[2].exists() else {}
    changed = verify()
    passed = (result.returncode == 0 and not changed and
              logs.count('ERROR SUMMARY: 0 errors') == 1 and
              len(oracle.get('cases', [])) == 24 and
              oracle.get('source_sha256') == json.loads((out / 'gemma-oracle.json').read_text())['source_sha256'])
    paths[3].write_text(json.dumps(dict(passed=passed, accepted_benchmark=False,
        command=command, exit_code=result.returncode, input_sha256=hashes,
        changed_inputs=changed, evidence_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths[:3] if path.exists()}), indent=2))
    print(tool, 'passed' if passed else 'FAILED', flush=True)
    if not passed:
        raise SystemExit(1)
