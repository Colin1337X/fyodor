"""Build a separately named phase-logged lifecycle; never alter active inputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

root = Path.cwd()
out = Path(__file__).resolve().parent
base = root / 'build-cuda/model-attention-query-cache-v2-20261115'
work = root / 'build-cuda/lifecycle-verbose-v2-20261115'
proof_path = root / 'backend/benchmarks/model-attention-query-cache-20261115/build.json'
proof = json.loads(proof_path.read_text())
if work.exists():
    raise SystemExit('Refusing to overwrite evidence')
assert proof['build_passed'] is True
for name, expected in proof['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
source = (base / 'lifecycle.c').read_text()
helper = '''#include <time.h>
static void lifecycle_phase(unsigned line, const char *operation, const char *event)
{
    fprintf(stderr, "LIFECYCLE unix_seconds=%.0f source_line=%u operation=%s event=%s\\n",
        (double)time(NULL),
        line, operation, event);
    fflush(stderr);
}
'''
lines = []
stages = []
for number, line in enumerate(source.splitlines(), 1):
    match = re.search(r'\b(nya_llm_(?:load|session_(?:create|run|verify|reset|rewind|free)))\(', line)
    # No logging inside numerical comparison loops; wrap the existing statement
    # unchanged, including inline if/for statements and their original semantics.
    if match:
        operation = match[1]
        stages.append(dict(source_line=number, operation=operation, statement=line.strip()))
        lines.append(f'    lifecycle_phase({number}, "{operation}", "begin");')
    lines.append(line)
    if match and 'return result;' not in line:
        lines.append(f'    lifecycle_phase({number}, "{operation}", "end");')
instrumented = '\n'.join(lines).replace('int main(', helper + '\nint main(', 1) + '\n'
work.mkdir()
(work / 'lifecycle.c').write_text(instrumented)
env = dict(os.environ, PATH=str(root / '.tools/w64devkit/bin') + os.pathsep + os.environ['PATH'])
commands = []
for original in proof['commands']:
    if str(base / 'lifecycle.c') not in original:
        continue
    command = [str(work / 'lifecycle.c') if arg == str(base / 'lifecycle.c') else arg for arg in original]
    command[-1] = str(work / Path(command[-1]).name)
    subprocess.run(command, env=env, check=True)
    commands.append(command)
assert len(commands) == 2
for name, expected in proof['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
paths = [Path(__file__).resolve(), proof_path, *work.iterdir()]
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out / 'build.json').write_text(json.dumps(dict(build_passed=True, gpu_validated=False,
    accepted_benchmark=False, commands=commands, stages=stages, sha256=hashes,
    note='Phase logs are for diagnostics, not performance timing; existing active binaries unchanged.'), indent=2))
print('C17/C23 verbose lifecycle built; no GPU execution launched')
