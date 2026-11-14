"""Compare real model prompt token IDs with CPU-only tokenizer executables."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

if not __debug__: raise SystemExit('Assertions are required')
root = Path.cwd()
out = Path(__file__).resolve().parent
base = out.parent / 'aligned-prefix-runtime-20261114'
evidence = out / 'prompt-token-parity'
if evidence.exists():
    raise SystemExit('Refusing to overwrite tokenizer evidence')
evidence.mkdir()
work = root / 'build-cuda/batch-graph-tokenizers-20261114'
if work.exists(): raise SystemExit('Refusing to overwrite tokenizer build')
work.mkdir()
upstream = root / '.tools/reference-llama-b10809'
libraries = root / '.tools/llama-b10809-cpu'
compiler = root / '.tools/w64devkit/bin/gcc.exe'
source = base / 'tokenize_prompt.c'
env = {key: value for key, value in os.environ.items() if not key.startswith('NYA_')}
env.update(NYA_COMPUTE_BACKEND='cpu', PYTHONDONTWRITEBYTECODE='1')
env['PATH'] = str(compiler.parent) + os.pathsep + str(libraries) + os.pathsep + env['PATH']
cache = (root / 'build-cpu/CMakeCache.txt').read_text()
assert 'NYA_ENABLE_CUDA:BOOL=OFF' in cache
commands = []
inputs = [source, Path(__file__), root / 'build-cpu/libnya-engine.a', libraries / 'llama.dll',
          libraries / 'ggml.dll', libraries / 'ggml-base.dll', base / 'generation.c', base / 'llama_generation.c']
for implementation in ('fyodor', 'llama'):
    for standard in ('c17', 'c23'):
        exe = work / f'tokenize-{implementation}-{standard}.exe'
        command = [str(compiler), f'-std={standard}', '-O3', '-Wall', '-Wextra', '-Wpedantic',
                   '-Wconversion', '-Wshadow', '-Werror']
        if implementation == 'llama':
            command += ['-DREFERENCE_TOKENIZER', '-I', str(upstream / 'include'),
                        '-I', str(upstream / 'ggml/include'), str(source), str(libraries / 'llama.dll')]
        else:
            command += ['-I', 'backend/include', '-I', 'backend/core', str(source),
                        'build-cpu/libnya-engine.a', '-lbcrypt']
        command += ['-o', str(exe)]
        subprocess.run(command, env=env, check=True)
        commands.append(command)
        inputs.append(exe)
# Keep the two benchmark prompt constructors independently visible, and verify
# the exact fragments before using them for this CPU-only comparison.
first = '<|turn>user\n'
last = 'Write a numbered list of twenty concise facts about the Moon.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
for name in ('generation.c', 'llama_generation.c'):
    text = (base / name).read_text()
    for fragment in (first, last, 'Hello. '):
        assert json.dumps(fragment) in text
    assert 'i<4090' in text
metadata = dict(accepted_benchmark=False, commands=commands, runs=[], cpu_only=True,
                model='E:/models/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf')
for context in ('short', 'long'):
    prompt = evidence / f'{context}.txt'
    prompt.write_bytes((first + ('Hello. ' * 4090 if context == 'long' else '') + last).encode())
    outputs = []
    for implementation in ('fyodor', 'llama'):
        command = [str(work / f'tokenize-{implementation}-c17.exe'), metadata['model'], str(prompt)]
        result = subprocess.run(command, env=env, capture_output=True)
        stdout = evidence / f'{context}-{implementation}.stdout'
        stderr = evidence / f'{context}-{implementation}.stderr'
        stdout.write_bytes(result.stdout); stderr.write_bytes(result.stderr)
        metadata['runs'].append(dict(context=context, implementation=implementation, command=command,
                                     exit_code=result.returncode))
        if result.returncode:
            (evidence / 'result.json').write_text(json.dumps(metadata, indent=2))
            raise SystemExit(f'Tokenizer failed: {context} {implementation}')
        ids = [int(token) for token in result.stdout.split()]
        assert ids and all(token >= 0 for token in ids)
        outputs.append(ids)
        inputs += [stdout, stderr]
    metadata[context] = dict(prompt_tokens=len(outputs[0]), identical=outputs[0] == outputs[1])
    inputs.append(prompt)
metadata['sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
(evidence / 'result.json').write_text(json.dumps(metadata, indent=2))
assert all(metadata[context]['identical'] for context in ('short', 'long')), 'Prompt token mismatch'
print(json.dumps({context: metadata[context] for context in ('short', 'long')}))
