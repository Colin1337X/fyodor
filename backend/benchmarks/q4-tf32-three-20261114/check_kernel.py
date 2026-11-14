"""Independent numerical admission and guards; never accept timing here."""
import hashlib
import json
from pathlib import Path
import sys

if not __debug__:
    raise SystemExit('Assertions are required')
out = Path(__file__).resolve().parent
root = Path.cwd()
if sys.argv[1:] not in ([], ['--memcheck'], ['--racecheck'], ['--synccheck']):
    raise SystemExit('Expected no arguments or a sanitizer label')
destination = out / ((sys.argv[1][2:] + '-oracle.json') if len(sys.argv) == 2 else 'oracle.json')
if destination.exists():
    raise SystemExit('Refusing to overwrite numerical evidence')
pending = out.parent / 'qkv-input-reuse-20261114/model-memcheck-short/result.json'
proofs = [(pending, 'passed'), (out / 'compile.json', 'compile_passed'),
          (out / 'assembly.json', 'assembled'), (out / 'oracle-control.json', 'passed')]
input_hashes = {}
for path, key in proofs:
    if not path.exists():
        raise SystemExit('Wait for completed prerequisite: ' + str(path))
    proof = json.loads(path.read_text())
    if proof.get(key) is not True:
        raise ValueError('Failed prerequisite: ' + str(path))
    for name, expected in proof['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed prerequisite: ' + name)
        input_hashes[name] = expected
    input_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
sys.path.insert(0, str(out.parent / 'gemma4-qkv-20261029'))
import environment
if any(not environment.desktop_identity_allowed(p) for p in environment.snapshot()['processes']):
    raise SystemExit('Another GPU workload remains')
paths = [root / 'CUDA' / n for n in ('matvec.cu', 'gemm.cu', 'resident.cu')]
paths += [root / 'optcpp/matvec' / n for n in ('small_batch.cu', 'small_batch_reuse.cu',
    'interleaved_reuse.cu', 'interleaved_integer.cu', 'integer_launch.cu',
    'integer_metadata.cu', 'q4_tf32_three.cu')]
paths += [Path(__file__), out / 'common.py', Path(environment.__file__)]
input_hashes.update({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

# All pending-job and evidence checks precede this context-creating import.
from common import *

library = C.CDLL(str(root / 'build-cuda/q4-tf32-three-oracle-20261114/oracle-c17.dll'))
oracle = library.q4_oracle
oracle.argtypes = [P, P, Z, Z, P, P]
oracle.restype = I
random.seed(713)
shapes = [(256, r) for r in (1, 7, 9, 15, 16, 17, 63, 64, 65)]
shapes += [(3840, r) for r in (512, 2048, 4096, 8192, 15360)]
shapes += [(4096, 3840), (8192, 3840), (15360, 3840)]
bits = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
        0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x27ae, 0xa75a]
cases = [(k, n, None, 'random') for k, n in shapes]
cases += [(256, 17, (d, m), 'random') for d in bits for m in bits]
cases += [(256, 17, None, p) for p in ('zero', 'one', 'alternating', 'subnormal', 'large', 'impulse')]
result = dict(passed=False, accepted_benchmark=False, cases=[], rejection_cases=[],
              numerical_contract='absolute error <= 2e-6 * sum(abs(double products)) + 1e-6',
              bitwise_equivalence_required=False, source_sha256=hashlib.sha256(source).hexdigest(),
              options=[v.decode() for v in opts])
try:
    for columns, rows, edge, pattern in cases:
        stride = columns // 256 * 144
        data = bytearray(random.randbytes(rows * stride))
        for offset in range(0, len(data), 144):
            data[offset:offset + 4] = struct.pack('<HH', *edge) if edge else struct.pack('<ee', .03, -.004)
        values = [random.uniform(-1, 1) for _ in range(columns * 4)]
        if pattern != 'random':
            values = [0.0 if pattern == 'zero' else 1.0 if pattern == 'one' else
                      (-1.0 if j % 2 else 1.0) if pattern == 'alternating' else
                      (-1.0 if j % 2 else 1.0) * 2.0**-140 if pattern == 'subnormal' else
                      values[j] * 2.0**60 if pattern == 'large' else
                      float(j % columns == (j // columns) * 37) for j in range(columns * 4)]
        x_host = array.array('f', values).tobytes()
        weights_host = C.create_string_buffer(bytes(data))
        inputs_host = C.create_string_buffer(x_host)
        expected, bounds = (C.c_double * (4 * rows))(), (C.c_double * (4 * rows))()
        if oracle(weights_host, inputs_host, columns, rows, expected, bounds):
            raise ValueError('Independent oracle rejected fixture')
        weights, x = buffer(data), buffer(x_host)
        packed_bytes = array.array('f', [12345.] * (columns * 4 + 8)).tobytes()
        packed = buffer(packed_bytes)
        call('nya_interleave_four', (columns + 255) // 256, 1, [Q(x), Q(packed + 16), Q(columns)])
        check(sync())
        input_values = array.array('f'); input_values.frombytes(x_host)
        packed_expected = array.array('f', [12345.] * 4 +
            [input_values[t * columns + j] for j in range(columns) for t in range(4)] +
            [12345.] * 4).tobytes()
        assert read(packed, len(packed_bytes)) == packed_expected, 'Interleave or packed guards failed'
        guard = array.array('f', [12345.] * (rows * 4 + 2)).tobytes()
        fractions = {}
        for name in ('nya_small_batch_12', 'nya_q4_tf32_three'):
            output = buffer(guard)
            native = name == 'nya_small_batch_12'
            tile = 8 if native else 64
            call(name, (rows + tile - 1) // tile, 4 if native else 1,
                 [Q(weights), Q(x if native else packed + 16), Q(output + 4),
                  Q(columns), Q(stride), U(rows), U(4)])
            check(sync())
            raw = read(output, len(guard))
            assert raw[:4] == guard[:4] and raw[-4:] == guard[-4:], 'Output guard modified'
            actual = array.array('f'); actual.frombytes(raw[4:-4])
            ratio = 0.0
            for j, value in enumerate(actual):
                assert math.isfinite(value), (name, columns, rows, edge, pattern, j, value)
                fraction = abs(value - expected[j]) / bounds[j]
                assert fraction <= 1.0, (name, columns, rows, edge, pattern, j, value, expected[j], fraction)
                ratio = max(ratio, fraction)
            fractions[name] = ratio
        assert read(weights, len(data)) == data, 'Weights modified'
        assert read(x, len(x_host)) == x_host, 'Row-major input modified'
        assert read(packed, len(packed_bytes)) == packed_expected, 'Packed input modified'
        result['cases'].append(dict(columns=columns, rows=rows, half_bits=edge, pattern=pattern,
            guards_clear=True, maximum_fraction_of_float64_bound=fractions))
        for address in owned: check(free(address))
        owned.clear()
        if edge is None or len(result['cases']) % 20 == 0:
            print('Passed', len(result['cases']), 'of', len(cases), flush=True)
    for columns, batch in ((0, 4), (1, 4), (255, 4), (257, 4), (256, 0), (256, 1), (256, 2), (256, 8)):
        guard = array.array('f', [12345.] * 6).tobytes()
        output = buffer(guard)
        call('nya_q4_tf32_three', 1, 1, [Q(0), Q(0), Q(output + 4), Q(columns), Q(144), U(1), U(batch)])
        check(sync())
        assert read(output, len(guard)) == guard, 'Unsupported shape touched output'
        result['rejection_cases'].append(dict(columns=columns, batch=batch, output_unchanged=True))
        for address in owned: check(free(address))
        owned.clear()
    for name, expected_hash in input_hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash, name
    result['passed'] = True
except BaseException as error:
    result['failure'] = str(error) or type(error).__name__
    raise
finally:
    for address in owned: check(free(address))
    check(unload(module)); check(release(device))
    result['input_sha256'] = input_hashes
    with destination.open('x') as output: json.dump(result, output, indent=2)
