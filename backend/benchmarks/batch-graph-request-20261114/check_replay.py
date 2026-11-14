"""Changing-token CUDA graph parity. Gated before importing CUDA bindings."""
import array
import ctypes as C
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import sys

out = Path(__file__).resolve().parent
root = out.parents[2]


def prerequisites():
    paths = [out.parent/'integer-metadata-20261114/model-memcheck-short/result.json',
             out/'host.json', out/'compile.json']
    inputs = set(paths)
    for path in paths:
        if not path.exists():
            raise RuntimeError(f'Wait for completed prerequisite: {path}')
        proof = json.loads(path.read_text())
        if proof.get('passed', proof.get('compile_passed')) is not True:
            raise RuntimeError(f'Failed prerequisite: {path}')
        if path == paths[0] and proof.get('exit_code') != 0:
            raise RuntimeError('Full-model sanitizer did not exit successfully')
        for name, expected in proof['sha256'].items():
            p = root/name
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Changed prerequisite: {name}')
            inputs.add(p)
    return inputs


def weights(dtype, width):
    rng = random.Random(713+dtype+width)
    if dtype in (0, 1):
        code = 'f' if dtype == 0 else 'e'
        data = b''.join(struct.pack('<'+code, rng.uniform(-2, 2)) for _ in range(17*width))
        return data, width*(4 if dtype == 0 else 2)
    block = 144 if dtype == 12 else 210
    stride = width//256*block
    data = bytearray(rng.randbytes(17*stride))
    for i in range(0, len(data), block):
        if dtype == 12:
            data[i:i+4] = struct.pack('<ee', .03125, .00390625)
        else:
            data[i+208:i+210] = struct.pack('<e', .03125)
    return bytes(data), stride


def main():
    if sys.argv[1:] not in ([], ['--memcheck']):
        raise SystemExit('Expected no arguments or --memcheck')
    destination = out/('memcheck-oracle.json' if sys.argv[1:] else 'oracle.json')
    if destination.exists():
        raise RuntimeError('Refusing to overwrite replay evidence')
    inputs = prerequisites()
    gate_path = out.parent/'gemma4-qkv-20261029/environment.py'
    spec = importlib.util.spec_from_file_location('environment', gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    if any(not gate.desktop_identity_allowed(p) for p in gate.snapshot()['processes']):
        raise RuntimeError('Another GPU workload remains')
    inputs.update([Path(__file__).resolve(), out/'common.py', gate_path,
                   root/'optcpp/graph/batch_request.cu'])
    inputs.update(root/'CUDA'/name for name in ('matvec.cu','gemm.cu','resident.cu'))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    # Only now may the module create a context and compile/load CUDA code.
    import common as g
    P, U, Q = g.P, g.U, g.Q
    bind = lambda name, args: g.bind(g.cu, name, args)
    stream_create = bind('cuStreamCreate', [C.POINTER(P), U])
    stream_destroy = bind('cuStreamDestroy_v2', [P])
    begin = bind('cuStreamBeginCapture_v2', [P, g.I])
    end = bind('cuStreamEndCapture', [P, C.POINTER(P)])
    instantiate = bind('cuGraphInstantiateWithFlags', [C.POINTER(P), P, Q])
    replay = bind('cuGraphLaunch', [P, P])
    graph_destroy = bind('cuGraphDestroy', [P])
    exec_destroy = bind('cuGraphExecDestroy', [P])
    class Packet4(C.Structure):
        _fields_ = [('values', U*4)]
    class Packet512(C.Structure):
        _fields_ = [('values', U*512)]
    result = dict(passed=False, accepted_benchmark=False, cases=[], errors=[],
                  options=[v.decode() for v in g.opts], sha256=hashes)
    stream = P()
    try:
        g.check(stream_create(C.byref(stream), 1))
        for dtype in (0, 1, 12, 14):
            for width in (256, 3840):
                data, stride = weights(dtype, width)
                w = g.buffer(data)
                for count in (2, 4):
                    guard = array.array('f', [12345.]*(width*count+2)).tobytes()
                    actual, reference = g.buffer(guard), g.buffer(guard)
                    request_guard = struct.pack('<7I', *([0xdeadbeef]*7))
                    request = g.buffer(request_guard)
                    g.check(g.sync())  # Order pageable uploads before the nonblocking stream.
                    graph, executable = P(), P()
                    capturing = False
                    try:
                        g.check(begin(stream, 1))
                        capturing = True
                        args = [Q(w), Q(actual+4), Q(width), Q(stride), U(dtype),
                                U(count), C.c_float(.5), Q(request+4)]
                        g.call('nya_gather_graph_batch', (width*count+255)//256, 1, args, stream)
                        status = end(stream, C.byref(graph))
                        capturing = False
                        g.check(status)
                        g.check(instantiate(C.byref(executable), graph, 0))
                        for round_ in range(12):
                            tokens = [(round_*3+q*5)%17 for q in range(4)]
                            position = (0,255,256,8191,8192,0xfffffffb)[round_%6]
                            small, large = Packet4(), Packet512()
                            for q, token in enumerate(tokens):
                                small.values[q] = large.values[q] = token
                            g.call('nya_decode_batch_request', 1, 1,
                                   [Q(request+4), small, U(position)], stream)
                            g.check(replay(executable, stream))
                            g.call('nya_gather_batch', (width*count+255)//256, 1,
                                   [Q(w), Q(reference+4), Q(width), Q(stride), U(dtype),
                                    large, U(count), C.c_float(.5)], stream)
                            g.check(g.sync())
                            actual_bytes = g.read(actual, len(guard))
                            expected = g.read(reference, len(guard))
                            assert actual_bytes == expected
                            assert actual_bytes[:4] == guard[:4] and actual_bytes[-4:] == guard[-4:]
                            values = array.array('f'); values.frombytes(actual_bytes)
                            assert all(math.isfinite(v) for v in values)
                            packet = struct.unpack('<7I', g.read(request, 28))
                            assert packet == (0xdeadbeef,tokens[0],position,*tokens[1:],0xdeadbeef)
                            result['cases'].append(dict(dtype=dtype,width=width,count=count,
                                round=round_,position=position,tokens=tokens,parity=True,guards=True))
                        assert g.read(w, len(data)) == data
                    finally:
                        if capturing:
                            end(stream, C.byref(graph))
                        g.check(g.sync())
                        if executable: g.check(exec_destroy(executable))
                        if graph: g.check(graph_destroy(graph))
                        for address in (request, reference, actual):
                            g.check(g.free(Q(address)))
                            g.owned[:] = [v for v in g.owned if v.value != address]
                g.check(g.free(Q(w)))
                g.owned[:] = [v for v in g.owned if v.value != w]
        assert len(result['cases']) == 192
        for name, expected in hashes.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
        result['passed'] = True
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__)
        raise
    finally:
        try:
            g.check(g.sync())
            if stream: g.check(stream_destroy(stream))
            for address in g.owned: g.check(g.free(address))
            g.check(g.unload(g.module))
            g.check(g.release(g.device))
        except BaseException as error:
            result['passed'] = False
            result['errors'].append('cleanup: '+str(error))
            raise
        finally:
            destination.write_text(json.dumps(result, indent=2))
    print('192 changing-token graph replays passed; no timing acceptance')


if __name__ == '__main__':
    main()
