"""Read raw and formatted PDH values from one collection; no acceptance policy."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import sys
import time

here = Path(__file__).resolve().parent
dependency = here.parent / 'gpu-monitor-20261114/pdh_counter.py'
sys.path.insert(0, str(dependency.parent))
from pdh_counter import Counter


class FileTime(C.Structure):
    _fields_ = [('low', C.c_uint32), ('high', C.c_uint32)]


class Raw(C.Structure):
    _fields_ = [('status', C.c_uint32), ('timestamp', FileTime),
                ('first', C.c_int64), ('second', C.c_int64), ('multi', C.c_uint32)]


class Item(C.Structure):
    _fields_ = [('name', C.c_wchar_p), ('raw', Raw)]


def raw_values(query):
    fn = query.lib.PdhGetRawCounterArrayW
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32), C.c_void_p]
    fn.restype = C.c_uint32
    for _ in range(4):
        size, count = C.c_uint32(), C.c_uint32()
        status = fn(query.counter, C.byref(size), C.byref(count), None)
        if status != 0x800007d2:
            query.check(status, 'raw array sizing')
        if not size.value:
            raise RuntimeError('Empty raw array')
        data = C.create_string_buffer(size.value)
        status = fn(query.counter, C.byref(size), C.byref(count), data)
        if status == 0x800007d2:
            continue
        query.check(status, 'raw array data')
        if count.value * C.sizeof(Item) > len(data):
            raise RuntimeError('Raw array count exceeds allocation')
        items = C.cast(data, C.POINTER(Item))
        result = {}
        for i in range(count.value):
            name, raw = items[i].name.lower(), items[i].raw
            if name in result:
                raise RuntimeError('Duplicate raw instance')
            result[name] = dict(status=raw.status, first=raw.first, second=raw.second,
                timestamp=(raw.timestamp.high << 32) | raw.timestamp.low, multi=raw.multi)
        return result
    raise RuntimeError('Raw array repeatedly changed size')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--luid', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    luid = args.luid.lower()
    if not re.fullmatch(r'luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}', luid) or args.pid <= 0 or not 2 <= args.samples <= 120:
        raise SystemExit('Invalid diagnostic arguments')
    if C.sizeof(C.c_void_p) != 8 or (C.sizeof(Raw), C.sizeof(Item), Raw.first.offset) != (40, 48, 16):
        raise SystemExit('Unsupported Windows x64 ABI')
    paths = (Path(__file__), dependency)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with args.output.open('x') as output:
        def emit(row):
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
        query = None
        failed = False
        try:
            query = Counter()
            emit(dict(kind='header', accepted_benchmark=False, sha256=hashes,
                      pid=args.pid, luid=luid, samples=args.samples))
            for index in range(args.samples):
                time.sleep(1)
                before = time.perf_counter()
                formatted = query.collect()
                raw = raw_values(query)  # No intervening collection.
                selected = []
                for counter in formatted:
                    match = re.match(r'^pid_(\d+)_' + re.escape(luid) + '_', counter['instance'])
                    if match and int(match[1]) in (4, args.pid):
                        selected.append(dict(formatted=counter, raw=raw.get(counter['instance'])))
                emit(dict(kind='sample', index=index, before=before,
                          after=time.perf_counter(), counters=selected))
        except BaseException as error:
            failed = True
            emit(dict(kind='error', message=str(error)))
            raise
        finally:
            if query is not None:
                query.close()
            changed = [str(p) for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]]
            emit(dict(kind='end', failed=failed, changed_inputs=changed))
            if changed:
                raise RuntimeError('Diagnostic inputs changed')


if __name__ == '__main__':
    main()
