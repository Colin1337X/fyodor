"""Retain same-query raw and formatted PDH data; never accept performance."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
MONITOR = HERE.parent / 'gpu-monitor-20261114'
sys.path.insert(0, str(MONITOR))
from pdh_counter import Counter, Item

class FileTime(C.Structure):
    _fields_ = [('low', C.c_uint32), ('high', C.c_uint32)]

class Raw(C.Structure):
    _fields_ = [('status', C.c_uint32), ('timestamp', FileTime),
                ('first', C.c_int64), ('second', C.c_int64), ('multi', C.c_uint32)]

class RawItem(C.Structure):
    _fields_ = [('name', C.c_void_p), ('value', Raw)]


def read_array(counter, raw):
    fn = counter.lib.PdhGetRawCounterArrayW if raw else counter.lib.PdhGetFormattedCounterArrayW
    kind = RawItem if raw else Item
    calls = []
    for attempt in range(4):
        size, count = C.c_uint32(), C.c_uint32()
        def call(data):
            args = [counter.counter] + ([] if raw else [0x8200])
            status = fn(*args, C.byref(size), C.byref(count), data)
            calls.append(dict(phase='size' if data is None else 'data', attempt=attempt,
                              status=status, size=size.value, count=count.value))
            return status
        status = call(None)
        if status not in (0, 0x800007d2) or not size.value:
            return dict(calls=calls, rows=None)
        data = C.create_string_buffer(size.value)
        status = call(data)
        if status == 0x800007d2:
            continue
        if status:
            return dict(calls=calls, rows=None)
        if count.value * C.sizeof(kind) > len(data):
            raise ValueError('Counter count exceeds allocation')
        rows = []
        items = C.cast(data, C.POINTER(kind))
        for i in range(count.value):
            entry = items[i]
            if raw:
                # Bound provider string reads to the returned allocation.
                address = entry.name
                begin, end = C.addressof(data), C.addressof(data) + len(data)
                if not address or not begin <= address < end or (address-begin) % 2:
                    raise ValueError('Raw instance pointer outside allocation')
                units = (end-address)//C.sizeof(C.c_wchar)
                name = C.wstring_at(address, units).split('\0', 1)
                if len(name) != 2:
                    raise ValueError('Unterminated raw instance name')
                v = entry.value
                rows.append(dict(instance=name[0].lower(), status=v.status,
                    timestamp=(v.timestamp.high << 32) | v.timestamp.low,
                    first=v.first, second=v.second, multi=v.multi))
            else:
                v = entry.formatted
                # Store exact bits even for invalid/NaN values, never reinterpret as zero.
                rows.append(dict(instance=entry.name.lower(), status=v.status,
                    value_bits=C.string_at(C.addressof(v.value), C.sizeof(v.value)).hex()))
        return dict(calls=calls, rows=rows)
    return dict(calls=calls, rows=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=900)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if not 2 <= args.samples <= 3600 or not args.label.isascii() or not args.label.isalnum():
        raise SystemExit('Invalid sample count or label')
    folder = HERE / args.label
    folder.mkdir()  # Refuse overwrite.
    inputs = [Path(__file__), MONITOR/'pdh_counter.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    result = dict(accepted_benchmark=False, errors=[], formatted_failure=False, samples=0,
                  sha256=hashes, limitation='Diagnostic during sanitizer; no timing acceptance or PID attribution.')
    counter = None
    try:
        counter = Counter()
        fn = counter.lib.PdhGetRawCounterArrayW
        fn.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32), C.c_void_p]
        fn.restype = C.c_uint32
        with (folder/'samples.jsonl').open('x') as output:
            for index in range(args.samples):
                time.sleep(1)
                start = time.perf_counter()
                status = counter.lib.PdhCollectQueryData(counter.query)
                row = dict(index=index, qpc_start=start, collect_status=status)
                if not status:
                    row['raw'] = read_array(counter, True)
                    row['formatted'] = read_array(counter, False)
                row['qpc_end'] = time.perf_counter()
                output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush()
                result['samples'] += 1
                if status or row['formatted']['rows'] is None:
                    result['formatted_failure'] = not status
                    break
    except BaseException as error:
        result['errors'].append(str(error) or type(error).__name__)
        raise
    finally:
        if counter is not None:
            try: counter.close()
            except BaseException as error: result['errors'].append(str(error))
        result['inputs_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in inputs)
        log = folder/'samples.jsonl'
        if log.exists(): result['sha256'][str(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
        (folder/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'}))

if __name__ == '__main__':
    main()
