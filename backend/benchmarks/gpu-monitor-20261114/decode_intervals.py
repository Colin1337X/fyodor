"""Read complete sequential generation intervals; never accept benchmarks."""
import math
import re

PROFILE = re.compile(r'Generation profile: mode=(\S+) prompt=(\d+) generated=(\d+) decode_loop_ms=(\S+)')
INTERVAL = re.compile(r'Generation interval: mode=(\S+) prompt=(\d+) generated=(\d+) clock=(\S+) start_seconds=(\S+) end_seconds=(\S+) valid=(\d+)')


def parse(text, *, clock):
    if clock not in ('qpc', 'monotonic'):
        raise ValueError('Unknown clock')
    rows, pending = [], None
    for line in text.splitlines():
        if line.startswith('Generation profile:'):
            match = PROFILE.fullmatch(line)
            if not match or pending is not None:
                raise ValueError('Malformed or unpaired duration')
            pending = (match[1], int(match[2]), int(match[3]), float(match[4]))
        elif line.startswith('Generation interval:'):
            match = INTERVAL.fullmatch(line)
            if not match or pending is None:
                raise ValueError('Malformed or unpaired interval')
            mode, prompt, generated, milliseconds = pending
            start, end = float(match[5]), float(match[6])
            if (match[1], int(match[2]), int(match[3])) != (mode, prompt, generated):
                raise ValueError('Duration and interval identify different calls')
            if match[4] != clock or match[7] != '1':
                raise ValueError('Wrong or failed clock')
            if not all(math.isfinite(v) for v in (milliseconds, start, end)) or start <= 0 or end <= start:
                raise ValueError('Invalid measured interval')
            # Both lines derive from identical clock reads, printed to different
            # units/precision. Allow only decimal-format rounding, not drift.
            if abs((end - start) * 1000 - milliseconds) > 0.000003:
                raise ValueError('Duration does not match endpoints')
            if rows and start < rows[-1]['end']:
                raise ValueError('Overlapping or reordered sequential calls')
            rows.append(dict(mode=mode, prompt=prompt, generated=generated,
                             clock=clock, start=start, end=end, milliseconds=milliseconds))
            pending = None
    if pending is not None or not rows:
        raise ValueError('Incomplete timing evidence')
    return rows
