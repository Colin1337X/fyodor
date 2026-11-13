"""Validate the external reference's generation contract and QPC intervals."""
import json
import math


def parse(stdout, expected_ids, bounds):
    lines = stdout.splitlines()
    prompts = [line for line in lines if line.startswith('PROMPT_IDS')]
    if len(prompts) != 1 or prompts[0].split()[0] != 'PROMPT_IDS':
        raise ValueError('Missing or duplicate prompt token record')
    if [int(value) for value in prompts[0].split()[1:]] != expected_ids:
        raise ValueError('Reference prompt IDs differ from verified native IDs')
    rows = [json.loads(line) for line in lines if line.startswith('{')]
    if len(rows) != 3:
        raise ValueError('Incomplete reference rounds')
    previous = bounds[0]
    baseline = None
    for index, row in enumerate(rows):
        if (type(row.get('round')) is not int or row['round'] != index or
            type(row.get('prompt')) is not int or row['prompt'] != len(expected_ids) or
            type(row.get('generated')) is not int or row['generated'] != 32 or row.get('eos') is not False):
            raise ValueError('Reference workload contract mismatch')
        start, end, ms = (row.get(key) for key in ('start', 'end', 'decode_ms'))
        if (not all(type(value) in (int, float) and math.isfinite(value) for value in (start, end, ms)) or
            not 0 < previous <= start < end <= bounds[1] or ms <= 0 or abs((end-start)*1000-ms) > .000003):
            raise ValueError('Invalid or unbounded reference interval')
        ids = row.get('ids')
        if not isinstance(ids, list) or len(ids) != 32 or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Invalid generated token IDs')
        raw = bytes.fromhex(row['text_hex'])
        raw.decode('utf-8')
        if not raw or len(raw) > 8192:
            raise ValueError('Invalid generated text size')
        signature = (ids, raw)
        if baseline is not None and signature != baseline:
            raise ValueError('Greedy output changed across reference rounds')
        baseline = signature
        previous = end
    return rows
