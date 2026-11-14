"""Summarize completed generation diagnostics without accepting new scores."""
import hashlib
import json
import math
from pathlib import Path
import statistics

MODES = ('ordinary', 'mtp-serial2', 'mtp-batch2', 'mtp-batch4')
MODE_SETS = {'baseline': MODES,
             'joint-base': ('ordinary', 'mtp-serial2', 'mtp-batch2', 'mtp-joint3')}
ORDER = [(r, mode) for r in range(3) for mode in (reversed(MODES) if r == 1 else MODES)]


def native_text(stdout):
    if stdout.count('\nTEXT ') != 1:
        raise ValueError('Missing or ambiguous native text record')
    text, separator, _ = stdout.split('\nTEXT ', 1)[1].partition('\n{"round":')
    if not separator or not text:
        raise ValueError('Incomplete native text record')
    return text.encode('utf-8')


def statistics_ms(values):
    if len(values) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Expected two positive finite warmed observations')
    return dict(observations_ms=values, mean_ms=statistics.mean(values), stdev_ms=statistics.stdev(values))


def compare(native, reference, text, mode_set='baseline'):
    if mode_set not in MODE_SETS:
        raise ValueError('Unsupported generation mode set')
    modes = MODE_SETS[mode_set]
    order = [(r, mode) for r in range(3) for mode in (reversed(modes) if r == 1 else modes)]
    if [(row['round'], row['mode']) for row in native] != order:
        raise ValueError('Native round/mode inventory differs')
    if [row['round'] for row in reference] != [0, 1, 2]:
        raise ValueError('Reference round inventory differs')
    prompts = {row['interval']['prompt'] for row in native} | {row['prompt'] for row in reference}
    if len(prompts) != 1:
        raise ValueError('Cross-engine prompt count differs')
    if any(row['interval']['generated'] != 32 for row in native) or any(
        row['generated'] != 32 or row['eos'] is not False for row in reference):
        raise ValueError('Cross-engine generated count or stop differs')
    if any(bytes.fromhex(row['text_hex']) != text for row in reference):
        raise ValueError('Cross-engine greedy output bytes differ')
    llama = statistics_ms([row['decode_ms'] for row in reference if row['round'] in (1, 2)])
    cells = {}
    for mode in modes:
        cell = statistics_ms([row['interval']['milliseconds'] for row in native if row['mode'] == mode and row['round'] in (1, 2)])
        cell['diagnostic_speed_percent_of_llama'] = 100 * llama['mean_ms'] / cell['mean_ms']
        cells[mode] = cell
    return dict(mode_set=mode_set, prompt_tokens=next(iter(prompts)), output_bytes_equal=True,
                output_sha256=hashlib.sha256(text).hexdigest(), llama=llama, fyodor=cells)


def main():
    out = Path(__file__).resolve().parent
    destination = out / 'generation-comparison.json'
    if destination.exists():
        raise SystemExit('Refusing to overwrite comparison evidence')
    folders = [out / name for name in ('monitored-generation', 'monitored-llama-generation')]
    if any(not (folder / 'gpu-coverage.json').exists() for folder in folders):
        raise SystemExit('Both completed generation audits are required')
    result = dict(accepted_benchmark=False, contexts={}, errors=[], sha256={},
                  contract='32 generated outputs including the first sample from prefill logits; two warmed observations per cell',
                  limitations=['Diagnostic only; fixed-forward accepted scores are unchanged.',
                    'Only two warmed observations; external engine order has not been reversed.',
                    'GPU sampled coverage and legacy idle gates are retained, not waived.',
                    'CPU/thermal equivalence and accounting differences require separate review.'])
    try:
        audits = []
        for folder in folders:
            path = folder / 'gpu-coverage.json'
            result['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            audit = json.loads(path.read_text())
            if audit['errors']:
                raise ValueError(f'Incomplete audit: {folder.name}')
            for name, expected in audit['sha256'].items():
                source = Path(name) if Path(name).is_absolute() else folder / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Changed audit input: {source}')
            if [run['context'] for run in audit['runs']] != ['short', 'long']:
                raise ValueError('Both context audits are required in order')
            audits.append(audit)
        result['sampled_coverage_clear'] = all(audit['sampled_coverage_clear'] is True for audit in audits)
        result['legacy_idle_gates'] = [json.loads((folder / 'timing.json').read_text())['gate_passed'] for folder in folders]
        for native, reference in zip(audits[0]['runs'], audits[1]['runs']):
            path = folders[0] / (native['context'] + '.stdout')
            result['contexts'][native['context']] = compare(native['observations'], reference['observations'], native_text(path.read_text()))
    except (OSError, KeyError, TypeError, ValueError) as error:
        result['errors'].append(str(error))
    with destination.open('x') as output:
        json.dump(result, output, indent=2)
    if result['errors']:
        raise SystemExit('Comparison rejected; see retained evidence')
    print(json.dumps(result['contexts'], indent=2))


if __name__ == '__main__':
    main()
