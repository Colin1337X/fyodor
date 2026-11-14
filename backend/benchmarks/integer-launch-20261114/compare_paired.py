"""Aggregate an ABBA sequence of completed, clean generation diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

here = Path(__file__).resolve().parent
original_import_path = sys.path[:]
sys.path.insert(0, str(here.parent / 'mtp-joint-base-v2-20261114'))
from compare_monitored import load_audit
from compare_generation import compare, native_text, MODE_SETS
sys.path.insert(0, str(here.parent / 'gpu-monitor-20261114'))
from audit_generation import lifetime
sys.path[:] = original_import_path


def aggregate(cells):
    if len(cells) != 4 or [c['engine'] for c in cells] != ['fyodor','llama','llama','fyodor']:
        raise ValueError('Expected Fyodor/llama/llama/Fyodor order')
    spans = [lifetime(c['timing']['runs'][0])[1] for c in cells]
    if any(a[1] > b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError('External engine order overlaps or differs from ABBA')
    if len({c['audit']['runs'][0]['context'] for c in cells}) != 1:
        raise ValueError('Cross-engine context differs')
    if any(c['identity'] != cells[0]['identity'] for c in cells):
        raise ValueError('External model identities differ')
    for a,b in ((cells[0],cells[3]),(cells[1],cells[2])):
        if a['build'] != b['build']:
            raise ValueError('Engine build or configuration changed between repetitions')
    native = [cells[i]['audit']['runs'][0]['observations'] for i in (0,3)]
    references = [cells[i]['audit']['runs'][0]['observations'] for i in (1,2)]
    texts = [cells[i]['text'] for i in (0,3)]
    if texts[0] != texts[1]:
        raise ValueError('Native output differs between repetitions')
    # Reuse the existing full round, mode, token-count and greedy-output checks.
    pairs = [compare(n,r,t,mode_set='joint-base') for n,r,t in zip(native,references,texts)]
    def summary(values):
        return dict(observations_ms=values, mean_ms=statistics.mean(values),
                    sample_stddev_ms=statistics.stdev(values))
    llama = summary([v for pair in pairs for v in pair['llama']['observations_ms']])
    modes = {}
    for mode in MODE_SETS['joint-base']:
        item = summary([v for pair in pairs for v in pair['fyodor'][mode]['observations_ms']])
        item['diagnostic_speed_percent_of_llama'] = 100 * llama['mean_ms'] / item['mean_ms']
        modes[mode] = item
    return dict(context=cells[0]['audit']['runs'][0]['context'], external_order=['fyodor','llama','llama','fyodor'],
                prompt_tokens=pairs[0]['prompt_tokens'], generated_outputs=32,
                warmed_observations_per_engine_mode=4, llama=llama, fyodor=modes)


def load(folder, engine):
    audit = load_audit(folder)
    timing = json.loads((folder/'timing.json').read_text())
    if len(timing['runs']) != 1:
        raise ValueError('Expected one process per recorded run')
    if engine == 'fyodor':
        result = json.loads((folder/'result.json').read_text())
        if (audit.get('mode_set') != 'joint-base' or result.get('inputs_unchanged') is not True or
            result.get('output_parity') is not True or result.get('runner_exit') != 0 or
            result.get('joint_rounds') != 3 or len(result.get('integer_calls',[])) != 3 or
            min(result['integer_calls']) <= 0):
            raise ValueError('Native execution or input-integrity proof failed')
        for name, digest in result['sha256'].items():
            if hashlib.sha256((Path.cwd()/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Changed native input: {name}')
        identity = result['model_identity']['models']
        build = dict(binary=timing['binary_sha256'], environment=timing['environment'])
        text = native_text((folder/(audit['runs'][0]['context']+'.stdout')).read_text())
    else:
        identity = timing['model_identity']['models']
        build = dict(manifest=timing['build_manifest_sha256'], libraries=timing['library_sha256'])
        text = None
    # Modification time is not model identity; byte length and digest are.
    identity = {name:dict(size=value['size'],sha256=value['sha256']) for name,value in identity.items()}
    if not identity:
        raise ValueError('Missing model identity')
    return dict(engine=engine,audit=audit,timing=timing,identity=identity,build=build,text=text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folders',nargs=4,type=Path,help='Fyodor A, llama B, llama C, Fyodor D')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite comparison')
    result=dict(accepted_benchmark=False,errors=[],sha256={},limitations=[
        'Generation diagnostic, not replacement of fixed-forward accepted scores.',
        'Four warmed samples; CPU/thermal equivalence and accounting need separate review.',
        '32 generated outputs include the first sample from prefill logits.'])
    try:
        cells=[load(folder,engine) for folder,engine in zip(args.folders,('fyodor','llama','llama','fyodor'))]
        result['comparison']=aggregate(cells)
        paths=[Path(__file__).resolve(),
               here.parent/'mtp-joint-base-v2-20261114/compare_monitored.py',
               here.parent/'aligned-prefix-runtime-20261114/compare_generation.py',
               here.parent/'gpu-monitor-20261114/audit_generation.py']
        for folder in args.folders:
            paths.extend(p for p in folder.iterdir() if p.is_file())
        result['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    except (OSError,KeyError,TypeError,ValueError) as error:
        result['errors'].append(str(error))
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    if result['errors']:
        raise SystemExit('Paired comparison rejected; no ratios emitted')


if __name__=='__main__':
    main()
