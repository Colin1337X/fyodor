"""Describe rejected measurements without changing performance acceptance."""
import hashlib
import json
from pathlib import Path
import sys

root=Path.cwd();out=Path(__file__).resolve().parent
destination=out/'diagnostic-summary.json'
if destination.exists():raise SystemExit('Refusing to overwrite diagnostic summary')
reference=out.parent/'aligned-prefix-runtime-20261114'
sys.path.insert(0,str(reference))
from compare_generation import compare,native_text

inputs={Path(__file__).resolve(),reference/'compare_generation.py'}
result=dict(accepted_benchmark=False,target_125_proven=False,contexts={},
    contract='32 generated outputs including the first sample from prefill logits; prefill and teardown excluded',
    limitations=['All runs failed sampled GPU activity coverage; these ratios are descriptive arithmetic, not verified speedups.',
                 'Only two warmed observations per cell; first rounds excluded.',
                 'Cross-engine order was not reversed; CPU/thermal equivalence is not established.',
                 'Generation-loop measurements do not replace the historical fixed-forward scorecard.'])
for context,label in (('short','B'),('long','A')):
    native=out/f'monitored-{context}-{label}'
    llama=out/f'reference-{context}-A'
    audits=[]
    for folder in (native,llama):
        path=folder/'gpu-coverage.json';inputs.add(path)
        audit=json.loads(path.read_text())
        if audit.get('errors') or len(audit['runs'])!=1 or audit['runs'][0]['context']!=context:
            raise ValueError('Incomplete or mismatched audit: '+str(folder))
        for name,expected in audit['sha256'].items():
            p=Path(name) if Path(name).is_absolute() else folder/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(str(p))
            inputs.add(p)
        audits.append(audit)
    model=json.loads((native/'result.json').read_text())
    if model.get('output_parity') is not True or model.get('inputs_unchanged') is not True or model.get('graphs_reused') is not True:
        raise ValueError('Incomplete candidate correctness or graph evidence')
    for name,expected in model['sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(str(p))
        inputs.add(p)
    inputs.add(native/'result.json')
    source=native/f'{context}.stdout';inputs.add(source)
    cell=compare(audits[0]['runs'][0]['observations'],audits[1]['runs'][0]['observations'],native_text(source.read_text()),'joint-base')
    cell['sampled_coverage_clear']=[a['sampled_coverage_clear'] for a in audits]
    cell['coverage_failures']=[[
        dict(round=o['round'],mode=o.get('mode','llama'),errors=o['coverage']['errors'])
        for o in a['runs'][0]['observations']] for a in audits]
    result['contexts'][context]=cell
result['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
with destination.open('x') as f:json.dump(result,f,indent=2)
for context,cell in result['contexts'].items():
    print(context,'UNACCEPTED diagnostic ratio',cell['fyodor']['mtp-joint3']['diagnostic_speed_percent_of_llama'])
