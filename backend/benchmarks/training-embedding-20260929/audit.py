"""Verify embedding residency, numerical-order regression and package evidence."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
snapshot=json.loads((out/'snapshot.json').read_text())
for name,digest in snapshot['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
matrix=json.loads((out/'matrix/validation.json').read_text())
assert len(matrix)==15 and all(stage['exit_code']==0 for stage in matrix)
sanitizers=json.loads((out/'sanitizers.json').read_text())
assert {run['tool'] for run in sanitizers}=={'memcheck','initcheck','synccheck','racecheck'}
for run in sanitizers:
    assert run['exit_code']==0
    log=(out/(run['tool']+'.log')).read_text()
    assert ('0 hazards displayed (0 errors, 0 warnings)' if run['tool']=='racecheck' else 'ERROR SUMMARY: 0 errors') in log
    assert 'resident embeddings:' in (out/(run['tool']+'.out')).read_text()
probes=json.loads((out/'probes.json').read_text())
assert len(probes)==5 and all(run['exit_code']==0 for run in probes)
for run in probes:
    result=json.loads((out/(run['label']+'.out')).read_text())
    assert result['tolerance']==1e-5 and not result['timing_accepted'] and len(result['ms'])==5
    for key,value in result.items():
        if key.startswith('max_scaled_'): assert 0<=value<=1e-5,(run['label'],key)
    if result['component']=='embedding-rms-ffn':
        assert result['uploads']==7 and result['downloads']==4 and result['table_rows']==32000
        assert result['live_bytes']<result['peak_bytes']<768*1024*1024
    else:
        baseline_folder='training-norm-20260929' if result['component']=='silu-ffn-with-trainable-rms' else 'training-elementwise-20260928'
        mode='rmsffn8' if baseline_folder=='training-norm-20260929' else 'ffn8'
        previous=json.loads((out.parent/baseline_folder/(mode+'.out')).read_text())
        for field in ('peak_bytes','live_bytes','uploads','downloads'): assert result[field]==previous[field]
experiment=json.loads((out/'matrix-order-experiment.json').read_text())
assert experiment['after_sha256']==snapshot['source_sha256']['CUDA/training.cu']
assert experiment['failed_real_probe_exit_code']==experiment['failed_reference_probe_exit_code']==experiment['failed_weight_gradient_regression_exit_code']==1
assert 'scaled_error=1.5427189e-05' in (out/'embedffn8.log').read_text()
assert 'value==initial' in (out/'order-regression-before.log').read_text()
assert (out/'matrix-accumulation-order.patch').stat().st_size>0
package=json.loads((out/'desktop-package.json').read_text())
assert package['cli_exit_code']==0 and len(package['http_calls'])==8
assert not package['installed'] and not package['interactive_ui_checked']
assert snapshot['inference_bundle_identical_to_module_split']
record={'source_hashes_match':True,'matrix_stages_passed':15,'complete_embedding_sanitizers_passed':4,
        'component_probes_passed':5,'matrix_order_failure_and_fix_preserved':True,
        'real_ffn_tokens':[8,64],'real_ffn_scaled_tolerance':1e-5,'inference_bundle_unchanged':True,
        'timings_accepted':False,'complete_gpu_training':False,
        'package_sha256':package['sha256'],'package_bytes':package['bytes']}
(out/'audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('Embedding and matrix-order evidence audit passed.')
