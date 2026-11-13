"""Recover one known interleaved log without upgrading it to a successful run."""
import hashlib,json
from pathlib import Path
out=Path(__file__).resolve().parent
path=out/'prefix-kv-narrow-prefix-ctx2048.log'
raw=path.read_bytes();text=raw.decode().replace('\r\n','\n')
traces=['CUDA batched wide-value attention calls: 43008\n',
        'CUDA interleaved batch-two Q6 calls: 45056\n']
for trace in traces:
 assert text.count(trace)==1
 text=text.replace(trace,'')
rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
assert len(rows)==145 and rows[0]['kind']=='logits'
assert {(r['layer'],r['position']) for r in rows[1:]}=={(l,p) for l in range(48) for p in (0,1024,2047)}
with (out/'prefix-kv-narrow-prefix-ctx2048-recovered.json').open('x') as output:
 json.dump({'accepted_benchmark':False,'complete_run_metadata':False,'child_exit_code':None,
 'wrapper_exit_code':1,'raw_log_sha256':hashlib.sha256(raw).hexdigest(),'removed_trace_strings':traces,
 'note':'Original wrapper failed parsing merged stdout/stderr before saving child exit status or metadata. Only reported numerical rows are recovered; original log is unchanged.',
 'rows':rows},output,indent=2)
print(json.dumps(rows[0]))
