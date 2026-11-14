import hashlib,json
from pathlib import Path
out=Path(__file__).resolve().parent
counts={}
for standard in ('17','23'):
 p=json.loads((out/('c'+standard+'.json')).read_text());assert p['passed'] and all(p['absent'].values())
 for name,h in p['sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
 for name,h in p['source_sha256'].items():assert hashlib.sha256((Path(p['source'])/name).read_bytes()).hexdigest()==h,name
 counts[standard]=len(p['source_sha256'])+len(p['sha256'])
print('All C17/C23 source/artifact hashes verified:',counts)
