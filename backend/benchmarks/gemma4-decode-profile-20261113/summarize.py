from pathlib import Path
import re,json
out=Path(__file__).resolve().parent;meta=json.loads((out/'metadata.json').read_text());results=[]
assert len(meta['runs'])==4 and all(r['exit_code']==0 for r in meta['runs'])
for run in meta['runs']:
 log=(out/(run['name']+'.log')).read_text()
 match=re.search(r'CUDA kernel GPU milliseconds: ([^\n]+)',log)
 categories={k:float(v) for k,v in re.findall(r'(matmul|norm|rope|vector|attention|gather)=([0-9.]+)',match[1])}
 total=sum(categories.values())
 matrices=[]
 for match in re.finditer(r'CUDA matrix profile: path=(\S+) type=(\d+) rows=(\d+) columns=(\d+) batch=(\d+) calls=(\d+) milliseconds=([0-9.]+)',log):
  path,kind,rows,cols,batch,calls,ms=match.groups();assert batch=='1'
  matrices.append(dict(path=path,type=int(kind),rows=int(rows),columns=int(cols),calls=int(calls),milliseconds=float(ms)))
 matrices.sort(key=lambda m:m['milliseconds'],reverse=True)
 results.append({'name':run['name'],'decode_tokens':16,'event_milliseconds_per_token':{k:v/16 for k,v in categories.items()},'event_attributed_percent':{k:100*v/total for k,v in categories.items()},'matrix_operations':matrices})
summary={'accepted_performance':False,'limitations':'Single diagnostic profile per shape, graphs disabled; event brackets may include submission gaps and other scheduling effects. Not kernel hardware-counter measurements, benchmark throughput, or accepted score changes.','results':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for row in results:print(row['name'],row['event_milliseconds_per_token'],row['event_attributed_percent'])
