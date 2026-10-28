from pathlib import Path
import struct,json,collections
out=Path(__file__).resolve().parent
files=[Path('E:/models')/n for n in ('Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf','mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf','mtp-gemma-4-12B-it.gguf','nomic-embed-text-v2-moe-q8_0.gguf')]
results=[]
for path in files:
 with path.open('rb') as f:
  def take(fmt):
   return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
  def string(keep=True):
   n=take('Q')
   if keep:return f.read(n).decode('utf-8',errors='replace')
   f.seek(n,1)
  formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
  def value(t,keep=True):
   if t==8:return string(keep)
   if t==9:
    subtype=take('I');n=take('Q')
    if n<=64:return [value(subtype,keep) for _ in range(n)]
    if subtype in formats:f.seek(n*struct.calcsize('<'+formats[subtype]),1)
    else:
     for _ in range(n):value(subtype,False)
    return {'array_type':subtype,'count':n}
   return take(formats[t])
  assert f.read(4)==b'GGUF';version=take('I');tensors=take('Q');count=take('Q');meta={}
  for _ in range(count):key=string();meta[key]=value(take('I'))
  shapes=[];types=collections.Counter()
  for _ in range(tensors):
   name=string();dims=[take('Q') for _ in range(take('I'))];typ=take('I');offset=take('Q');types[typ]+=1;shapes.append({'name':name,'dimensions':dims,'type':typ,'offset':offset})
  results.append({'path':str(path),'bytes':path.stat().st_size,'version':version,'tensor_count':tensors,'types':dict(types),'metadata':meta,'tensors':shapes})
(out/'model-inventory.json').write_text(json.dumps(results,indent=2))
for model in results:
 print(model['path'],model['bytes'],model['tensor_count'],model['types'])
 print(json.dumps({k:v for k,v in model['metadata'].items() if k.startswith(('general.architecture','gemma4.','nomic-bert-moe.','clip.'))},indent=2))
