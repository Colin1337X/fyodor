"""CPU-only activation-error screen, not a GPU kernel or numerical admission."""
import array,ctypes as C,hashlib,json,math,random,struct
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
proof_path=out.parent/'q4-tf32-three-20261114/oracle-control.json'
proof=json.loads(proof_path.read_text());assert proof['passed']
inputs={Path(__file__),proof_path}
for n,h in proof['sha256'].items():
 p=root/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n;inputs.add(p)
lib=C.CDLL(str(root/'build-cuda/q4-tf32-three-oracle-20261114/oracle-c17.dll'))
oracle=lib.q4_oracle;oracle.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_void_p];oracle.restype=C.c_int
f32=lambda x:C.c_float(x).value

def reconstruct(values,passes):
 result=[0.]*len(values)
 for base in range(0,len(values),32):
  residual=values[base:base+32]
  for step in range(passes):
   scale=f32(max(map(abs,residual))/127.)
   if scale==0:break
   for j,x in enumerate(residual):
    ratio=f32(x/scale);q=math.copysign(math.floor(abs(ratio)+.5),ratio)
    value=f32(q*scale)
    result[base+j]=f32(result[base+j]+value)
    residual[j]=f32(x-value)
 return result

def evaluate(weights,values,k,rows):
 expected,bounds=(C.c_double*(rows*4))(),(C.c_double*(rows*4))()
 x=array.array('f',values).tobytes()
 assert oracle(C.create_string_buffer(weights),C.create_string_buffer(x),k,rows,expected,bounds)==0
 return list(expected),list(bounds)

assert reconstruct([0.]*32,3)==[0.]*32
assert reconstruct([127.]*32,1)==[127.]*32
assert reconstruct([-127.]*32,1)==[-127.]*32
rng=random.Random(713);results=[]
for k in (256,3840,8192,15360):
 rows=17;stride=k//256*144;w=bytearray(rng.randbytes(rows*stride))
 for offset in range(0,len(w),144):w[offset:offset+4]=struct.pack('<ee',.03,-.004)
 for pattern in ('random','multiscale','impulse'):
  values=[f32(rng.uniform(-1,1)) for _ in range(k*4)]
  if pattern=='multiscale':values=[f32((-1 if j%2 else 1)*2.**(-(j%24))) for j in range(k*4)]
  if pattern=='impulse':values=[float(j%k==(j//k)*37) for j in range(k*4)]
  expected,bounds=evaluate(bytes(w),values,k,rows)
  for passes in (1,2,3):
   approx,_=evaluate(bytes(w),reconstruct(values,passes),k,rows)
   fractions=[abs(a-b)/bound for a,b,bound in zip(approx,expected,bounds)]
   results.append(dict(columns=k,rows=rows,pattern=pattern,passes=passes,maximum_bound_fraction=max(fractions),outside_bound=sum(x>1 for x in fractions)))
inputs.add(root/'build-cuda/q4-tf32-three-oracle-20261114/oracle-c17.dll')
result=dict(accepted_benchmark=False,gpu_execution=False,cases=results,
 limitation='Only activation-reconstruction error evaluated with double dots; no CUDA accumulation, performance or full numerical admission. F32 scales differ from llama Q8_1 half scales.',
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
for passes in (1,2,3):
 selected=[r for r in results if r['passes']==passes]
 print(passes,'passes: max fraction',max(r['maximum_bound_fraction'] for r in selected),'outside outputs',sum(r['outside_bound'] for r in selected))
