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

# Isolate a small lane in the dot while larger zero-weight lanes determine scale.
results=[]
for coefficient in (.03,1.,32.,65504.):
 for magnitude in (1.,1024.,1048576.):
  for selected in (1,2,3,4):
   w=bytearray(144);w[:4]=struct.pack('<ee',coefficient,0.);w[4]=1;w[16+selected]=1
   one=[0.]*256
   for j,v in enumerate((1.,.003,1e-5,1e-8,1e-11)):one[j]=f32(v*magnitude)
   values=one*4
   expected,bounds=evaluate(bytes(w),values,256,1)
   for passes in (1,2,3,4,5):
    recovered=reconstruct(values,passes);approx,_=evaluate(bytes(w),recovered,256,1)
    fractions=[abs(a-b)/bound for a,b,bound in zip(approx,expected,bounds)]
    results.append(dict(coefficient=struct.unpack('<e',w[:2])[0],magnitude=magnitude,selected_lane=selected,passes=passes,original=values[selected],reconstructed=recovered[selected],expected=expected[0],approximate=approx[0],bound=bounds[0],maximum_bound_fraction=max(fractions),outside_bound=sum(x>1 for x in fractions)))
inputs.add(out/'screen.py')
result=dict(accepted_benchmark=False,gpu_execution=False,cases=results,
 limitation='Adversarial activation error only; excludes CUDA accumulation and performance.',
 sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
with (out/'adversarial.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
for passes in (1,2,3,4,5):
 selected=[r for r in results if r['passes']==passes]
 print(passes,'passes: max fraction',max(r['maximum_bound_fraction'] for r in selected),'outside outputs',sum(r['outside_bound'] for r in selected))
