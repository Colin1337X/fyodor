"""Compare native and eager F32 gradients with the diagnostic F64 oracle."""
from pathlib import Path
import argparse, hashlib, json, struct
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native',type=Path,required=True)
parser.add_argument('--f32',type=Path,required=True)
parser.add_argument('--f64',type=Path,required=True)
args = parser.parse_args()
data = args.native.read_bytes(); count,step = struct.unpack_from('<QQ',data,24)
assert step == 1
offset = 64; native = []
for _ in range(count):
    rows,cols = struct.unpack_from('<QQ',data,offset); offset += 16
    state = []
    for _ in range(4):
        state.append(torch.tensor(struct.unpack_from('<'+str(rows*cols)+'f',data,offset),dtype=torch.float64).reshape(rows,cols))
        offset += rows*cols*4
    native.append(state)
assert offset+8 == len(data)
f32 = torch.load(args.f32,map_location='cpu',weights_only=True)
f64 = torch.load(args.f64,map_location='cpu',weights_only=True)
assert len(native) == len(f32) == len(f64)
report = {'inputs':{key:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                    for key,path in vars(args).items()},'per_parameter':[]}
for index,(n,f,d) in enumerate(zip(native,f32,f64)):
    row = {'parameter':index}
    for label,state in (('native',n),('torch_f32',f)):
        delta = state[1].double()-d[1]
        row[label] = {'gradient_maximum_absolute_error':delta.abs().max().item(),
                      'gradient_rms_error':delta.square().mean().sqrt().item(),
                      'parameter_maximum_scaled_error':((state[0].double()-d[0]).abs()/(1+d[0].abs())).max().item()}
    report['per_parameter'].append(row)
report['selected_coordinates'] = []
for index,flat in ((10,149315),(9,108383),(7,44365)):
    report['selected_coordinates'].append({'parameter':index,'flat_index':flat,
        'native_gradient':native[index][1].flatten()[flat].item(),
        'torch_f32_gradient':f32[index][1].flatten()[flat].item(),
        'torch_f64_gradient':f64[index][1].flatten()[flat].item()})
print(json.dumps(report,indent=2))
