"""Compare every optimizer array without requiring PyTorch or NumPy."""
import argparse, hashlib, json, math, struct
from pathlib import Path

def read(path):
    data = path.read_bytes()
    count, step = struct.unpack_from('<QQ',data,24)
    offset = 64; states = []
    for _ in range(count):
        rows, cols = struct.unpack_from('<QQ',data,offset); offset += 16
        arrays = []
        for _ in range(4):
            arrays.append(struct.unpack_from('<'+str(rows*cols)+'f',data,offset))
            offset += rows*cols*4
        states.append(((rows,cols),arrays))
    assert offset+8 == len(data)
    return data,step,states

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('left',type=Path); parser.add_argument('right',type=Path)
args = parser.parse_args()
left,ls,a = read(args.left); right,rs,b = read(args.right)
assert ls == rs and len(a) == len(b)
errors = [0.]*4
for (shape,x),(other,y) in zip(a,b):
    assert shape == other
    for index,(u,v) in enumerate(zip(x,y)):
        assert all(math.isfinite(t) for t in (*u,*v))
        errors[index] = max(errors[index],max(abs(i-j)/(1+abs(i)) for i,j in zip(u,v)))
record = {'left':str(args.left),'right':str(args.right),'step':ls,
          'left_sha256':hashlib.sha256(left).hexdigest(),'right_sha256':hashlib.sha256(right).hexdigest(),
          'maximum_scaled_errors':dict(zip(('parameter','gradient','moment','variance'),errors)),
          'tolerance':1e-5,'passed':all(x<=1e-5 for x in errors)}
print(json.dumps(record,indent=2)); assert record['passed']
