"""Host algebra check only; does not execute or validate the CUDA kernel."""
import hashlib
import json
from pathlib import Path
import random

here = Path(__file__).resolve().parent
root = here.parents[2]
destination = here / 'metadata-identity.json'
if destination.exists():
    raise SystemExit('Refusing to overwrite algebra evidence')


def check(data):
    a, c, e = (int.from_bytes(data[i:i+4], 'little') for i in (0, 4, 8))
    scales = (a & 0x3f3f3f3f, (e & 0x0f0f0f0f) | ((a & 0xc0c0c0c0) >> 2))
    minima = (c & 0x3f3f3f3f, ((e >> 4) & 0x0f0f0f0f) | ((c & 0xc0c0c0c0) >> 2))
    for g in range(8):
        scale = data[g] & 63 if g < 4 else (data[g+4] & 15) | ((data[g-4] >> 6) << 4)
        minimum = data[g+4] & 63 if g < 4 else (data[g+4] >> 4) | ((data[g] >> 6) << 4)
        shift = (g % 4) * 8
        assert (scales[g//4] >> shift) & 255 == scale
        assert (minima[g//4] >> shift) & 255 == minimum


for a in range(256):
    for e in range(256):
        # Covers every byte pair for each high scale/minimum reconstruction.
        check(bytes([a]*8 + [e]*4))
rng = random.Random(713)
for _ in range(10000):
    check(rng.randbytes(12))  # Independent bytes also check lane isolation.
paths = (Path(__file__), root/'optcpp/matvec/integer_metadata_bound4.cu')
with destination.open('x') as output:
    json.dump(dict(passed=True, accepted_benchmark=False, gpu_execution=False,
        exhaustive_byte_pairs=65536, random_metadata_blocks=10000, seed=713,
        sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitation='Host algebra identity only; raw CUDA parity and timing remain required.'), output, indent=2)
print('65536 byte pairs and 10000 independent metadata blocks matched')
