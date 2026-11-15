"""Export the exact existing Q4 diagnostic module for offline PTXAS controls."""
import importlib.util
from pathlib import Path
root=Path.cwd();p=root/'backend/benchmarks/q4-pair-rows-20261114/common.py'
out=root/'build-cuda/sanitizer-kernel-overhead-20261115/kernels.ptx'
if out.exists():raise SystemExit('Refusing overwrite')
spec=importlib.util.spec_from_file_location('export_common',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(m.ptx.value)
finally:m.check(m.unload(m.module));m.check(m.release(m.device))
