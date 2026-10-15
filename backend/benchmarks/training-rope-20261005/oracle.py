"""Independent 100-digit values used by the CPU large-angle regression."""
import json,mpmath as m
from pathlib import Path
m.mp.dps=100
angle=m.mpf(2)**128-m.mpf(2)**104
record={"oracle":"mpmath","version":m.__version__,"decimal_digits":100,"angle_exact":str(angle),"cos":str(m.cos(angle)),"sin":str(m.sin(angle))}
Path(__file__).with_name("oracle.json").write_text(json.dumps(record,indent=2))
