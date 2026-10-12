"""Validate both complete final-policy workloads and their desktop windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('rank2', type=Path)
parser.add_argument('rank8', type=Path)
args = parser.parse_args()
summarizer = Path(__file__).resolve().parent.parent / 'training-q4-dx-20261011/summarize.py'
for rank, directory in ((2, args.rank2), (8, args.rank8)):
    results = json.loads((directory / 'results.json').read_text())
    config = results['configuration']
    assert config['rank'] == rank and config['monitor_environment'] and config['desktop_window']
    assert config['contexts'] == ([8, 32, 128] if rank == 2 else [32])
    assert config['steps'] == 20 and config['warmup'] == 5 and config['repeats'] == 2
    environment = json.loads((directory / 'environment.json').read_text())
    assert environment['method'].startswith('Ten consecutive desktop samples:')
    for activity in environment['activities']:
        for phase in ('before', 'after'):
            window = activity[phase][-10:]
            assert len(window) == 10 and sum(s['utilization'] for s in window) <= 300
            assert all(0 <= s['utilization'] <= 45 and float(s['power'].split()[0]) <= 15 for s in window)
    # Also validates all activities, sample competitors, exact serialized outputs,
    # both executable orders and all process repetitions before pooling.
    subprocess.run([sys.executable, str(summarizer), str(directory)], check=True)
