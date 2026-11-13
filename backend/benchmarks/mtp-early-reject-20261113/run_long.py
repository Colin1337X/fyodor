"""Follow the short-context screen with matched long-context evidence."""
import json,os,subprocess,sys
from pathlib import Path
out=Path(__file__).resolve().parent
short=json.loads((out/'short-summary.json').read_text())
assert all(row['parity'] for case in short['results'].values() for row in case['rows'])
env=os.environ.copy()
env.update(NYA_CUDA_SMALL_BATCH_REUSE='1',NYA_CUDA_WIDE_BATCH='1',NYA_CUDA_MTP_WIDE_VALUE='0')
for label,enabled in [('long-off','0'),('long-on','1')]:
 env['NYA_SPECULATIVE_EARLY_REJECT']=enabled
 subprocess.run([sys.executable,'backend/benchmarks/mtp-cost-20261113/run.py','../mtp-early-reject-20261113/'+label,'build-cuda/mtp-early-reject.exe','long'],env=env,check=True)
