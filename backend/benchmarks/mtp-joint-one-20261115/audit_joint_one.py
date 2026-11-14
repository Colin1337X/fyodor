"""Explicit joint-one inventory using unchanged coverage/lifetime validation."""
import sys
from pathlib import Path
MONITOR=Path(__file__).resolve().parent.parent/'gpu-monitor-20261114'
sys.path.insert(0,str(MONITOR))
import audit_generation as base
base.MODE_SETS['joint-one']=('ordinary','mtp-serial2','mtp-batch2','mtp-joint1')
audit_run=base.audit_run
if __name__=='__main__':base.main()
