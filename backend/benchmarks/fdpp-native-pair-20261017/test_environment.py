"""A pre-existing C+G game must never become an allowed desktop identity."""
from pathlib import Path
import importlib.util
root=Path(__file__).resolve().parents[3]
for path in (root/'backend/benchmarks/score-refresh-20261011/environment.py',
             Path(__file__).with_name('environment.py')):
 spec=importlib.util.spec_from_file_location('gate_under_test',path)
 gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 for name,kind,expected in (
     ('C:\\Windows\\System32\\dwm.exe','C+G',True),
     ('C:/Apps/CODEX.EXE','C+G',True),
     ('C:/Apps/chrome.exe','C+G',True),
     ('C:/Apps/chrome.exe','C',False),
     ('C:/Apps/RobloxPlayerBeta.exe','C+G',False),
     ('C:/Apps/OtherGame.exe','C+G',False),
     ('C:/Apps/NVIDIA Overlay.exe','C+G',False)):
  process={'pid':'99','name':name,'type':kind}
  assert gate.desktop_identity_allowed(process)==expected,(path,process)
  gate.desktop={(process['pid'],process['name'])} if expected else set()
  assert bool(gate.competitors({'processes':[process]}))== (not expected)
 print(path.name, 'desktop classification and competitor rejection passed')
