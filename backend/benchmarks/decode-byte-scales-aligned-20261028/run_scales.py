import importlib.util,sys,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('environment',here.parent/'score-refresh-20261011/environment.py')
env=importlib.util.module_from_spec(spec);spec.loader.exec_module(env)
out=here/'monitor-scales';out.mkdir(exist_ok=False);env.configure(out)
@env.capture('byte scale metadata diagnostic')
def run(script,target,extra=[]):
 p=subprocess.Popen([sys.executable,str(here/script),str(here/target),*extra]);env.set_active_pid(p.pid)
 try: assert p.wait()==0
 finally: env.set_active_pid(None)
run('probe_scales.py','scales')
run('probe_scales.py','scales-extended',['extended'])
