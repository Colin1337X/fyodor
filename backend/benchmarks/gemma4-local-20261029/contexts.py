from pathlib import Path
import subprocess,os,json,time
r=Path.cwd();d=Path(__file__).resolve().parent;m=json.loads((d/'smoke-command.json').read_text());env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(m['environment']);results=[]
for prefix in (256,1024,4096,8192):
 cmd=m['command'].copy();cmd[cmd.index('--context')+1]=str(prefix)
 start=time.monotonic()
 with (d/f'context-{prefix}.json').open('wb') as f,(d/f'context-{prefix}.log').open('wb') as g:
  try:run=subprocess.run(cmd,env=env,stdout=f,stderr=g,timeout=300);code=run.returncode
  except subprocess.TimeoutExpired:code='timeout'
 row={'prefix':prefix,'command':cmd,'exit_code':code,'wall_seconds':time.monotonic()-start};results.append(row);(d/'contexts.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
