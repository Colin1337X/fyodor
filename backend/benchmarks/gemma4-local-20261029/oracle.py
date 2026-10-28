from pathlib import Path
import subprocess,os,json
r=Path.cwd();d=Path(__file__).resolve().parent;m=json.loads((d/'smoke-command.json').read_text());env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(m['environment']);env.update(NYA_COMPUTE='cuda',NYA_TEST_CPU_OPTIMIZED='1');env['PATH']=str(r/'.tools/llama-b10809-cuda')+';'+env['PATH']
cmd=[str(r/'build-cuda/nya-test-inference.exe'),m['command'][2],'resident','9']
with (d/'cpu-oracle-9.log').open('wb') as f:
 try:run=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240);code=run.returncode
 except subprocess.TimeoutExpired:code='timeout'
(d/'oracle-command.json').write_text(json.dumps({'command':cmd,'exit_code':code},indent=2));print(code)
