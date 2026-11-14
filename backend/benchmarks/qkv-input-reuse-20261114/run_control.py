"""Compile and run the actual C reuse helper with controlled dispatch failures."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent;work=root/'build-cuda/qkv-input-reuse-control-20261114'
if work.exists() or (out/'control.json').exists():raise SystemExit('Refusing evidence overwrite')
proof_path=out/'build.json';proof=json.loads(proof_path.read_text())
if proof.get('build_passed') is not True:raise ValueError('Failed build')
for name,h in proof['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError(name)
resident=root/'build-cuda/qkv-input-reuse-20261114/resident.inc';text=resident.read_text()
a=text.index('static int plan_matvec_norm_reuse(');b=text.index('static int plan_gate_up(',a)
work.mkdir();helper=work/'helper.inc';helper.write_text(text[a:b])
compiler=root/'.tools/w64devkit/bin/gcc.exe';inputs=[Path(__file__).resolve(),out/'control.c',resident,helper,proof_path,compiler]
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
env=dict(os.environ,PATH=str(compiler.parent)+os.pathsep+os.environ['PATH']);commands=[];runs=[]
for standard in ('c17','c23'):
    exe=work/f'control-{standard}.exe'
    command=[str(compiler),'-std='+standard,'-O3','-Wall','-Wextra','-Wpedantic','-Wconversion','-Wshadow','-Werror','-I',str(work),str(out/'control.c'),'-o',str(exe)]
    subprocess.run(command,env=env,check=True);commands.append(command)
    run=subprocess.run([str(exe)],env=env,capture_output=True,text=True,check=True)
    if run.stdout.strip()!='384 admission/failure cases and two independent QKV sequences passed':raise ValueError('Wrong control inventory')
    runs.append(dict(standard=standard,exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr))
    hashes[str(exe.relative_to(root))]=hashlib.sha256(exe.read_bytes()).hexdigest()
for name,h in hashes.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed input: '+name)
(out/'control.json').write_text(json.dumps(dict(passed=True,gpu_execution=False,accepted_benchmark=False,commands=commands,runs=runs,sha256=hashes),indent=2))
print('C17/C23: 384 control cases and two QKV sequences each passed')
