import hashlib,importlib.util,json,os,subprocess,time,sys
from pathlib import Path
from contextlib import nullcontext
root=Path.cwd(); out=Path(__file__).resolve().parent
mode_set=os.environ.get('NYA_BENCH_MODE_SET','joint-one')
if mode_set not in ('joint-base','joint-one'):
    raise SystemExit('Unknown benchmark mode set')
prerequisites=[out/'comparison-build-proof.json',out/'model-memcheck-short/result.json',out/'long-correctness/correctness.json',out.parent/'mtp-joint-base-v2-20261114/model-memcheck-short/result.json',out.parent/'mtp-joint-base-v2-20261114/short-correctness/correctness.json']
checked={}
for path in prerequisites:
    if not path.exists():raise SystemExit('Wait for completed prerequisite: '+str(path))
    proof=json.loads(path.read_text())
    if proof.get('passed') is not True:raise SystemExit('Failed prerequisite: '+str(path))
    for name,expected in proof['sha256'].items():
        source=root/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise SystemExit('Changed prerequisite: '+name)
        checked[str(source.resolve())]=expected
    checked[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
if os.environ.get('NYA_BENCH_GPU_MONITOR') != '1':
    raise SystemExit('Continuous GPU monitoring is required for this comparison')
sys.path.insert(0,str(out.parent/'mtp-joint-base-v2-20261114'))
import model_identity
identity=model_identity.verify_gemma()
checked[str(Path(model_identity.__file__).resolve())]=hashlib.sha256(Path(model_identity.__file__).read_bytes()).hexdigest()
expected_work='joint-one-generation-20261115' if mode_set=='joint-one' else 'joint-base-generation-v2-20261114'
expected_exe=(root/'build-cuda'/expected_work/'generation.exe').resolve()
if len(sys.argv)<3 or (root/sys.argv[2]).resolve()!=expected_exe:raise SystemExit('Choose the verified executable for this mode set')
if str(expected_exe) not in checked:raise SystemExit('Executable absent from verified comparison inputs')
contexts=sys.argv[3:] or ['short','long']
if len(set(contexts))!=len(contexts) or any(context not in ('short','long') for context in contexts):
    raise SystemExit('Contexts must be distinct short/long names')
from comparison_settings import load_settings
settings=load_settings([out/'short-correctness/timing.json',
                        out.parent/'mtp-joint-base-v2-20261114/short-correctness/timing.json'],checked)
settings_source=Path(__file__).with_name('comparison_settings.py')
checked[str(settings_source.resolve())]=hashlib.sha256(settings_source.read_bytes()).hexdigest()
if len(sys.argv)>1:out=out/sys.argv[1]
out.mkdir(exist_ok=False)
if (out/'timing.json').exists():raise SystemExit('Refusing to overwrite existing timing evidence')
spec=importlib.util.spec_from_file_location('gate',root/'backend/benchmarks/gemma4-qkv-20261029/environment.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
gate.configure(out)
try:gate.idle();accepted=True
except RuntimeError:accepted=False
env={k:v for k,v in os.environ.items() if not k.startswith('NYA_')};env.update(settings)
exe=root/(sys.argv[2] if len(sys.argv)>2 else 'build-cuda/mtp-cost.exe')
meta={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'environment':settings,'gate_passed':accepted,'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'backend/core/llm_cpu.c',Path(__file__),Path(__file__).with_name('generation.c')]},'runs':[]}
meta['accepted_benchmark']=False
meta['prerequisites_sha256']=checked
meta['model_identity']=identity
meta['mode_set']=mode_set
meta['continuous_gpu_monitor']=os.environ.get('NYA_BENCH_GPU_MONITOR')=='1'
monitor_directory=root/'backend/benchmarks/gpu-monitor-20261114'
for source in [Path(__file__).with_name('audit_joint_one.py'),
               *[monitor_directory/name for name in ('monitor_capture.py','capture_pdh.py','pdh_counter.py','coverage.py','audit_generation.py')]]:
    checked[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
monitor_context=nullcontext()
if meta['continuous_gpu_monitor']:
    if b'Generation interval:' not in exe.read_bytes():
        raise SystemExit('Monitored runs require a binary with native decode endpoints')
    spec=importlib.util.spec_from_file_location('monitor_capture',monitor_directory/'monitor_capture.py')
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    monitor_context=monitor.capture(out)
with monitor_context as monitor_state:
    for context in contexts:
        if monitor_state is not None:monitor_state.check_alive()
        command=[str(exe),context]
        row={'context':context,'command':command,'samples':[],
             'observer_clock':'qpc' if os.name=='nt' else 'perf_counter'};meta['runs'].append(row)
        with (out/(context+'.stdout')).open('x') as stdout,(out/(context+'.stderr')).open('x') as stderr:
            row['spawn_before_seconds']=time.perf_counter()
            process=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
            row['spawn_after_seconds']=time.perf_counter()
            row['process_id']=process.pid
            try:
                while process.poll() is None:
                    if monitor_state is not None:monitor_state.check_alive()
                    row['samples'].append(gate.snapshot())
                    (out/'timing.json').write_text(json.dumps(meta,indent=2)+'\n')
                    time.sleep(1)
            except BaseException as error:
                row['observer_error']=str(error) or type(error).__name__
                raise
            finally:
                if process.poll() is None:
                    process.terminate();process.wait(timeout=15)
                    row['terminated_by_observer']=True
                row['exit_observed_seconds']=time.perf_counter()
                row['exit_code']=process.returncode
                (out/'timing.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(context,row['exit_code'],flush=True)
        if process.returncode:raise SystemExit(process.returncode)
if meta['continuous_gpu_monitor']:
    command=[sys.executable,str(Path(__file__).with_name('audit_joint_one.py')),str(out),
             '--samples',str(out/'gpu-pdh.jsonl'),'--luid',monitor_state['adapter']['counter_luid'],
             '--output',str(out/'gpu-coverage.json'),'--mode-set',mode_set]
    result=subprocess.run(command,env=env)
    if result.returncode:raise SystemExit(result.returncode)

for name,expected in checked.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise SystemExit('Inputs changed during run: '+name)
