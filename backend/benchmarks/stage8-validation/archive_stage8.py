import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'backend/benchmarks/stage8-validation';out.mkdir(exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p in (root/'.tools').glob('stage8-*'):
    if p.is_file() and p.suffix in ('.log','.json'):shutil.copy2(p,out/p.name)
for name in ['stage8_probe.py','stage8_final_bench.py','stage8_validate.py','stage8_post.py','stage8_resume.py','stage8_browser.py','stage8_tables.py','archive_stage8.py']:
    shutil.copy2(root/'.tools'/name,out/name)
shutil.copy2(root/'frontend/qa/stage3/results.json',out/'browser-results.json')
shutil.copy2(root/'build-cuda/Testing/Temporary/LastTest.log',out/'accepted-ctest-detail.log')
sources=[]
for directory in ('backend/core','backend/include','backend/tests','backend/cmake','CUDA','frontend/src','frontend/tests','frontend/scripts'):
    for p in (root/directory).rglob('*'):
        if p.is_file() and 'vendor' not in p.parts and p.suffix in ('.c','.h','.cu','.inc','.cmake','.js','.mjs','.css'):sources.append(p)
sources += [root/p for p in ('CMakeLists.txt','backend/CMakeLists.txt','CUDA/CMakeLists.txt','CUDA/cutlass/CMakeLists.txt','.gitattributes','backend/README.md','CUDA/cutlass/README.md','backend/benchmarks/STAGE8.md')]
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':{p.relative_to(root).as_posix():sha(p) for p in sources},
 'source_lf_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources},
 'evidence_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='metadata.json'},'artifact_sha256':{},
 'notes':['BF16x6 failed correctness and was removed. Failure logs and source are historical experiment evidence, not production features.',
 'The first BF16 build exposed an independent MSVC Debug /RTC1 versus NVCC -O3 conflict; accepted Debug/Release builds and their quant tests pass.',
 'GPU racecheck filters nya_expand_t_; memcheck/initcheck/synccheck instrument the full quantized-matrix test process including producer kernels.',
 'Initial memcheck could not attach and was terminated. The accepted memcheck-retry and subsequent checks use application-only tracking, port 54321, a 60-second attach limit and --kill. The root cause of the original attach failure is not established.',
 'Initial paired layout probe uses seven repetitions; final comparison uses nine repetitions in each forward/reverse run and three warmups.',
 'No benchmark overlapped a Fyodor validation or packaging job. Desktop apps remained present, and GPU telemetry is preserved per run.',
 'Final accepted comparison uses a matched 7..12% desktop-activity gate; all observed idle readings were 9%. Roblox/browser contamination and mixed desktop states are rejected in separately preserved directories. A fully idle GPU result is not claimed.',
 'Raw source hashes describe worktree bytes; LF-normalized hashes permit comparison with Git blobs.']}
for relative in ['.tools/fyodor-bench-stage7.exe','build-cuda/fyodor-bench.exe','build-cuda/fyodor-backend.exe','build-cuda/nya-test-inference.exe','build-cuda/nya-test-quant-kernels.exe','build-cutlass/fyodor-cutlass.dll','build-cutlass-debug/fyodor-cutlass.dll',
 'frontend/src-tauri/resources/backend/fyodor-backend.exe','frontend/src-tauri/resources/backend/fyodor-cutlass.dll','frontend/src-tauri/resources/backend/CUTLASS-LICENSE.txt',
 '.tools/llama-b10809-cuda/llama.dll','.tools/llama-b10809-cuda/ggml-cuda.dll','.tools/llama-b10809-cuda/cublas64_13.dll',
 'frontend/src-tauri/target/release/bundle/msi/Fyodor_0.3.0_x64_en-US.msi','frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe']:
    p=root/relative;meta['artifact_sha256'][relative]={'sha256':sha(p),'bytes':p.stat().st_size}
assert meta['artifact_sha256']['build-cuda/fyodor-backend.exe']['sha256']==meta['artifact_sha256']['frontend/src-tauri/resources/backend/fyodor-backend.exe']['sha256']
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print('Archived',len(meta['evidence_sha256']),'evidence files;',len(sources),'source hashes')
