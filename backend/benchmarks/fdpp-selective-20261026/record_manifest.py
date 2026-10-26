"""Record the source and build inputs used for this experiment."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[3]
paths=['backend/CMakeLists.txt','backend/tests/test_inference.c','CUDA/CMakeLists.txt','CUDA/resident.inc','CUDA/resident.cu',
       'optcpp/attention/portable_f32.cu','optcpp/attention/flash_decoding_pp.cu',
       'optcpp/attention/flash_decoding_pp_warp.cu',
       'optcpp/attention/flash_decoding_pp_specialized.cu']
record={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},'builds':{}}
for build in ('build-cuda','build-cuda-c23'):
 record['builds'][build]=[l for l in (root/build/'CMakeCache.txt').read_text().splitlines()
   if l.startswith(('CMAKE_BUILD_TYPE:','CMAKE_C_COMPILER:','NYA_C_STANDARD:','NYA_ENABLE_','NYA_WARNINGS_AS_ERRORS:'))]
(Path(__file__).resolve().parent/'source-manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
