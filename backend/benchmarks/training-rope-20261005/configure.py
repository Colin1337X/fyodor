"""Configure isolated validation builds using installed local SDKs."""
from pathlib import Path
import subprocess, os, json
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
reference=Path("C:/Users/Colin/fyodor")
env=dict(os.environ)
env["PATH"]=str(root/".tools/w64devkit/bin")+os.pathsep+env["PATH"]
keys={"CMAKE_C_COMPILER","CMAKE_MAKE_PROGRAM","CMAKE_BUILD_TYPE","NYA_C_STANDARD","NYA_ENABLE_CUDA","NYA_ENABLE_VULKAN","NYA_ENABLE_ONNXRUNTIME","NYA_ENABLE_SANITIZERS","NYA_WARNINGS_AS_ERRORS","NYA_CUDA_INCLUDE_DIR","NYA_NVRTC_INCLUDE_DIR","NYA_CUDA_NVRTC_LIBRARY","Vulkan_INCLUDE_DIR","Vulkan_LIBRARY","NYA_ONNXRUNTIME_ROOT"}
records=[]
for build in ("build-cpu","build-cuda","build-cuda-c23","build-ort","build-sanitize"):
 settings={}
 for line in (reference/build/"CMakeCache.txt").read_text().splitlines():
  if "=" in line and ":" in line.split("=",1)[0]:
   key=line.split(":",1)[0]
   if key in keys: settings[key]=line.split("=",1)[1]
 command=[str(root/".tools/w64devkit/bin/cmake.exe"),"-S","backend","-B",build,"-G","MinGW Makefiles"]+[f"-D{k}={v}" for k,v in settings.items()]
 log=out/(build+"-configure.log")
 attempt=1
 while log.exists():
  log=out/(build+f"-configure-{attempt}.log"); attempt+=1
 with log.open("xb") as f: result=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append({"command":command,"exit_code":result.returncode,"log":log.name})
 (out/"configure.json").write_text(json.dumps(records,indent=2))
 print(build,result.returncode,flush=True)
 if result.returncode: raise SystemExit(result.returncode)
