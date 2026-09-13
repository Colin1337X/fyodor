import {copyFile,mkdir,stat,readFile,readdir,unlink} from 'node:fs/promises';
import {resolve,dirname,basename} from 'node:path';
import process from 'node:process';
const root=resolve(import.meta.dirname,'../..'),ext=process.platform==='win32'?'.exe':'';
const destination=resolve(root,'frontend/src-tauri/resources/backend');
await mkdir(destination,{recursive:true});
const exists=async path=>{try{return (await stat(path)).isFile()}catch{return false}};
// An explicit override is authoritative. A typo must not quietly package an old
// CPU executable. Otherwise prefer an existing CUDA build, then Vulkan, then CPU.
let backendSource;
for(const [stem,env] of [['fyodor-backend','FYODOR_BACKEND_BIN'],['fyodor-train','FYODOR_TRAIN_BIN']]){
  const candidates=process.env[env]?[resolve(process.env[env])]:['build-cuda','build-vulkan','build-cpu'].flatMap(dir=>[resolve(root,dir,stem+ext),resolve(root,dir,'Release',stem+ext)]);
  let source;
  for(const file of candidates)if(await exists(file)){source=file;break;}
  if(!source)throw new Error(`Build ${stem} or set ${env} to a valid native executable.`);
  await copyFile(source,resolve(destination,stem+ext));
  if(stem==='fyodor-backend')backendSource=source;
  console.log(`Prepared ${stem} from ${source}`);
}
// Only remove known generated CUDA compiler resources in this exact bundle
// directory. This prevents an old compiler surviving a subsequent CPU package.
for(const name of await readdir(destination))if(/^nvrtc(?:64_|-builtins64_).*\.dll$/.test(name))await unlink(resolve(destination,name));
// Optional BLAS is independent of CUDA's native kernels. Bundle only when
// explicitly supplied; removing these DLLs preserves the custom CUDA path.
for(const name of await readdir(destination))if(/^cublas(?:Lt)?64_\d+\.dll$/.test(name))await unlink(resolve(destination,name));
if(process.platform==='win32'&&process.env.NYA_CUDA_BLAS_LIBRARY){
  const library=resolve(process.env.NYA_CUDA_BLAS_LIBRARY),match=basename(library).match(/^cublas64_(\d+)\.dll$/);
  if(!match||!await exists(library))throw new Error('NYA_CUDA_BLAS_LIBRARY must name an existing cublas64 DLL.');
  const companion=resolve(dirname(library),`cublasLt64_${match[1]}.dll`);
  if(!await exists(companion))throw new Error('Optional cuBLAS package is missing its matching cuBLASLt DLL.');
  for(const file of [library,companion])await copyFile(file,resolve(destination,basename(file)));
  console.log('Prepared optional cuBLAS and cuBLASLt (native CUDA remains available without them).');
}
// A standalone CUTLASS build exposes only a C ABI DLL. Its presence is optional,
// and a subsequent package without the override must remove the generated copy.
for(const name of ['fyodor-cutlass.dll','CUTLASS-LICENSE.txt'])if(await exists(resolve(destination,name)))await unlink(resolve(destination,name));
if(process.platform==='win32'&&process.env.NYA_CUDA_CUTLASS_LIBRARY){
  const library=resolve(process.env.NYA_CUDA_CUTLASS_LIBRARY);
  if(basename(library)!=='fyodor-cutlass.dll'||!await exists(library))throw new Error('NYA_CUDA_CUTLASS_LIBRARY must name an existing fyodor-cutlass.dll.');
  const license=resolve(dirname(library),'CUTLASS-LICENSE.txt');
  if(!await exists(license))throw new Error('Optional CUTLASS package is missing CUTLASS-LICENSE.txt; reconfigure its standalone CMake build.');
  await copyFile(library,resolve(destination,'fyodor-cutlass.dll'));
  await copyFile(license,resolve(destination,'CUTLASS-LICENSE.txt'));
  console.log('Prepared optional CUTLASS bridge and upstream license (cuBLAS retains priority).');
}
if(process.platform==='win32'){
  let runtime=process.env.NYA_CUDA_NVRTC;
  if(!runtime){
    for(const dir of [dirname(backendSource),resolve(dirname(backendSource),'..')]){
      try{const cache=await readFile(resolve(dir,'CMakeCache.txt'),'utf8');runtime=cache.match(/^NYA_CUDA_NVRTC_LIBRARY:FILEPATH=(.+)$/m)?.[1].trim();if(runtime)break;}catch{}
    }
  }
  if(runtime&&await exists(runtime)){
    const dir=dirname(runtime);
    const names=(await readdir(dir)).filter(name=>/^nvrtc(?:64_|-builtins64_).*\.dll$/.test(name));
    for(const name of names)await copyFile(resolve(dir,name),resolve(destination,name));
    // Preserve the SDK archive's license alongside redistributed runtime files.
    for(const dirCandidate of [dir,resolve(dir,'..'),resolve(dir,'../..')]){
      for(const name of ['LICENSE','LICENSE.txt'])if(await exists(resolve(dirCandidate,name)))await copyFile(resolve(dirCandidate,name),resolve(destination,'CUDA-LICENSE.txt'));
    }
    console.log(`Prepared CUDA runtime: ${basename(runtime)} and sibling builtins`);
  }else console.log('No bundled NVRTC compiler; CUDA requires a runtime installed on the target machine.');
}
