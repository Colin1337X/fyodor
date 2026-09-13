import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,copyFile,writeFile,readFile,stat,rm} from 'node:fs/promises';
import {resolve,dirname,basename} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';

// Packaging fixtures are inert bytes: never launch a backend or load a DLL.
test('CUTLASS packaging requires its license and removes stale artifacts', {skip:process.platform!=='win32'}, async()=>{
  const root=await mkdtemp(resolve(tmpdir(),'fyodor-sidecar-test-'));
  try {
    const script=resolve(root,'frontend/scripts/prepare-sidecar.mjs');
    await mkdir(dirname(script),{recursive:true});
    await copyFile(new URL('../scripts/prepare-sidecar.mjs',import.meta.url),script);
    const backend=resolve(root,'backend.exe'),trainer=resolve(root,'trainer.exe');
    const library=resolve(root,'fyodor-cutlass.dll'),license=resolve(root,'CUTLASS-LICENSE.txt');
    for(const [path,content] of [[backend,'backend fixture'],[trainer,'trainer fixture'],[library,'DLL fixture']])await writeFile(path,content);
    const env={...process.env,FYODOR_BACKEND_BIN:backend,FYODOR_TRAIN_BIN:trainer};
    for(const key of ['NYA_CUDA_NVRTC','NYA_CUDA_BLAS_LIBRARY','NYA_CUDA_CUTLASS_LIBRARY'])delete env[key];
    const run=extra=>spawnSync(process.execPath,[script],{env:{...env,...extra},encoding:'utf8',windowsHide:true});
    let result=run({NYA_CUDA_CUTLASS_LIBRARY:library});
    assert.notEqual(result.status,0);assert.match(result.stderr,/missing CUTLASS-LICENSE/);
    await writeFile(license,'license fixture');
    result=run({NYA_CUDA_CUTLASS_LIBRARY:library});assert.equal(result.status,0,result.stderr);
    const destination=resolve(root,'frontend/src-tauri/resources/backend');
    assert.equal(await readFile(resolve(destination,'fyodor-cutlass.dll'),'utf8'),'DLL fixture');
    assert.equal(await readFile(resolve(destination,'CUTLASS-LICENSE.txt'),'utf8'),'license fixture');
    result=run({});assert.equal(result.status,0,result.stderr);
    await assert.rejects(stat(resolve(destination,'fyodor-cutlass.dll')), {code:'ENOENT'});
    await assert.rejects(stat(resolve(destination,'CUTLASS-LICENSE.txt')), {code:'ENOENT'});
    assert.equal(await readFile(resolve(destination,'fyodor-backend.exe'),'utf8'),'backend fixture');
    result=run({NYA_CUDA_CUTLASS_LIBRARY:resolve(root,'missing.dll')});
    assert.notEqual(result.status,0);assert.match(result.stderr,/existing fyodor-cutlass/);
  } finally {
    // Verify the resolved temporary target before recursive Windows cleanup.
    assert.equal(dirname(root),resolve(tmpdir()));
    assert.ok(basename(root).startsWith('fyodor-sidecar-test-'));
    await rm(root,{recursive:true,force:true});
  }
});
