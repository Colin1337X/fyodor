#Requires -Version 7.0
# Windows PowerShell 5 treats native informational stderr as a terminating error
# under Stop. Run this script with pwsh so the native exit code is authoritative.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$env:FYODOR_BACKEND_BIN = Join-Path $taskRoot 'build-cuda/fyodor-backend.exe'
$env:FYODOR_TRAIN_BIN = Join-Path $taskRoot 'build-cuda/fyodor-train.exe'
$env:NYA_CUDA_BLAS_LIBRARY = Join-Path $taskRoot '.tools/llama-b10809-cuda/cublas64_13.dll'
$env:NYA_CUDA_CUTLASS_LIBRARY = Join-Path $taskRoot 'build-cutlass/fyodor-cutlass.dll'
$env:CARGO_BUILD_JOBS = '1'
$env:PATH = 'C:\Program Files\nodejs;' + $env:PATH
$buildLog = Join-Path $PSScriptRoot 'installer-build.log'
$attempt = 1
while (Test-Path -LiteralPath $buildLog) {
    $buildLog = Join-Path $PSScriptRoot "installer-build-$attempt.log"
    $attempt++
}
Push-Location (Join-Path $taskRoot 'frontend')
try {
    & $env:ComSpec /d /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && npm run desktop:build -- --bundles nsis' *> $buildLog
    if ($LASTEXITCODE -ne 0) { throw "Desktop package failed: $LASTEXITCODE" }
} finally { Pop-Location }
