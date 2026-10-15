#Requires -Version 7.0
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$env:FYODOR_TEST_TRAINER=Join-Path $taskRoot 'build-cuda/fyodor-train.exe'
$taskLog=Join-Path $PSScriptRoot 'rust-tests-4.log'
if(Test-Path -LiteralPath $taskLog){throw 'Use a fresh Rust evidence log'}
Push-Location $taskRoot
try {
    & $env:ComSpec /d /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cargo test --lib --jobs 1 --release --manifest-path frontend/src-tauri/Cargo.toml' *> $taskLog
    if($LASTEXITCODE -ne 0){throw "Rust tests failed: $LASTEXITCODE"}
} finally {Pop-Location}
