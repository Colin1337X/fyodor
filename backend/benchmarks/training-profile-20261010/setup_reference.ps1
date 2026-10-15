# Reference-only environment: never linked into or shipped with Fyodor.
$ErrorActionPreference = 'Stop'
$reference = 'D:\fyodor-validation\pytorch-reference'
if (Test-Path -LiteralPath $reference) { throw 'Inspect the existing reference environment before replacing it' }
py -3.14 -m venv $reference
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
New-Item -ItemType Directory -Path 'D:\fyodor-validation\pytorch-tmp' -Force | Out-Null
$env:TEMP = 'D:\fyodor-validation\pytorch-tmp'
$env:TMP = $env:TEMP
& "$reference\Scripts\python.exe" -m pip install --no-cache-dir 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --report "$PSScriptRoot/torch-install.json" *> "$PSScriptRoot/torch-install.log"
exit $LASTEXITCODE
