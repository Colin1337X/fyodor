$ErrorActionPreference='Stop'
$taskOutput=Join-Path $PSScriptRoot 'selftest'
if(Test-Path -LiteralPath $taskOutput){throw 'Refusing to overwrite self-test evidence'}
New-Item -ItemType Directory -Path $taskOutput | Out-Null
$adapter='luid_0x00000000_0x00013020'
$other='luid_0x00000000_0x00014526'
$script:testMode='normal'
$script:testStop=Join-Path $taskOutput 'normal.stop'
function Get-Counter {
    param($Counter,$SampleInterval,$MaxSamples)
    if($script:testMode -eq 'error'){throw 'Synthetic PDH failure'}
    if($script:testMode -eq 'stop'){[IO.File]::WriteAllText($script:testStop,'stop')}
    [pscustomobject]@{
        Timestamp=[DateTime]::Now
        CounterSamples=@(
            [pscustomobject]@{InstanceName="pid_123_${adapter}_phys_0_eng_0_engtype_cuda";Status=0;CookedValue=0.0},
            [pscustomobject]@{InstanceName="pid_456_${adapter}_phys_0_eng_1_engtype_3d";Status=5;CookedValue=1.0},
            [pscustomobject]@{InstanceName="pid_789_${other}_phys_0_eng_0_engtype_3d";Status=1;CookedValue=0.5}
        )
    }
}
function Read-Records($path){@(Get-Content -LiteralPath $path | ForEach-Object {$_ | ConvertFrom-Json})}
$normal=Join-Path $taskOutput 'normal.jsonl'
& (Join-Path $PSScriptRoot 'capture.ps1') -OutputPath $normal -StopPath $script:testStop -AdapterLuid $adapter.ToUpperInvariant() -MaximumSamples 2
$records=Read-Records $normal
if($records.Count -ne 3 -or $records[0].adapter_luid -ne $adapter){throw 'Header mismatch'}
$sample=$records[1]
if($sample.target_counters.Count -ne 2 -or $sample.other_positive_engines.Count -ne 1){throw 'Adapter attribution mismatch'}
if($sample.target_counters[0].engine_percent -ne 0 -or $sample.target_counters[1].status -ne 5){throw 'Zero/invalid target data lost'}
if($records[-1].failed -or $records[-1].samples -ne 1){throw 'Normal completion mismatch'}
$hash=(Get-FileHash -LiteralPath $normal).Hash
$refused=$false
try{& (Join-Path $PSScriptRoot 'capture.ps1') -OutputPath $normal -StopPath $script:testStop -AdapterLuid $adapter -MaximumSamples 2}catch{$refused=$true}
if(!$refused -or (Get-FileHash -LiteralPath $normal).Hash -ne $hash){throw 'Overwrite protection failed'}
$script:testMode='stop';$script:testStop=Join-Path $taskOutput 'requested.stop'
$stop=Join-Path $taskOutput 'stop.jsonl'
& (Join-Path $PSScriptRoot 'capture.ps1') -OutputPath $stop -StopPath $script:testStop -AdapterLuid $adapter -MaximumSamples 2
$records=Read-Records $stop
if(!$records[-1].requested_stop -or $records[-1].failed -or $records[-1].samples -ne 1){throw 'Graceful stop failed'}
$script:testMode='error';$script:testStop=Join-Path $taskOutput 'error.stop'
$errorPath=Join-Path $taskOutput 'error.jsonl'
& (Join-Path $PSScriptRoot 'capture.ps1') -OutputPath $errorPath -StopPath $script:testStop -AdapterLuid $adapter -MaximumSamples 2
if($LASTEXITCODE -ne 1){throw 'Counter error must exit nonzero'}
$records=Read-Records $errorPath
if(!$records[-1].failed -or $records[1].kind -ne 'error' -or $records[1].message -ne 'Synthetic PDH failure'){throw 'Counter error not preserved'}
[ordered]@{passed=$true;synthetic_only=$true;checks=@('adapter attribution','uppercase normalization','zero target retention','invalid target retention','other positive retention','terminal record','no overwrite','graceful stop','counter failure')} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskOutput 'result.json') -Encoding utf8
Write-Output 'Nine synthetic monitor checks passed; no GPU counters queried'
exit 0
