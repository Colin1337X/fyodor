param(
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [Parameter(Mandatory=$true)][string]$StopPath,
    [Parameter(Mandatory=$true)][ValidatePattern('^luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}$')][string]$AdapterLuid,
    [ValidateRange(2,3600)][int]$MaximumSamples=900
)
$ErrorActionPreference='Stop'
$AdapterLuid=$AdapterLuid.ToLowerInvariant()
if(Test-Path -LiteralPath $StopPath){throw 'Stop path must not already exist'}
$stream=[IO.File]::Open($OutputPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false))
$writer.AutoFlush=$true
$sampleCount=0
$requestedStop=$false
$failed=$false
$clock=[Diagnostics.Stopwatch]::StartNew()
function Write-Record($record) {
    $writer.WriteLine(($record | ConvertTo-Json -Depth 8 -Compress))
}
try {
    Write-Record ([ordered]@{
        kind='header';schema=1;adapter_luid=$AdapterLuid
        counter_path='\GPU Engine(*)\Utilization Percentage';interval_seconds=1
        maximum_samples=$MaximumSamples;observer_pid=$PID
        timestamp=[DateTime]::UtcNow.ToString('o')
        qpc_seconds=[Diagnostics.Stopwatch]::GetTimestamp()/[double][Diagnostics.Stopwatch]::Frequency
        accepted_benchmark=$false
    })
    Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples $MaximumSamples |
        ForEach-Object {
            $target=[Collections.Generic.List[object]]::new()
            $other=[Collections.Generic.List[object]]::new()
            foreach($counter in $_.CounterSamples) {
                $instance=$counter.InstanceName.ToLowerInvariant()
                $isTarget=$instance.Contains($AdapterLuid)
                $valid=$counter.Status -eq 0 -or $counter.Status -eq 1
                if(!$isTarget -and (!$valid -or $counter.CookedValue -le 0)){continue}
                $processId=$null
                if($instance -match '^pid_(\d+)_'){$processId=[int]$Matches[1]}
                $row=[ordered]@{
                    instance=$instance;process_id=$processId;status=$counter.Status
                    engine_percent=$counter.CookedValue
                }
                if($isTarget){$target.Add($row)}else{$other.Add($row)}
            }
            ++$sampleCount
            Write-Record ([ordered]@{
                kind='sample';index=$sampleCount;timestamp=$_.Timestamp.ToUniversalTime().ToString('o')
                qpc_observed_seconds=[Diagnostics.Stopwatch]::GetTimestamp()/[double][Diagnostics.Stopwatch]::Frequency
                elapsed_seconds=$clock.Elapsed.TotalSeconds
                observer_cpu_seconds=(Get-Process -Id $PID).TotalProcessorTime.TotalSeconds
                all_counter_count=$_.CounterSamples.Count
                target_counters=@($target.ToArray());other_positive_engines=@($other.ToArray())
            })
            if(Test-Path -LiteralPath $StopPath){$requestedStop=$true;throw 'FyodorCounterStopRequested'}
        }
} catch {
    if(!$requestedStop -or $_.Exception.Message -ne 'FyodorCounterStopRequested') {
        $failed=$true
        Write-Record ([ordered]@{kind='error';message=$_.Exception.Message})
    }
} finally {
    Write-Record ([ordered]@{
        kind='end';samples=$sampleCount;requested_stop=$requestedStop;failed=$failed
        timestamp=[DateTime]::UtcNow.ToString('o');elapsed_seconds=$clock.Elapsed.TotalSeconds
    })
    $writer.Dispose()
}
if($failed){exit 1}
