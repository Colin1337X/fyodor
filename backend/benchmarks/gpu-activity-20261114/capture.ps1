param(
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [ValidateRange(2,60)][int]$Samples=5
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'Refusing to overwrite GPU activity evidence' }
$clock=[Diagnostics.Stopwatch]::StartNew()
$rows=[Collections.Generic.List[object]]::new()
Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples $Samples |
    ForEach-Object {
        $active=[Collections.Generic.List[object]]::new()
        $invalid=[Collections.Generic.List[object]]::new()
        $valid=0
        foreach ($counter in $_.CounterSamples) {
            if ($counter.Status -ne 0 -and $counter.Status -ne 1) {
                $invalid.Add([pscustomobject]@{instance=$counter.InstanceName;status=$counter.Status})
                continue
            }
            ++$valid
            if ($counter.CookedValue -le 0) { continue }
            $processId=$null
            $processName=$null
            if ($counter.InstanceName -match '^pid_(\d+)_') {
                $processId=[int]$Matches[1]
                $process=Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($process) { $processName=$process.ProcessName }
            }
            $active.Add([pscustomobject]@{
                instance=$counter.InstanceName
                process_id=$processId
                process_name=$processName
                engine_percent=$counter.CookedValue
                status=$counter.Status
            })
        }
        $rows.Add([pscustomobject]@{
            timestamp=$_.Timestamp.ToUniversalTime().ToString('o')
            observer_elapsed_seconds=$clock.Elapsed.TotalSeconds
            valid_counter_count=$valid
            invalid_counters=@($invalid.ToArray())
            positive_engines=@($active.ToArray())
        })
    }
[pscustomobject]@{
    accepted_benchmark=$false
    note='Per-engine diagnostic only. Do not sum engine percentages or infer clean execution from missing/invalid instances. First sample may not cover a full interval. PID names are best-effort observations, not durable process identity. All adapters are included; no NVIDIA-only attribution is claimed.'
    counter_path='\GPU Engine(*)\Utilization Percentage'
    sample_interval_seconds=1
    samples=@($rows.ToArray())
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
