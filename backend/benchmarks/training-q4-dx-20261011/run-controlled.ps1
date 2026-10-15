$ErrorActionPreference = 'Stop'
$controller = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'nvcontainer.exe' -and $_.CommandLine -match 'plugins\\SPUser' }
if (@($controller).Count -ne 1) { throw 'Expected one verified overlay user controller' }
$controller | Select-Object ProcessId,Name,ExecutablePath,CommandLine,CreationDate | ConvertTo-Json | Set-Content D:/fyodor-validation/q4-specialized-controller-20261011.json
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class Q4Overlay { [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(uint a,bool b,uint c); [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h); [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h); }'
$handle = [Q4Overlay]::OpenProcess(0x800,$false,$controller.ProcessId)
if ($handle -eq [IntPtr]::Zero) { throw 'Cannot open overlay controller' }
$suspended = $false
try {
 if ([Q4Overlay]::NtSuspendProcess($handle) -ne 0) { throw 'Suspend failed' }
 $suspended = $true
 $overlays = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'NVIDIA Overlay.exe' -and $_.ExecutablePath -like '*NVIDIA Corporation*NVIDIA App*' }
 @($overlays | Select-Object ProcessId,Name,ExecutablePath,CreationDate) | ConvertTo-Json | Set-Content D:/fyodor-validation/q4-specialized-stopped-overlays-20261011.json
 foreach ($overlay in $overlays) { Stop-Process -Id $overlay.ProcessId -Force }
 & D:/fyodor-validation/pytorch-reference/Scripts/python.exe backend/benchmarks/training-torch-20261011/narrow_compare.py --output D:/fyodor-validation/q4-specialized-controlled-20261011 --before D:/fyodor-validation/fyodor-train-before-q4-specialized-a3ce6be.exe --contexts 8 32 128 --steps 20 --warmup 5 --repeats 2 --monitor-environment
 $benchmarkExit = $LASTEXITCODE
} finally {
 if ($suspended) {
  $resumeStatus = [Q4Overlay]::NtResumeProcess($handle)
  @{ProcessId=$controller.ProcessId;Status=$resumeStatus;Resumed=($resumeStatus -eq 0);Time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content D:/fyodor-validation/q4-specialized-resumed-20261011.json
 }
 [void][Q4Overlay]::CloseHandle($handle)
}
exit $benchmarkExit
