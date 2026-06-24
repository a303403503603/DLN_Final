$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = 'C:\ProgramData\anaconda3\envs\dl_final\python.exe'
$pinfo.Arguments = '-X utf8 pipeline\discord_bot.py'
$pinfo.WorkingDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true
$pinfo.RedirectStandardOutput = $true
$pinfo.RedirectStandardError = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $pinfo
[void]$p.Start()
Write-Output "PID: $($p.Id)"
