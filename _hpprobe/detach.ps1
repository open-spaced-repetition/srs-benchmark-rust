param([string]$Cmd, [string]$Log)
$line = "cmd.exe /c `"$Cmd`" > `"$Log`" 2>&1"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$line}
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
