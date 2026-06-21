# SIMULTANEOUS ACT-R timing: before vs after, 1 thread each, 200 users, ACT-R --short --secs.
# Detached (survives interrupts, unlike the Bash background tool). Writes ACTR_TIMING_DONE when both finish.
$repo = 'C:\Users\Andrew\srs-benchmark-rust'
$data = 'C:\Users\Andrew\anki-revlogs-10k'
$bd = "$repo\_speedup\phase2\actr_before"
$ad = "$repo\_speedup\phase2\actr_after"
New-Item -ItemType Directory -Force "$bd\result", "$ad\result" | Out-Null
Remove-Item "$bd\result\ACT-R-short-secs.jsonl", "$ad\result\ACT-R-short-secs.jsonl" -ErrorAction SilentlyContinue
Remove-Item "$repo\_speedup\phase2\ACTR_TIMING_DONE" -ErrorAction SilentlyContinue
$cfg = @('--algo', 'ACT-R', '--short', '--secs', '--data', $data, '--max-user-id', '200', '--processes', '1')
$pb = Start-Process -FilePath "$repo\target\release\script_actr_before.exe" -ArgumentList $cfg -WorkingDirectory $bd -PassThru -WindowStyle Hidden
$pa = Start-Process -FilePath "$repo\target\release\script_actr_after.exe" -ArgumentList $cfg -WorkingDirectory $ad -PassThru -WindowStyle Hidden
$pb.WaitForExit()
$pa.WaitForExit()
New-Item -ItemType File "$repo\_speedup\phase2\ACTR_TIMING_DONE" -Force | Out-Null
