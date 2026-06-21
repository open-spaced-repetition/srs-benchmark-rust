# SIMULTANEOUS ACT-R timing: before (committed powd) vs after (ln-reuse, a.ln().mul(e).exp()),
# 1 thread each, 200 users, ACT-R --short --secs. Writes ACTR_LNREUSE_DONE when both finish.
$repo = 'C:\Users\Andrew\srs-benchmark-rust'
$data = 'C:\Users\Andrew\anki-revlogs-10k'
$bd = "$repo\_speedup\phase3\actr_ln_before"
$ad = "$repo\_speedup\phase3\actr_ln_after"
New-Item -ItemType Directory -Force "$bd\result", "$ad\result" | Out-Null
Remove-Item "$bd\result\ACT-R-short-secs.jsonl", "$ad\result\ACT-R-short-secs.jsonl" -ErrorAction SilentlyContinue
Remove-Item "$repo\_speedup\phase3\ACTR_LNREUSE_DONE" -ErrorAction SilentlyContinue
$cfg = @('--algo', 'ACT-R', '--short', '--secs', '--data', $data, '--max-user-id', '200', '--processes', '1')
$pb = Start-Process -FilePath "$repo\target\release\script_pc_after.exe" -ArgumentList $cfg -WorkingDirectory $bd -PassThru -WindowStyle Hidden
$pa = Start-Process -FilePath "$repo\target\release\script_actr_lnreuse_after.exe" -ArgumentList $cfg -WorkingDirectory $ad -PassThru -WindowStyle Hidden
$pb.WaitForExit()
$pa.WaitForExit()
New-Item -ItemType File "$repo\_speedup\phase3\ACTR_LNREUSE_DONE" -Force | Out-Null
