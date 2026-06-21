# SIMULTANEOUS FSRS-5 timing: before (powd, per-row predict) vs after (per-card predict), 1 thread
# each, 200 users, FSRS-5 --short --secs. Confirms the per-card predict win generalizes beyond v6.
$repo = 'C:\Users\Andrew\srs-benchmark-rust'
$data = 'C:\Users\Andrew\anki-revlogs-10k'
$bd = "$repo\_speedup\phase3\fsrs5_before"
$ad = "$repo\_speedup\phase3\fsrs5_after"
New-Item -ItemType Directory -Force "$bd\result", "$ad\result" | Out-Null
Remove-Item "$bd\result\FSRS-5-short-secs.jsonl", "$ad\result\FSRS-5-short-secs.jsonl" -ErrorAction SilentlyContinue
Remove-Item "$repo\_speedup\phase3\FSRS5_TIMING_DONE" -ErrorAction SilentlyContinue
$cfg = @('--algo', 'FSRS-5', '--short', '--secs', '--data', $data, '--max-user-id', '200', '--processes', '1')
$pb = Start-Process -FilePath "$repo\target\release\script_fsrs6_before.exe" -ArgumentList $cfg -WorkingDirectory $bd -PassThru -WindowStyle Hidden
$pa = Start-Process -FilePath "$repo\target\release\script_pc_after.exe" -ArgumentList $cfg -WorkingDirectory $ad -PassThru -WindowStyle Hidden
$pb.WaitForExit()
$pa.WaitForExit()
New-Item -ItemType File "$repo\_speedup\phase3\FSRS5_TIMING_DONE" -Force | Out-Null
