# SIMULTANEOUS Anki + SM2-trainable timing: before (committed, per-row predict) vs after (per-card
# predict), 1 thread each, 200 users, --short --secs. All four run together so each before/after
# pair shares the same machine noise. Writes ANKI_SM2_TIMING_DONE when all finish.
$repo = 'C:\Users\Andrew\srs-benchmark-rust'
$data = 'C:\Users\Andrew\anki-revlogs-10k'
$before = "$repo\target\release\script_pc_after.exe"
$after  = "$repo\target\release\script_anki_sm2_after.exe"
Remove-Item "$repo\_speedup\phase3\ANKI_SM2_TIMING_DONE" -ErrorAction SilentlyContinue
$procs = @()
foreach ($pair in @(@('Anki','anki'), @('SM2-trainable','sm2'))) {
  $algo = $pair[0]; $tag = $pair[1]
  foreach ($side in @(@('before',$before), @('after',$after))) {
    $name = $side[0]; $exe = $side[1]
    $dir = "$repo\_speedup\phase3\${tag}_${name}"
    New-Item -ItemType Directory -Force "$dir\result" | Out-Null
    Get-ChildItem "$dir\result\*.jsonl" -ErrorAction SilentlyContinue | Remove-Item -Force
    $cfg = @('--algo', $algo, '--short', '--secs', '--data', $data, '--max-user-id', '200', '--processes', '1')
    $procs += Start-Process -FilePath $exe -ArgumentList $cfg -WorkingDirectory $dir -PassThru -WindowStyle Hidden
  }
}
$procs | ForEach-Object { $_.WaitForExit() }
New-Item -ItemType File "$repo\_speedup\phase3\ANKI_SM2_TIMING_DONE" -Force | Out-Null
