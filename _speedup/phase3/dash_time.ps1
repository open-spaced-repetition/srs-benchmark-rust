# SIMULTANEOUS DASH timing: before (per-row O(N^2) feature build) vs after (per-card O(N log N)),
# 1 thread each, 200 users. DASH --secs and DASH --short --secs run together (each before/after pair
# shares machine noise). Writes DASH_TIMING_DONE when all finish.
$repo = 'C:\Users\Andrew\srs-benchmark-rust'
$data = 'C:\Users\Andrew\anki-revlogs-10k'
$before = "$repo\target\release\script_actr_lnreuse_after.exe"
$after  = "$repo\target\release\script_dash_after.exe"
Remove-Item "$repo\_speedup\phase3\DASH_TIMING_DONE" -ErrorAction SilentlyContinue
$procs = @()
foreach ($pair in @(@('DASH --secs','dsecs'), @('DASH --short --secs','dssecs'))) {
  $algo = $pair[0]; $tag = $pair[1]
  foreach ($side in @(@('before',$before), @('after',$after))) {
    $name = $side[0]; $exe = $side[1]
    $dir = "$repo\_speedup\phase3\${tag}_${name}"
    New-Item -ItemType Directory -Force "$dir\result" | Out-Null
    Get-ChildItem "$dir\result\*.jsonl" -ErrorAction SilentlyContinue | Remove-Item -Force
    $cfg = @('--algo') + ($algo -split ' ') + @('--data', $data, '--max-user-id', '200', '--processes', '1')
    $procs += Start-Process -FilePath $exe -ArgumentList $cfg -WorkingDirectory $dir -PassThru -WindowStyle Hidden
  }
}
$procs | ForEach-Object { $_.WaitForExit() }
New-Item -ItemType File "$repo\_speedup\phase3\DASH_TIMING_DONE" -Force | Out-Null
