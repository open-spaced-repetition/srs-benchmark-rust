# 16-experiment HDBSCAN smart-preset sweep on 1000 users, 100-user chunks, 2 threads.
# Uses script_hdbscan.exe (snapshot with the validated HDBSCAN code).
Set-Location "C:\Users\Andrew\srs-benchmark-rust"
foreach ($M in 100,200,300,400,500,600,700,800,900,1000) {
  Write-Output "=== chunk max-user-id=$M @ $(Get-Date -Format HH:mm:ss) ==="
  & ".\target\release\script_hdbscan.exe" --algo FSRS-7 --short --secs --partitions smart --cluster_method hdbscan --cluster_sweep --data "C:\Users\Andrew\anki-revlogs-10k" --processes 2 --max-user-id $M
}
Write-Output "=== ALL DONE @ $(Get-Date -Format HH:mm:ss) ==="
