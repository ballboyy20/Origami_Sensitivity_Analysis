# lab_queue.ps1 -- launch the weekend runs of 2026-11-14 (see LAB_QUEUE.md).
#
# Opens one PowerShell window per run, each with 2 BLAS threads, in this
# repo's folder. Windows stay open after the run ends so you can read the
# output (results and UTF-8 logs are also saved under results/).
#
#   powershell -ExecutionPolicy Bypass -File .\lab_queue.ps1 short   # runs 1, 2, 6, 7
#   powershell -ExecutionPolicy Bypass -File .\lab_queue.ps1 long    # runs 3, 4, 5, 8
#   powershell -ExecutionPolicy Bypass -File .\lab_queue.ps1 all     # all 8 at once

param([ValidateSet('short', 'long', 'all')][string]$Set = 'short')

$runs = @(
    @{ n = 1; set = 'short'; cmd = 'python -u screen_validation.py small' },
    @{ n = 2; set = 'short'; cmd = 'python -u screen_validation.py replay' },
    @{ n = 3; set = 'long';  cmd = 'python -u screen_validation.py birdsfoot-mixed' },
    @{ n = 4; set = 'long';  cmd = 'python -u screen_validation.py birdsfoot24 --incumbent best' },
    @{ n = 5; set = 'long';  cmd = 'python -u screen_validation.py birdsfoot24 --incumbent champion --max-hours 24' },
    @{ n = 6; set = 'short'; cmd = 'python -u gore_bnb.py single-gore' },
    @{ n = 7; set = 'short'; cmd = 'python -u gore_bnb.py full-gore-allv --max-hours 20' },
    @{ n = 8; set = 'long';  cmd = 'python -u gore_bnb.py full-gore-mixed --max-hours 24' }
)

$setup = "Set-Location '$PSScriptRoot'; " +
         '$env:OMP_NUM_THREADS=2; $env:OPENBLAS_NUM_THREADS=2; $env:MKL_NUM_THREADS=2; '

foreach ($r in $runs) {
    if ($Set -ne 'all' -and $r.set -ne $Set) { continue }
    $title = "`$Host.UI.RawUI.WindowTitle = 'run $($r.n): $($r.cmd)'; "
    Start-Process powershell -ArgumentList '-NoExit', '-Command', ($title + $setup + $r.cmd)
    Write-Host "started run $($r.n): $($r.cmd)"
}
