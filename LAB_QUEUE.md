# Lab computer queue — weekend of 2026-11-14

Eight independent runs: the one-restart screen validation
(`screen_validation.py`, KINEMATIC_COUPLING_SUMMARY.md §7.15) and the gore
experiments for Days 3 and 6 (`gore_bnb.py`). Lab machine: 14 cores, 20
logical processors; each run uses 2 BLAS threads.

## 0. Before starting

```powershell
cd C:\Users\craftyb\Documents\GitHub\Origami_Sensitivity_Analysis
git checkout Kinematic-Coupling-Placement
git pull
```

Make sure the machine won't sleep or reboot for updates in the next ~36 h
(Settings → System → Power: sleep "Never"; Windows Update → Pause updates).
A sleep pauses every run; a reboot kills them.

## 1. Why separate windows

Each `python` command runs in the foreground and holds its window until it
finishes, so commands typed one after another in the *same* window run in
sequence. Running each one in its *own* PowerShell window makes them separate
processes that run **at the same time**, each on its own cores. The
`$env:..._NUM_THREADS=2` line caps each process at 2 threads so 7 runs fit in
14 cores without fighting over them. It only applies to the window it's
typed in, which is why every window needs it.

Closing a window kills its run. Leave them open (minimizing is fine).

## 2. Easy way: the launcher

From the repo folder, this opens one window per run with the setup already
done:

```powershell
powershell -ExecutionPolicy Bypass -File .\lab_queue.ps1 short   # runs 1, 2, 6, 7: start now
powershell -ExecutionPolicy Bypass -File .\lab_queue.ps1 long    # runs 3, 4, 5, 8: start when leaving
```

Windows stay open after a run ends so its output can be read; close them
once read.

## 3. Manual way: one window per run

Open a new PowerShell window for each run and paste:

```powershell
cd C:\Users\craftyb\Documents\GitHub\Origami_Sensitivity_Analysis
$env:OMP_NUM_THREADS=2; $env:OPENBLAS_NUM_THREADS=2; $env:MKL_NUM_THREADS=2
```

then the run's command:

| # | start | command | expected | question |
|---|---|---|---|---|
| 1 | now | `python -u screen_validation.py small` | ~15 min | screen regression on small systems |
| 2 | now | `python -u screen_validation.py replay` | ~40 min | would the screen have lost any incumbent-raising leaf in the 9.6 h run? |
| 6 | now | `python -u gore_bnb.py single-gore` | ~1–2 h | Day 3: B&B vs. exhaustive on the single gore |
| 7 | now | `python -u gore_bnb.py full-gore-allv --max-hours 20` | sweep minutes; B&B ≤ 20 h | Day 6: does any 42-row all-V design exist, and which is best? |
| 3 | leaving | `python -u screen_validation.py birdsfoot-mixed` | ~4–5 h | the 9.6 h mixed run, screened: identical search, cheaper? |
| 4 | leaving | `python -u screen_validation.py birdsfoot24 --incumbent best` | < 6 h | 24-site run vs. the 6.2 h unscreened run |
| 5 | leaving | `python -u screen_validation.py birdsfoot24 --incumbent champion --max-hours 24` | ≤ 24 h | does the run that didn't finish in 8 h finish now? |
| 8 | leaving | `python -u gore_bnb.py full-gore-mixed --max-hours 24` | ≤ 24 h | Day 6, mixed types: a valid 42-row design, and how good |

If run 1 or 2 reports FAIL, stop 3, 4 and 5 (close their windows): their
timings no longer mean much until the screen is fixed.

## 4. Where results land

All in `results/`:

- `screen_validation_<part>_{summary.txt,results.json}` and `screen_validation_<part>.log`
- `gore_bnb_<part>_{summary.txt,results.json}` and `gore_bnb_<part>.log`

Logs are UTF-8 and written as the run goes, so progress can be checked
any time. When done:

```powershell
git add results/screen_validation_* results/gore_bnb_*
git commit -m "Screen validation and gore runs"
git push
```

## 5. Reading the results

- Timings won't match earlier runs exactly (more processes share the
  machine). Runs 3 and 4 also check that the search is *identical* (nodes,
  pruning counts, certified bound), which doesn't depend on load.
- Runs 7 and 8 use the screen before runs 1–2 have validated it. The
  certificate is sound regardless; the only risk is a slightly worse
  incumbent, limited by the end-of-run recheck.
- Run 8 probably won't finish; its first full-rank design already answers
  Day 6.
