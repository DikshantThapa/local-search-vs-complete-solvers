# nq_v3e runtime estimate (4 vCPU, 4 jobs, strict T = 10 s; long3e T = 100 s)
Counts from the grids (python3 selftest_local.py prints them): confirm3e 175 runs per seed x 20 = 3500 (+ 140 cpsat_tuned if E5 finds a signal); long3e 80; tuning 48; smoke 40.
| step | runs | worst case (every run uses its T) | expected |
|---|---|---|---|
| install, gcc check, compile, self-tests, smoke (incl. selftest_local, about 1 min) | 40 | 0.15 h | 0.1 h |
| warm-up (real-size instances incl. qwh N=40 generation, 3-SAT n=2000) | | 0.02 h | 0.01 h |
| tune_v3e.py (48 runs at 5 s) | 48 | 48*5/4 = 0.02 h + model builds | 0.02 h |
| confirm3e | 3,500 (+140) | 3,640*10/4 = 2.5 h, x1.05 fork/generation overhead = 2.7 h | 1.3 to 1.8 h |
| long3e (ls_py, T = 100 s) | 80 | 80*100/4 = 0.56 h | 0.3 to 0.5 h |
| analysis, zip | | 0.05 h | 0.02 h |
Total worst about 3.5 h, expected about 2.2 to 2.8 h. Limit 9 h, so a 2.5x safety margin holds on slower Kaggle CPUs (worst case is bounded by T per run; a slower CPU only reduces the early-finish share). The notebook passes `--hours 7` to confirm3e and `--hours 1.2` to long3e; with the tuning and smoke steps the cap sum is 7 + 1.2 + 0.4 = 8.6 h < 9 h. Expected reasoning: ls_nb, ls_c, ls_nb_n10, ls_nbk, probsat_nb stop at first solution; complete solvers run to T on the hard half of the cells (bnq n=200 p >= .88, 3-SAT n >= 1000 near 4.2, 5-SAT). Memory: bnq n=200 CP-SAT/CNF and ksat n=2000 CNF with 4 jobs are far below 30 GB. Not measured on any machine (no numba/ortools locally); mean per-run figure reuses the v3c/v3d 0.3 to 0.5 s. A truncated CSV is resumable (seed-outermost keeps cells balanced). Trim options if the first session is slow: drop bnq n=50 (3 of 21 bnq cells per size, about 14% of bnq runs) and 3-SAT n=1000; E3's long3e is the first thing to skip (saves 0.5 h) and must then be reported as "skipped, runtime".
