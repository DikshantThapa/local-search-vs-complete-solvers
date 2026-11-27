# nq_v3f runtime estimate (4 vCPU, 4 jobs, strict T = 10 s; long3f T = 100 s; one session, limit planned <= 6 h)
Run counts come from `tasks_for` (python3 selftest_local.py prints them; with a fix 73 per seed in main3f, 66 without): main3f 1,460 (1,220 one-worker + 240 four-worker), lat3f 160, long3f 160 (80 per arm), diag about 70 processes, smoke about 50.
| step | runs | worst case (every run uses its T) | expected |
|---|---|---|---|
| pip install, self-tests (real numba: stubbed-path test reloads the kernels 4 times), smoke (stage + 2-rep diag + fix check) | about 100 | 0.30 h | 0.20 h |
| F1 diagnosis (7 parents, 35 forked + 9 spawned children; spawn pays import + compile about 5-8 s each, serial) | 44 | 0.20 h | 0.10 h |
| main3f one-worker (1,220 runs) | 1,220 | 1,220*10/4 = 0.85 h, x1.05 fork/generation = 0.89 h | 0.30 to 0.40 h |
| main3f four-worker, alone (cpsat_strong_w4, cpsat_alt_w4; 6 cells x 2 x 20) | 240 | 240*10 = 0.67 h | 0.35 to 0.45 h |
| lat3f (ls_nb, ls_nb_n10) | 160 | 160*10/4 = 0.11 h | 0.05 h |
| long3f arm 1 (ls_py_n0, T = 100 s) | 80 | 80*100/4 = 0.56 h | 0.25 to 0.40 h |
| long3f arm 2 (ls_py, T = 100 s) | 80 | 0.56 h | 0.40 to 0.55 h |
| analysis, zip | | 0.05 h | 0.03 h |
Total worst about 3.4 h, expected about 1.8 to 2.3 h. Limit 6 h, so a 1.7x safety margin holds in the worst case (worst case is bounded by T per run; a slower CPU only changes model-build times, which are inside T, and the early-finish share). The notebook caps are `--hours` 2.5 (main3f), 0.3 (lat3f), 0.9 (long3f arm 1), 0.9 (arm 2); with smoke and diag (0.5 h worst) the cap sum is 5.1 h < 6 h, each cap stops launching new runs and lets running ones finish (at most 100 s late). Expected reasoning: ls_nb, ls_nb_fixed, ls_py (n >= 100 bnq), ls_nbk at r <= 18, probsat_nb stop at the first solution; complete solvers run to T on the hard half (bnq n=200 every cell, cadical195 at r = 20, 20.5); ls_py/ls_py_n0 at 100 s use their full T on the unsolved Latin cells (v3e: ls_py solved 29 of 80). Not measured on any machine (no numba/ortools locally); the per-run figures reuse v3c/v3e (0.3 to 0.5 s mean for fast solvers, T for failures). A truncated CSV is resumable (seed-outermost; four-worker runs last: if main3f is cut, F4b/F4d w4 rows are the first to be missing). Trim options if the session is slow: skip long3f arm 2 (F3b then NOT EVALUABLE), then drop `ls_py` from main3f (F5 then NOT EVALUABLE), then the n=100 cells of the four-worker runs.
