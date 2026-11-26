# nq_v3d runtime estimate (4 vCPU, 4 jobs, strict T = 10 s)
Counts are computed from the grid (per seed: bnq 12 cells x 4 solvers = 48; qwh 8x5 = 40; qcp 6x5 = 30; ksat 12x5 = 60; ksat5 24x4 = 96; total 274) x 20 seeds.
| step | runs | worst case (every run uses T) | expected |
|---|---|---|---|
| tune_cpsat.py cpsat (T 5 s) | 360 | 360*5/4 = 0.13 h + model builds (ksat5/qwh up to a few s, count as part of T) | 0.1 h |
| tune_cpsat.py ls | 96 | 0.03 h | 0.02 h |
| confirm3d | 5,480 | 5,480*10/4 = 3.8 h (+ child fork/generation overhead ~5% = 4.0 h) | 2.0 to 2.8 h (LS and probSAT stop at first solution; CP-SAT/CDCL UNSAT-side and hard cells use T) |
| ext3d | 62 files x 4 = 248 | 0.2 h | 0.1 h |
| numba warm-up, pip, fetch | | 0.1 h | 0.1 h |
Total worst about 4.5 h, expected about 3 h; limit 9 h, so it fits with a 2x safety margin on slower Kaggle CPUs (worst case is bounded by T, a slower CPU only shifts the early-finish share). The notebook passes `--hours 7` to confirm3d. Wall-clock is the sum because the steps are sequential. Memory: bnq n=200 CP-SAT and ksat5 CNF with 4 jobs are well under 30 GB. Not measured on any machine (no numba/ortools locally); estimates reuse the v3c figure of about 0.3 to 0.5 s mean per run.
If a run is cut by --hours the CSV is resumable; seed-outermost order keeps cells balanced. Cut options if the first session is slow: drop ksat5 n=100 and ksat n=250 (saves about 25% of the runs).
