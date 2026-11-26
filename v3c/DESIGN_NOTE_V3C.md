# DESIGN NOTE nq_v3c: confirmatory run on fresh instances (written 2026-11-26, BEFORE any run of this stage)

Why: a referee-style review of the v3 plan found that (a) main10 re-uses the instances (seeds 0-9) and part of the reasoning behind H8 and the local-search tuning after seeing v2, so it is not a fresh confirmatory test; (b) the local search (1 core) was compared with CP-SAT using 1 and 4 workers without a core-matched local-search arm; (c) 10 instances per cell is thin. This stage fixes (a) to (c). Code: `nq_v3c.py` (wraps `nq_v3.py`, unchanged). Nothing in sections 1 to 6 may be edited after the first non-smoke run; changes go under "Deviations".

## 1. Design
- Fresh instances: seeds 100-119 (20 per cell). The CSV column `seed` holds the index 0-19; the generator seed is index + 100. Tuning of the local search used seeds 1000-1002 (v3 note), v2 and v3 main/long used 0-19 at most, so these 20 are unseen by every design decision.
- Grid: the same 96 cells as main10 (bnq 3x8, qwh 3x8, qcp 3x8, ksat 3x8), T = 10 s strict budget, all constants of `nq_v3.py` unchanged (no re-tuning).
- Solvers: `ls_nb` (1 core), `ls_nb_w4` (best of 4 independent `ls_nb` runs, seeds 10*s+i, i=0..3, first verified SAT wins, 4 cores), `cpsat_strong` (1 worker), `cpsat_strong_w4` (4 workers), `cadical195` (1 core). The w4 solvers run alone after all 1-core runs, so `ls_nb_w4` and `cpsat_strong_w4` are core-matched.
- Equal-effort statement: only the local search was tuned (on seeds 1000-1002); CP-SAT and CaDiCaL run with their documented strong or default settings. This asymmetry is a stated limit, not hidden.

## 2. Hypotheses (one primary, the rest exploratory; thresholds fixed here)
Cell labels and "solved" as in DESIGN_NOTE_V3.md section 3, computed on this stage's own data.
- **C1 (PRIMARY, replicates H1 on fresh data).** bnq n=200, p in {.84,.86,.88,.90}: solved(ls_nb) - solved(cpsat_strong), paired 95% interval lower bound >= 0.5 in at least 3 of the 4 cells.
- **C2 (core-matched, exploratory).** bnq n=200 cells where ls_nb_w4 solves every instance: solved(cpsat_strong_w4) < 0.5, in at least 3 of such cells (if fewer than 3 such cells exist the test is "not evaluable").
- **C3 (replicates H8, exploratory).** Over SAT-side cells of qwh and qcp at N in {30, 40}, the share of cells with solved(ls_nb) >= 0.8 * best complete solved is >= 0.75, separately for qwh and qcp.
- **C4 (Latin-square limit, exploratory).** On qcp SAT-side cells at N=40, best complete solved - solved(ls_nb) >= 0.2 in at least one cell (the loss claim made in the paper for the tested local search only).
- **C5 (k-SAT, exploratory).** ksat n=350 SAT-side cells with r <= 4.2: solved(ls_nb) >= 0.9 at r = 4.0 and cadical195 decided >= ls_nb solved at r in {4.26, 4.3}.
All verdicts are reported, none dropped. With 5 hypotheses and one primary, no multiplicity correction is applied to the primary; the exploratory ones are labelled as such and, where p-values are quoted, Holm-adjusted within this list.

## 3. Statistics
Wilson 95% intervals for single proportions (n=20 per cell); paired differences by exact McNemar interval or a paired bootstrap over instances (10000 resamples, seed 0); instance-level survival curves (ECDF of time to solve, censored at T) and PAR10 are reported for the headline cells. Cells with fewer than 16 completed instances are marked and excluded from tests.

## 4. Compute
Expected: about 9,600 runs, 2.5 to 4 h on a 4-vCPU Kaggle CPU session (worst case about 7 h), run with `--hours 7`. Environment JSON is written next to the CSV.

## 5. Known limits stated in advance
Compiled numba local search vs solvers reached through Python wrappers; one machine class; four random families with small sizes; no structured or competition instances; only the local search tuned; hypotheses C2-C5 exploratory.

## Deviations
(none yet)
