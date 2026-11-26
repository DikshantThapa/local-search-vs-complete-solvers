# Pre-registration note for nq_v2 (written 2026-11-25, BEFORE any non-smoke run)
Only the 70-run smoke test (T = 2 s, tiny grid) had been run when this was written. The main run must not start until this file is saved.
Anything changed after the main run starts goes in the "Deviations" section at the bottom, with date and reason. Never edit the text above it.

## Design (fixed; the code constants in nq_v2.py are the authority)
Families and grids (sizes x parameter values), 20 instances per cell (seeds 0-19), instance seed = string "family-size-param-seed":
- bnq Blocked n-Queens: n in {50,100,200}, p(blocked) in {.80,.84,.86,.88,.90,.92,.94,.96}
- qwh quasigroup with holes (always SAT): N in {20,30,40}, h(holes) in {.25,.30,.35,.40,.45,.50,.60,.70}; Latin square from a Jacobson-Matthews chain
- qcp quasigroup completion (SAT or UNSAT): N in {20,30,40}, f(preassigned, forward-checked) in {.30,.35,.38,.40,.42,.44,.46,.50}
- ksat uniform random 3-SAT: n in {150,250,350}, ratio in {3.8,4.0,4.1,4.2,4.26,4.3,4.4,4.6}
Solvers: ls (min-conflicts / row-permutation min-conflicts / WalkSAT-SKC noise .567), cpsat_naive (Boolean model), cpsat_strong (integer + AllDifferent; AddBoolOr for ksat), glucose4, cadical195 (pysat, pairwise/seqcounter AMO). CP-SAT 1 worker, seed = instance seed.
Budget T = 10 s wall clock per (instance, solver) incl. model/encoding, hard kill at T + 3 s (recorded as timeout). 4 jobs in parallel. Stage w4: cpsat_strong with 4 workers, 5 instances per cell, run alone (jobs = 1).
Every SAT answer is checked by an independent verifier; a failed check is status "bad" and counts as unsolved.

## Metrics (fixed)
- solved = SAT answer verified within T. decided = solved or UNSAT proof within T.
- PAR2 (decision) = mean over instances of time if decided, else 2T. PAR2 (solve) = same with solved only.
- Time-to-solution is right-censored at T; report medians of solved runs only together with the solved fraction, plus Kaplan-Meier style survival curves (censored at T).
- 95% CIs: percentile bootstrap over instances within a cell, 10,000 resamples, numpy seed 12345; paired bootstrap (same resampled instances) for solver differences. No p-values; effects are judged by CI and the thresholds below.
- Cell-level satisfiability label (for "SAT side" and "UNSAT side"): SAT-side cell = at least 90% of its 20 instances solved by some solver; UNSAT-side cell = at least 90% proven UNSAT; other cells = transition region.
- Consistency check: no instance may get SAT from one solver and UNSAT from another. If it happens, stop, treat as a bug, report it.

## Hypotheses and what would falsify them
H1 (easy-region effect replicates for Blocked n-Queens on a second CP-SAT-strong baseline). At n = 200, p in {.84,.86,.88,.90}, solved(ls) - solved(cpsat_strong) >= 0.5 in every cell.
  Falsified if the paired-bootstrap lower 95% bound is below 0.5 in two or more of the four cells.
H2 (the local-search advantage is family-specific). On SAT-side cells of qwh and qcp with the largest N (40), solved(ls) is at least 0.2 below the best complete solver in at least one cell of each family; on SAT-side cells of ksat n = 350 with ratio <= 4.2, ls solves >= 0.9 of instances.
  Falsified if ls is within 0.2 of the best complete solver on all SAT-side qwh (or qcp) N = 40 cells, or solves < 0.9 on ksat n = 350 ratio 4.0.
H3 (encoding matters for global-structure families, not for ksat). Averaged over SAT-side and transition cells, solved(cpsat_strong) - solved(cpsat_naive) is >= 0.1 for bnq and qwh/qcp pooled, and |difference| < 0.1 for ksat.
  Falsified if the pooled bnq or latin difference is < 0.1 (95% CI containing values below 0.1 counts), or the ksat |difference| >= 0.1.
H4 (complete solvers own the UNSAT side; CDCL beats CP-SAT on ksat, CP-SAT-strong beats CDCL on bnq). On UNSAT-side cells with the largest size: decided(cadical195) >= decided(cpsat_strong) + 0.1 for ksat n = 250 ratio >= 4.4 (when any solver decides), and decided(cpsat_strong) >= decided(cadical195) for bnq n = 100 and 200 with p >= .94.
  Falsified if either direction fails in the cells where at least one complete solver decides >= 20% of instances.
H5 (hardness peaks at the transition for every family). For each family and size, the grid value with the highest PAR2 (decision) of the best complete solver is within one grid step of the cell where the SAT fraction crosses 0.5 (ksat n = 350: peak in {4.2, 4.26, 4.3}).
  Falsified if the peak is more than one grid step from the crossing for any family at its largest size.
H6 (extra CP-SAT workers do not close the gap). In bnq n = 200 cells where ls solves all instances, cpsat_strong_w4 solves < 50% (5 instances per cell).
  Falsified if cpsat_strong_w4 solves >= 50% in two or more such cells.
H7 (CDCL variants). cadical195 decides at least as many instances as glucose4 in at least 75% of cells (ties count). Falsified otherwise. This one is exploratory; no claim beyond the sign is made.

## Reporting rules
All seven are reported whether they hold or fail. Cells with fewer than 20 completed instances (time budget) are marked and not used for H-tests unless >= 15 per cell. No cell, solver or family is dropped after seeing results. Extra analyses are labelled exploratory.

## Deviations
(none yet)
