# Where does local search beat complete solvers? Code, data and analysis

Code and raw results for the study "Where Does Local Search Beat Complete Solvers? A Pre-Specified Phase-Transition Comparison on Blocked n-Queens, Quasigroup Completion and Random 3-SAT" (author: Dikshant Bikram Thapa, Kathmandu University). An earlier short version is on Research Square (doi 10.21203/rs.3.rs-11855600/v1).

## Files
- `nq_v2.py`: instance generators (blocked n-queens, quasigroup with holes, quasigroup completion, random 3-SAT), solvers (min-conflicts / WalkSAT local search, CP-SAT with two encodings and a 4-worker run, glucose4, cadical195 through PySAT), independent verifier and the experiment runner (`--stage main|w4`, `--smoke`).
- `nq_v2_cpu.ipynb`: the Kaggle notebook that ran it (4 CPU session, 10 s limit per run).
- `DESIGN_NOTE.md`: the hypotheses H1 to H7, thresholds and analysis plan, written before the main run (not externally time-stamped).
- `results/main.csv` (9,600 runs), `results/w4.csv` (480 runs), `results/smoke.csv` (70-run smoke test): one row per (instance, solver) run.
- `analyze_v2.py`: analysis script. It applies the strict rule: an answer returned after the 10 s limit counts as a timeout (57 late answers were reclassified).
- `analysis/`: tables, hypothesis checks, figures produced by the script.

## Reproduce
`pip install ortools python-sat pandas numpy matplotlib tabulate` then `python nq_v2.py out.csv --stage main --jobs 4 --T 10` and `python analyze_v2.py results/main.csv results/w4.csv --out analysis`.

## Notes
Library versions and CPU of the original Kaggle run were not recorded. Random instances only; local search is untuned Python. Licence: MIT (code), CC BY 4.0 (result tables).
