# Where does local search beat complete solvers? Code, pre-registrations, results and analysis

Code, pre-registered design notes, raw results and analysis scripts for a budget-controlled comparison of local search
(min-conflicts, WalkSAT, probSAT), CP-SAT and CDCL (CaDiCaL, Glucose) on random constraint families
(blocked n-Queens, quasigroup with holes, quasigroup completion, random 3-SAT and 5-SAT) and on a SATLIB subset.

## Stages (each pre-registered by a design note committed before the run; git tags `prereg-*` mark those commits)
| folder | content | pre-registration tag |
|---|---|---|
| `.` (top level) | stage v2: first study, Python local search, 10 s | not externally time-stamped |
| `v3/` | strict-budget re-run with compiled local search (main10, long60), hypotheses H1-H8 | `prereg-v3` |
| `v3c/` | fresh-instance confirmation (seeds 100-119), core-matched best-of-four local search, hypotheses C1-C5 | `prereg-v3c` |
| `v3d/` | robustness: probSAT, tuned CP-SAT, 5-SAT, tabu Latin-square search, SATLIB subset, hypotheses D1-D6 | `prereg-v3d` |
| `v3e/` | fairness: corrected warm-up, native C baseline, noise ablation, WalkSAT versus probSAT at large n, hypotheses E1-E6 | `prereg-v3e` |
| `v3f/` | start-up diagnosis, 5-SAT sanity, slow noise-0 cell, second CP-SAT model, interpreted search on blocked n-Queens, hypotheses F1-F5 | `prereg-v3f` (note commit bcd2781; results in `v3f/results/`) |

Each stage folder holds the design note (`DESIGN_NOTE_*.md`), the harness (`nq_v3*.py`), the analysis script (`analyze_v3*.py`) and, under `results/`,
the raw CSVs (one row per instance and solver), environment JSON files and analysis outputs.
`paper_analysis/` holds the scripts that produced the paper's appendix tables and figures; `diagnostics/` holds the start-up-cost diagnostics.

## Reproduce
Install `ortools python-sat numba numpy pandas matplotlib scipy`, then follow the notebook of the stage (`kaggle_pack` notebooks are generated from the scripts).
Instances are regenerated from string seeds, so no instance files are needed.

## Licence
Code: MIT. Result tables and design notes: CC BY 4.0. SATLIB benchmark archives are not redistributed; the notebook downloads them and records SHA-256 hashes.

## Version history
- v1.0.0: stages v2 to v3e.
- v1.1.0: adds stage v3f results (`v3f/results/`), the figure/table/check scripts of the paper (`paper_analysis/`) and the start-up diagnostics (`diagnostics/`). Tags `prereg-v3` to `prereg-v3e` were created retroactively on 27 Nov 2026 (the commits themselves are older and carry the timing evidence); `prereg-v3f` was created before the v3f run.
