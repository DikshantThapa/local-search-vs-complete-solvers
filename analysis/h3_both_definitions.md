# H3 pooled solved(cpsat_strong) - solved(cpsat_naive), strict data (answers after 10 s = timeouts), paired bootstrap 95% CI, 10,000 resamples, numpy seed 12345 (fresh generator, order bnq, latin, ksat)
Cell sets. "all cells" = every cell of the family (what analyze_v2.py did). "prereg" = SAT-side and transition cells only (PREREG_NOTE.md), UNSAT-side cells (>=90% proven UNSAT) excluded: 7 bnq cells and 2 ksat cells; qwh/qcp have none.
| family | all cells | prereg definition |
|---|---|---|
| bnq | +0.156 [0.125, 0.190] n=480 | +0.221 [0.176, 0.268] n=340 |
| latin (qwh+qcp) | -0.026 [-0.045, -0.006] n=960 | -0.026 [-0.045, -0.006] n=960 |
| ksat | +0.004 [-0.004, 0.013] n=480 | +0.005 [-0.005, 0.014] n=440 |
Other paired solved differences (cadical195 minus cpsat_strong, all cells of family): pooled +0.001 [-0.014, 0.016] n=1920; bnq -0.179 [-0.215, -0.146]; qwh +0.052 [0.027, 0.079]; qcp +0.081 [0.054, 0.108]; ksat +0.048 [0.027, 0.071] (n=480 each).
Script: /tmp/nq3/b.py (uses analyze_v2.paired).
