# Hypothesis checks (thresholds from PREREG_NOTE.md; TODO-RESULTS if cells are missing)

## H1 bnq n=200, p in .84-.90: solved(ls) - solved(cpsat_strong), paired 95% CI; supported if lower bound >= 0.5 in at least 3 of 4 cells
- p=0.84: 1.00 [1.00, 1.00] (n=20)
- p=0.86: 1.00 [1.00, 1.00] (n=20)
- p=0.88: 1.00 [1.00, 1.00] (n=20)
- p=0.9: 1.00 [1.00, 1.00] (n=20)
- verdict: supported (4/4 cells)

## H2 family-specific LS advantage: ls vs best complete solver on SAT-side cells at largest size
- qwh n=40 p=0.25: ls 1.00, best complete 1.00
- qwh n=40 p=0.4: ls 0.00, best complete 0.65
- qwh n=40 p=0.45: ls 0.00, best complete 0.80
- qwh n=40 p=0.5: ls 0.00, best complete 1.00
- qwh n=40 p=0.6: ls 0.10, best complete 1.00
- qwh n=40 p=0.7: ls 0.85, best complete 0.95
- qcp n=40 p=0.3: ls 0.70, best complete 1.00
- qcp n=40 p=0.35: ls 0.25, best complete 0.95
- qcp n=40 p=0.38: ls 0.05, best complete 1.00
- qcp n=40 p=0.4: ls 0.05, best complete 1.00
- qcp n=40 p=0.42: ls 0.00, best complete 0.90
- qcp n=40 p=0.44: ls 0.00, best complete 0.90
- qcp n=40 p=0.46: ls 0.00, best complete 0.95
- qcp n=40 p=0.5: ls 0.00, best complete 0.80
- ksat n=350 p=3.8: ls 1.00, best complete 1.00
- ksat n=350 p=4.0: ls 1.00, best complete 1.00

## H3 encoding: pooled solved(cpsat_strong) - solved(cpsat_naive) over SAT-side and transition cells; supported if >= 0.1 for bnq and latin (qwh+qcp) and |.| < 0.1 for ksat
- bnq: 0.156 [0.123, 0.192] (n=480)
- latin: -0.026 [-0.046, -0.007] (n=960)
- ksat: 0.004 [-0.004, 0.013] (n=480)

## H4 UNSAT side: decided fraction by solver, largest sizes (bnq n=100,200 p>=.94; ksat n=250 ratio>=4.4)
- bnq n=100 param>=0.94: cadical195 0.75, cpsat_naive 0.70, cpsat_strong 0.97, cpsat_strong_w4 1.00, glucose4 0.75, ls 0.00
- bnq n=200 param>=0.94: cadical195 0.05, cpsat_naive 0.05, cpsat_strong 0.12, cpsat_strong_w4 0.10, glucose4 0.05, ls 0.00
- ksat n=250 param>=4.4: cadical195 1.00, cpsat_naive 0.95, cpsat_strong 0.97, cpsat_strong_w4 1.00, glucose4 1.00, ls 0.07

## H5 hardness peak: argmax of PAR2(decision) of the best complete solver vs. cell where SAT fraction crosses 0.5, largest size
- bnq n=200: peak at 0.8, SAT-fraction crosses 0.5 near 0.8 (grid index distance 0)
- qwh n=40: peak at 0.4, SAT-fraction crosses 0.5 near 0.3 (grid index distance 2)
- qcp n=40: peak at 0.5, SAT-fraction crosses 0.5 near 0.3 (grid index distance 7)
- ksat n=350: peak at 4.4, SAT-fraction crosses 0.5 near 4.26 (grid index distance 2)

## H6 4 workers, bnq n=200 cells where ls solves all instances (5 seeds): solved(cpsat_strong_w4)
- p=0.8: w4 solved 0.00 (n=5)
- p=0.84: w4 solved 0.00 (n=5)
- p=0.86: w4 solved 0.00 (n=5)
- p=0.88: w4 solved 0.00 (n=5)
- p=0.9: w4 solved 0.00 (n=5)
- p=0.92: w4 solved 0.00 (n=5)

## H7 exploratory: cadical195 decided >= glucose4 in 98% of cells (threshold 75%)
