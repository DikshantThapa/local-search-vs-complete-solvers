# nq_v3 hypothesis checks (definitions: DESIGN_NOTE_V3.md; every verdict is reported, none dropped)

# Stage main10 (T=10 s)

## H1  bnq n=200, p in {.84,.86,.88,.90}: solved(ls_nb) - solved(cpsat_strong), paired 95% CI. Supported if lower bound >= 0.5 in at least 3 of 4 cells.
- ls_nb p=0.84: +1.000 [+1.000, +1.000] (n=10)
- ls_nb p=0.86: +1.000 [+1.000, +1.000] (n=10)
- ls_nb p=0.88: +1.000 [+1.000, +1.000] (n=10)
- ls_nb p=0.9: +1.000 [+1.000, +1.000] (n=10)
- ls_py p=0.84: +1.000 [+1.000, +1.000] (n=10)
- ls_py p=0.86: +1.000 [+1.000, +1.000] (n=10)
- ls_py p=0.88: +1.000 [+1.000, +1.000] (n=10)
- ls_py p=0.9: +1.000 [+1.000, +1.000] (n=10)
- VERDICT (ls_nb): supported (4/4 cells)

## H2  family-specific LS advantage, on ls_py (v2 continuity). Falsified if ls_py is within 0.2 of the best complete solver on all SAT-side qwh (or all qcp) N=40 cells, or solves < 0.9 on ksat n=350 r=4.0. ls_nb shown for information (H8 is its test).
- qwh n=40 p=0.25: ls_py 1.00, ls_nb 1.00, best complete 1.00
- qwh n=40 p=0.4: ls_py 0.00, ls_nb 0.20, best complete 0.90
- qwh n=40 p=0.45: ls_py 0.00, ls_nb 1.00, best complete 0.70
- qwh n=40 p=0.5: ls_py 0.00, ls_nb 1.00, best complete 0.90
- qwh n=40 p=0.6: ls_py 0.20, ls_nb 1.00, best complete 0.90
- qwh n=40 p=0.7: ls_py 0.80, ls_nb 1.00, best complete 1.00
- qcp n=40 p=0.3: ls_py 0.50, ls_nb 1.00, best complete 1.00
- qcp n=40 p=0.35: ls_py 0.30, ls_nb 1.00, best complete 1.00
- qcp n=40 p=0.38: ls_py 0.00, ls_nb 1.00, best complete 1.00
- qcp n=40 p=0.4: ls_py 0.00, ls_nb 1.00, best complete 1.00
- qcp n=40 p=0.42: ls_py 0.00, ls_nb 1.00, best complete 0.90
- qcp n=40 p=0.44: ls_py 0.00, ls_nb 1.00, best complete 1.00
- qcp n=40 p=0.46: ls_py 0.00, ls_nb 1.00, best complete 0.90
- qcp n=40 p=0.5: ls_py 0.00, ls_nb 1.00, best complete 0.70
- ksat n=350 p=3.8: ls_py 1.00, ls_nb 1.00, best complete 1.00
- ksat n=350 p=4.0: ls_py 1.00, ls_nb 1.00, best complete 1.00
- ksat n=350 p=4.2: ls_py 0.80, ls_nb 0.90, best complete 0.40
- latin part (gap >= 0.2 in >= 1 SAT-side cell, qwh, qcp): [True, True]; ksat n=350 r=4.0 ls_py solved: 1.0
- VERDICT: supported

## H3  encoding: pooled solved(cpsat_strong) - solved(cpsat_naive) over SAT-side + transition cells (UNSAT-side cells excluded, labels from this run). Supported if bnq lower CI >= 0.1, latin (qwh+qcp) lower CI >= 0.1, and ksat CI inside (-0.1, 0.1).
- bnq: +0.206 [+0.147, +0.271] (n=170); cells used 17
- latin: -0.035 [-0.065, -0.006] (n=480); cells used 48
- ksat: -0.005 [-0.029, +0.014] (n=210); cells used 21
- VERDICT: falsified

## H4  UNSAT side. ksat n=250, ratio >= 4.4: decided(cadical195) >= decided(cpsat_strong) + 0.1; bnq n=100 and 200, p >= .94: decided(cpsat_strong) >= decided(cadical195). Only cells where some complete solver decides >= 20%; supported if it holds in every such cell.
- ksat n=250 p=4.4: cadical195 1.00, cpsat_strong 0.70 -> holds
- ksat n=250 p=4.6: cadical195 1.00, cpsat_strong 1.00 -> fails
- bnq n=100 p=0.94: cadical195 0.70, cpsat_strong 0.90 -> holds
- bnq n=100 p=0.96: cadical195 1.00, cpsat_strong 1.00 -> holds
- bnq n=200 p=0.94: skipped (max decided 0.00 < 0.2)
- bnq n=200 p=0.96: cadical195 0.10, cpsat_strong 0.20 -> holds
- VERDICT: falsified

## H5  hardness peak. Per family at the largest size: s(param) = share of instances solved by some solver. A crossing exists only if min s < 0.5 <= max s over the grid; otherwise NOT EVALUABLE (qwh is always SAT by construction: declared not evaluable in advance). Peak = argmax over param of min over complete solvers of PAR2(decision); NOT EVALUABLE if that max is >= 0.95*2T in more than 2 cells (plateau). Crossing cell = grid point with s closest to 0.5. Supported if |peak index - crossing index| <= 1 (ksat n=350: peak in {4.2, 4.26, 4.3}); H5 as a whole supported if every evaluable family is supported and >= 2 families are evaluable.
- bnq n=200: NOT EVALUABLE (PAR2 plateau at 2T); s range [0.00, 1.00], peak 0.84
- qwh n=40: NOT EVALUABLE (always SAT by construction); s range [0.70, 1.00], peak 0.45
- qcp n=40: NOT EVALUABLE (no 0.5 crossing in grid); s range [1.00, 1.00], peak 0.5
- ksat n=350: peak 4.4, crossing 4.26, distance 2 -> fails
- VERDICT: falsified (1 families evaluable)

## H7 (exploratory)  cadical195 decided >= glucose4 in >= 75% of cells (ties count).
- 99% of 96 cells
- VERDICT: supported

## H8 (main10)  tuned compiled LS closes the Latin-square gap. Over SAT-side cells of a family at N in {30,40}: share of cells with solved(ls_nb) >= 0.8 * best complete solved is >= 0.75. Supported if true for both qwh and qcp; partial if one; falsified if neither.
- qwh N=30 p=0.25: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.3: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.35: ls_nb 0.80, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.4: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.45: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.5: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.6: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.7: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=40 p=0.25: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=40 p=0.4: ls_nb 0.20, best complete 0.90 (cadical195) -> fails
- qwh N=40 p=0.45: ls_nb 1.00, best complete 0.70 (cadical195) -> holds
- qwh N=40 p=0.5: ls_nb 1.00, best complete 0.90 (cpsat_strong) -> holds
- qwh N=40 p=0.6: ls_nb 1.00, best complete 0.90 (cadical195) -> holds
- qwh N=40 p=0.7: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh: 13/14 SAT-side cells hold
- qcp N=30 p=0.3: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.35: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.38: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.4: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.42: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.44: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.46: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.5: ls_nb 0.90, best complete 0.90 (cadical195) -> holds
- qcp N=40 p=0.3: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.35: ls_nb 1.00, best complete 1.00 (cpsat_naive) -> holds
- qcp N=40 p=0.38: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.4: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.42: ls_nb 1.00, best complete 0.90 (cpsat_naive) -> holds
- qcp N=40 p=0.44: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.46: ls_nb 1.00, best complete 0.90 (cadical195) -> holds
- qcp N=40 p=0.5: ls_nb 1.00, best complete 0.70 (cadical195) -> holds
- qcp: 16/16 SAT-side cells hold
- VERDICT: supported

# Stage long60 (T=60 s)

## H6 (long60)  bnq n=200 cells where ls_nb solves every instance: cpsat_strong_w4 solves < 50%. Falsified if w4 solves >= 50% in two or more such cells.
- p=0.84: ls_nb 1.00 (n=5), w4 solved 0.40 (n=5)
- p=0.88: ls_nb 1.00 (n=5), w4 solved 0.00 (n=5)
- p=0.92: ls_nb 1.00 (n=5), w4 solved 0.00 (n=5)
- VERDICT: supported (0 of 3 cells with w4 >= 50%)

## H8 (long60)  tuned compiled LS closes the Latin-square gap. Over SAT-side cells of a family at N in {30,40}: share of cells with solved(ls_nb) >= 0.8 * best complete solved is >= 0.75. Supported if true for both qwh and qcp; partial if one; falsified if neither.
- qwh N=30 p=0.35: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.45: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=30 p=0.55: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=40 p=0.35: ls_nb 0.00, best complete 1.00 (cadical195) -> fails
- qwh N=40 p=0.45: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh N=40 p=0.55: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qwh: 5/6 SAT-side cells hold
- qcp N=30 p=0.35: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=30 p=0.42: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.35: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.42: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp N=40 p=0.5: ls_nb 1.00, best complete 1.00 (cadical195) -> holds
- qcp: 5/5 SAT-side cells hold
- VERDICT: supported
