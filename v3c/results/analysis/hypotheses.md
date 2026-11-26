# nq_v3c confirm hypotheses (definitions fixed in DESIGN_NOTE_V3C.md); cells excluded for < 16 instances: 0; rows 9600; status {'sat': 6498, 'to': 2289, 'unsat': 813}; bad 0

## C1 (PRIMARY) bnq n=200, p in {.84,.86,.88,.90}: solved(ls_nb) - solved(cpsat_strong), paired 95% bootstrap CI; supported if lower bound >= 0.5 in >= 3 of 4 cells.
- p=0.84: diff +1.000 [+1.000, +1.000] (n=20); ls_nb solved 1.00 Wilson [0.84, 1.00]; cpsat_strong 0.00
- p=0.86: diff +1.000 [+1.000, +1.000] (n=20); ls_nb solved 1.00 Wilson [0.84, 1.00]; cpsat_strong 0.00
- p=0.88: diff +1.000 [+1.000, +1.000] (n=20); ls_nb solved 1.00 Wilson [0.84, 1.00]; cpsat_strong 0.00
- p=0.9: diff +1.000 [+1.000, +1.000] (n=20); ls_nb solved 1.00 Wilson [0.84, 1.00]; cpsat_strong 0.00
- VERDICT: supported (4/4 cells)

## C2 (exploratory, core-matched) bnq n=200 cells where ls_nb_w4 solves every instance: cpsat_strong_w4 solves < 0.5, in >= 3 such cells.
- p=0.8: ls_nb_w4 1.00, cpsat_strong_w4 0.00 (n=20)
- p=0.84: ls_nb_w4 1.00, cpsat_strong_w4 0.00 (n=20)
- p=0.86: ls_nb_w4 1.00, cpsat_strong_w4 0.00 (n=20)
- p=0.88: ls_nb_w4 1.00, cpsat_strong_w4 0.00 (n=20)
- p=0.9: ls_nb_w4 1.00, cpsat_strong_w4 0.00 (n=20)
- p=0.92: ls_nb_w4 1.00, cpsat_strong_w4 0.00 (n=20)
- VERDICT: supported (6 of 6 cells)

## C3 (exploratory, replicates H8) SAT-side qwh/qcp cells at N in {30,40}: share with solved(ls_nb) >= 0.8*best complete is >= 0.75, per family.
- FAIL qwh N=30 p=0.35: ls_nb 0.50, best complete 1.00 (cpsat_strong)
- FAIL qwh N=40 p=0.4: ls_nb 0.25, best complete 0.70 (cpsat_strong_w4)
- qwh: 12/14 cells hold
- qcp: 16/16 cells hold
- VERDICT: supported

## C4 (exploratory) qcp SAT-side cells at N=40: best complete solved - solved(ls_nb) >= 0.2 in at least one cell.
- qcp N=40 p=0.3: best complete 1.00 (cadical195), ls_nb 1.00, gap +0.00
- qcp N=40 p=0.35: best complete 1.00 (cadical195), ls_nb 1.00, gap +0.00
- qcp N=40 p=0.38: best complete 1.00 (cadical195), ls_nb 1.00, gap +0.00
- qcp N=40 p=0.4: best complete 1.00 (cadical195), ls_nb 1.00, gap +0.00
- qcp N=40 p=0.42: best complete 1.00 (cadical195), ls_nb 1.00, gap +0.00
- qcp N=40 p=0.44: best complete 0.95 (cadical195), ls_nb 1.00, gap -0.05
- qcp N=40 p=0.46: best complete 0.85 (cadical195), ls_nb 1.00, gap -0.15
- qcp N=40 p=0.5: best complete 0.75 (cpsat_strong_w4), ls_nb 1.00, gap -0.25
- VERDICT: falsified (no qcp N=40 cell with gap >= 0.2)

## C5 (exploratory) ksat n=350: ls_nb solved >= 0.9 at r=4.0 and cadical195 decided >= ls_nb solved at r in {4.26, 4.3}.
- r=4.0: ls_nb 1.00
- r=4.26: cadical195 decided 0.35, ls_nb solved 0.55
- r=4.3: cadical195 decided 0.10, ls_nb solved 0.25
- VERDICT: falsified
