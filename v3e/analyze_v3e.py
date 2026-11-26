"""Analysis for nq_v3e (definitions = DESIGN_NOTE_V3E.md; constants here must match it). Reuses analyze_v3d.py (unchanged copy).
Usage: python analyze_v3e.py MAIN.csv [--long LONG.csv] [--tuned tuned_cpsat_bnq.json] [--out DIR]   |   python analyze_v3e.py --selftest
MAIN.csv.first.csv (side file written by nq_v3e.py) is read automatically. Outputs DIR/cells.csv, DIR/hypotheses.md."""
import csv, os, sys, json, math, argparse, random, tempfile, statistics
import numpy as np
import analyze_v3d as A
from analyze_v3d import Data, paired, boot_ci, wilson, mcnemar_p, holm, kofm, pn

E2_P = (0.84, 0.88, 0.90, 0.92)                       # primary cells, bnq n=200
LAT = [("qwh", N, h) for N in (30, 40) for h in (0.30, 0.35, 0.40, 0.45)] + [("qcp", N, f) for N in (30, 40) for f in (0.38, 0.42, 0.46, 0.50)]
E4C = [("ksat5", 200, r, "ls_nbk") for r in (20.0, 20.5, 21.0)] + [("ksat", n, r, "ls_nb") for n in (1000, 2000) for r in (4.0, 4.1, 4.2)]
THR = dict(e1_first=0.10, e1_min=0.30, e2_within=0.10, e2_gap=0.5, e3_within=0.15, e3_cells=12, e4_gap=0.10, e4_cells=3, e5_gap=0.10, e5_cells=2, e6_ovh=0.05, e6_ratio=1.5)

def p2(b, c): return min(1.0, 2 * mcnemar_p(max(b, c), min(b, c)))
def frac(D, cell, s): return D.frac(*cell, s)
def fmt(x): return "NA" if x is None else f"{x:.2f}"
def side(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None
def times(D, cell, s):
    return {k: float(r["t"]) for k, r in D.I.get((cell[0], str(cell[1]), pn(cell[2]), s), {}).items() if r["status"] == "sat"}

def e1(D, F):
    rule = f"ls_nb bnq children: median first-kernel-call time <= {THR['e1_first']} s (side file) AND min solved time of ls_nb on bnq n=200 <= {THR['e1_min']} s (v3d: about 1.0 s)"
    if F is None: return dict(id="E1", role="exploratory", rule=rule, verdict="NOT EVALUABLE", p=None, detail=["no side file"])
    fs = [float(r["first_s"]) for r in F if r["family"] == "bnq" and r["solver"] == "ls_nb"]
    tmin = [min(times(D, ("bnq", 200, p), "ls_nb").values(), default=None) for p in (0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96)]; tmin = [t for t in tmin if t is not None]
    if not fs or not tmin: return dict(id="E1", role="exploratory", rule=rule, verdict="NOT EVALUABLE", p=None, detail=["no ls_nb bnq rows"])
    med, mx, mn = statistics.median(fs), max(fs), min(tmin); ok = med <= THR["e1_first"] and mn <= THR["e1_min"]
    return dict(id="E1", role="exploratory", rule=rule, verdict="SUPPORTED" if ok else "FALSIFIED", p=None,
                detail=[f"first-call sum over {len(fs)} children: median {med:.4f} s, max {mx:.4f} s", f"min solved time ls_nb bnq n=200 over cells: {mn:.4f} s"])

def e2(D):
    rule = (f"PRIMARY. bnq n=200, p in {E2_P}: a cell passes if |solved(ls_c) - solved(ls_nb)| <= {THR['e2_within']} AND solved(ls_c) - solved(cpsat_strong) >= {THR['e2_gap']} "
            "(point estimates); supported if >= 3 of 4 cells pass")
    res, det = [], []
    for p in E2_P:
        c = ("bnq", 200, p); a = paired(D, c, "ls_c", "ls_nb"); b = paired(D, c, "ls_c", "cpsat_strong")
        if a is None or b is None: res.append(None); det.append(f"p={p}: not evaluable"); continue
        fc, fn = a[0].mean(), a[1].mean(); fs = b[1].mean(); ok = abs(fc - fn) <= THR["e2_within"] and fc - fs >= THR["e2_gap"]; res.append(ok)
        lo, hi = wilson(int(a[0].sum()), len(a[0])); lo2, hi2 = boot_ci(b[0] - b[1])
        extra = ""
        t = paired(D, c, "ls_c", "cpsat_tuned")
        if t: extra = f"; info vs cpsat_tuned diff {np.mean(t[0] - t[1]):+.2f}"
        det.append(f"p={p}: ls_c {fc:.2f} [{lo:.2f},{hi:.2f}] ls_nb {fn:.2f} cpsat_strong {fs:.2f} cadical195 {fmt(frac(D, c, 'cadical195'))}; ls_c-ls_nb {fc - fn:+.2f} (two-sided exact McNemar p {p2(a[2], a[3]):.3f}); "
                   f"ls_c-cpsat_strong {fc - fs:+.2f} paired CI [{lo2:.2f},{hi2:.2f}] one-sided McNemar p {mcnemar_p(b[2], b[3]):.4g}{extra}; cell passes: {ok}")
    return dict(id="E2", role="PRIMARY", rule=rule, verdict=kofm(res, 3), p=None, detail=det)

def e3(D, L):
    rule = (f"ls_nb (noise 0) vs ls_nb_n10 (noise 0.1) on {len(LAT)} Latin cells (qwh h .30-.45, qcp f .38-.50, N 30 and 40): |solved difference| <= {THR['e3_within']} in >= {THR['e3_cells']} cells "
            "= noise is NOT the driver (speed explains the Latin reversal); otherwise falsified (noise matters)")
    res, det = [], []
    for c in LAT:
        a = paired(D, c, "ls_nb", "ls_nb_n10")
        if a is None: res.append(None); det.append(f"{c}: not evaluable"); continue
        diff = a[0].mean() - a[1].mean(); res.append(abs(diff) <= THR["e3_within"])
        ref = max((s for s in A.COMPLETE if D.frac(*c, s) is not None), key=lambda s: D.frac(*c, s), default=None)
        det.append(f"{c}: ls_nb {a[0].mean():.2f} ls_nb_n10 {a[1].mean():.2f} diff {diff:+.2f} (only-noise0 {a[2]}, only-noise.1 {a[3]}); best complete {ref} {fmt(frac(D, c, ref) if ref else None)}")
    if L is not None:
        det.append("descriptive (not part of the rule): ls_py at T=100 s on seeds 0-4 vs ls_nb_n10 and ls_nb at T=10 s on the same instances")
        for c in LAT:
            lv = L.vec(*c, "ls_py")
            if not lv: continue
            sd = sorted(lv); k = sum(lv.values()); lo, hi = wilson(k, len(lv)); row = f"{c}: ls_py(100 s) {k}/{len(lv)} [{lo:.2f},{hi:.2f}]"
            for s in ("ls_nb_n10", "ls_nb"):
                v = D.vec(*c, s); row += f"; {s}(10 s) {sum(v.get(x, 0) for x in sd)}/{len(sd)}"
            det.append(row)
    return dict(id="E3", role="exploratory (descriptive, no p-value)", rule=rule, verdict=kofm(res, THR["e3_cells"]), p=None, detail=det)

def e4(D):
    rule = (f"direction fixed: probsat_nb ahead of WalkSAT/SKC (ls_nb on 3-SAT, ls_nbk on 5-SAT) by >= +{THR['e4_gap']} in >= {THR['e4_cells']} of the 9 cells "
            "(ksat5 n=200 r 20,20.5,21; 3-SAT n=1000 and 2000, r 4.0,4.1,4.2). Verdict SUPPORTED = WalkSAT = probSAT is NOT supported at these sizes; FALSIFIED = no separation. p = 3rd smallest one-sided McNemar")
    res, ps, det, ceil = [], [], [], 0
    for fam, n, r, w in E4C:
        c = (fam, n, r); a = paired(D, c, "probsat_nb", w)
        if a is None: res.append(None); det.append(f"{c}: not evaluable"); continue
        x, y = a[0].mean(), a[1].mean(); res.append(x - y >= THR["e4_gap"]); ps.append(mcnemar_p(a[2], a[3])); ceil += (x >= .95 and y >= .95)
        anyv = [max(D.vec(*c, s).get(k, 0) for s in ("probsat_nb", w, "cadical195")) for k in D.vec(*c, "probsat_nb")]
        det.append(f"{c}: probsat_nb {x:.2f} {w} {y:.2f} cadical195 {fmt(frac(D, c, 'cadical195'))}; diff {x - y:+.2f} (probSAT-only {a[2]}, WalkSAT-only {a[3]}) McNemar p {ps[-1]:.4f}; solved by any of the three {np.mean(anyv):.2f}")
    det.append(f"cells where both SLS solve >= 0.95 (ceiling, cannot separate): {ceil}")
    return dict(id="E4", role="exploratory", rule=rule, verdict=kofm(res, THR["e4_cells"]), p=sorted(ps)[THR["e4_cells"] - 1] if len(ps) >= THR["e4_cells"] else None, detail=det)

def e5(D, tuned):
    rule = "bnq CP-SAT tuning (8 variants, n=100 p .90 and .94, seeds 2000-2002): SIGNAL if a non-default variant won under the v3d rule; then cpsat_tuned - cpsat_strong >= +0.10 on bnq n=200 in >= 2 of the 4 primary cells (E5b)"
    if tuned is None: return [dict(id="E5", role="exploratory", rule=rule, verdict="NOT EVALUABLE", p=None, detail=["no tuned json"])]
    r = tuned.get("bnq", {}); det = [f"winner {r.get('variant')}; table " + json.dumps({v: x["solved"] for v, x in r.get("table", {}).items()})]
    if not r.get("params"): return [dict(id="E5", role="exploratory", rule=rule, verdict="NO SIGNAL", p=None, detail=det + ["default kept; cpsat_tuned was not run (identical to cpsat_strong)"])]
    res, ps = [], []
    for p in E2_P:
        a = paired(D, ("bnq", 200, p), "cpsat_tuned", "cpsat_strong")
        if a is None: res.append(None); det.append(f"p={p}: not evaluable"); continue
        res.append(a[0].mean() - a[1].mean() >= THR["e5_gap"]); ps.append(mcnemar_p(a[2], a[3])); det.append(f"p={p}: tuned {a[0].mean():.2f} strong {a[1].mean():.2f} McNemar p {ps[-1]:.4f}")
    return [dict(id="E5", role="exploratory", rule=rule, verdict="SIGNAL (variant %s)" % r["variant"], p=None, detail=det[:1]),
            dict(id="E5b", role="exploratory", rule="cpsat_tuned - cpsat_strong >= +0.10 in >= 2 of 4 cells; p = 2nd smallest one-sided McNemar", verdict=kofm(res, THR["e5_cells"]), p=sorted(ps)[1] if len(ps) >= 2 else None, detail=det[1:])]

def e6(D, F):
    rule = f"ls_c process overhead (wall minus internal time, side file) median <= {THR['e6_ovh']} s AND median paired ratio t(ls_c)/t(ls_nb) <= {THR['e6_ratio']} over bnq n=200 instances both solve"
    if F is None: return dict(id="E6", role="exploratory (descriptive, no p-value)", rule=rule, verdict="NOT EVALUABLE", p=None, detail=["no side file"])
    ov = [float(r["first_s"]) for r in F if r["solver"] == "ls_c"]; rat = []
    for p in (0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96):
        a, b = times(D, ("bnq", 200, p), "ls_c"), times(D, ("bnq", 200, p), "ls_nb"); rat += [a[k] / b[k] for k in set(a) & set(b) if b[k] > 0]
    if not ov or not rat: return dict(id="E6", role="exploratory (descriptive, no p-value)", rule=rule, verdict="NOT EVALUABLE", p=None, detail=["too few rows"])
    mo, mr = statistics.median(ov), statistics.median(rat)
    return dict(id="E6", role="exploratory (descriptive, no p-value)", rule=rule, verdict="SUPPORTED" if mo <= THR["e6_ovh"] and mr <= THR["e6_ratio"] else "FALSIFIED", p=None,
                detail=[f"ls_c overhead: {len(ov)} runs, median {mo:.4f} s, max {max(ov):.4f} s", f"time ratio ls_c/ls_nb: {len(rat)} pairs, median {mr:.2f}"])

def report(D, L, F, tuned, out):
    os.makedirs(out, exist_ok=True); bug, cnt, killed = D.checks()
    if L is not None: L.checks()
    with open(os.path.join(out, "cells.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "family", "size", "param", "solver", "n", "solved", "wilson_lo", "wilson_hi", "decided", "par10", "median_t_solved"])
        for tag, X in (("main", D), ("long", L)):
            for (fam, size, param, sv), d in sorted((X.I if X else {}).items()):
                T = float(next(iter(d.values()))["T"]); n = len(d); k = sum(r["status"] == "sat" for r in d.values()); dec = sum(r["status"] in ("sat", "unsat") for r in d.values())
                par = np.mean([float(r["t"]) if r["status"] in ("sat", "unsat") else 10 * T for r in d.values()]); ts = [float(r["t"]) for r in d.values() if r["status"] == "sat"]
                lo, hi = wilson(k, n); w.writerow([tag, fam, size, param, sv, n, k, f"{lo:.3f}", f"{hi:.3f}", dec, f"{par:.2f}", f"{np.median(ts):.3f}" if ts else ""])
    H = [e2(D), e1(D, F), e3(D, L), e4(D), *e5(D, tuned), e6(D, F)]
    ex = [h for h in H if h["role"].startswith("exploratory") and h["p"] is not None]; adj = holm([h["p"] for h in ex])
    for h, a in zip(ex, adj): h["p_holm"] = a
    Ls = ["# nq_v3e results (generated by analyze_v3e.py; verdict rules fixed in DESIGN_NOTE_V3E.md)", "",
          f"rows {len(D.rows)}; status counts {cnt}; killed {killed}; sat/unsat conflicts {len(bug)} {'**STOP: BUG**' if bug else ''}",
          "env: " + "; ".join(f"{e.get('cpu_model')} / ortools {e.get('ortools')} / numba {e.get('numba')} / gcc {e.get('v3e_gcc')} / stage {e.get('stage')}" for e in D.env), ""]
    for h in H:
        v = h["verdict"]
        if "p_holm" in h and v == "SUPPORTED": v = "SUPPORTED (Holm p=%.3f)" % h["p_holm"] if h["p_holm"] < 0.05 else "RULE MET, NOT Holm-significant (p_holm=%.3f)" % h["p_holm"]
        Ls += [f"## {h['id']} [{h['role']}] {v}", f"Rule: {h['rule']}", f"Raw p: {h['p']}" if h["p"] is not None else "", *[f"- {x}" for x in h["detail"]], ""]
    open(os.path.join(out, "hypotheses.md"), "w").write("\n".join(Ls)); print("\n".join(Ls)); return H

def selftest():
    assert p2(5, 0) == 2 / 32 and p2(0, 0) == 1.0
    rg = random.Random(2); tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "m.csv"); lp = os.path.join(tmp, "l.csv"); H = ["stage", "family", "size", "param", "seed", "solver", "workers", "T", "status", "t", "t_raw", "verified", "killed"]
    W = csv.writer(open(path, "w", newline="")); W.writerow(H); WL = csv.writer(open(lp, "w", newline="")); WL.writerow(H)
    def put(Wr, fam, n, p, s, sv, pr, T=10):
        ok = rg.random() < pr; Wr.writerow(["fake", fam, n, p, s, sv, 1, T, "sat" if ok else "to", round(rg.uniform(.01, 5), 3) if ok else T, "", 1 if ok else "", 0])
    P = {"ls_nb": .95, "ls_c": .95, "cpsat_strong": .1, "cadical195": .05, "cpsat_tuned": .5, "ls_nb_n10": .93, "probsat_nb": .95, "ls_nbk": .6}
    for s in range(20):
        for p in (0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96):
            for sv in ("ls_nb", "ls_c", "cpsat_strong", "cadical195", "cpsat_tuned"): put(W, "bnq", 200, p, s, sv, P[sv])
        for fam, n, p in LAT:
            for sv in ("ls_nb", "ls_nb_n10", "cpsat_strong", "cadical195"): put(W, fam, n, p, s, sv, P[sv] if sv != "cpsat_strong" else .9)
        for fam, n, r, w in E4C:
            for sv in ("probsat_nb", w, "cadical195"): put(W, fam, n, r, s, sv, P[sv])
    for s in range(5):
        for fam, n, p in LAT: put(WL, fam, n, p, s, "ls_py", .5, 100)
    F = [dict(family="bnq", size="200", param=".9", seed="0", solver=sv, first_s=str(.005 if sv == "ls_nb" else .003), detail="") for sv in ("ls_nb", "ls_c") for _ in range(10)]
    tuned = {"bnq": {"variant": "lin2", "params": "linearization_level: 2", "table": {"default": {"solved": 1}, "lin2": {"solved": 3}}}}
    Hs = report(Data(path), Data(lp), F, tuned, os.path.join(tmp, "out")); V = {h["id"]: h["verdict"] for h in Hs}
    assert V["E2"] == "SUPPORTED" and V["E1"] == "SUPPORTED" and V["E3"] == "SUPPORTED" and V["E4"] == "SUPPORTED" and V["E5b"] == "SUPPORTED" and V["E6"] == "SUPPORTED", V
    assert V["E5"].startswith("SIGNAL"), V
    rep = open(os.path.join(tmp, "out", "hypotheses.md")).read(); assert "E2 [PRIMARY]" in rep
    print("SELFTEST OK", V)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="?"); ap.add_argument("--long"); ap.add_argument("--tuned"); ap.add_argument("--out", default="analysis_v3e"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    else:
        D = Data(a.csv); report(D, Data(a.long) if a.long else None, side(a.csv + ".first.csv"), json.load(open(a.tuned)) if a.tuned else None, a.out)
