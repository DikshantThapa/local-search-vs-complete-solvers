"""Analysis for nq_v3f (definitions = DESIGN_NOTE_V3F.md; constants here must match it). Reuses analyze_v3d.py (unchanged copy) and diag_f1.py (summary only).
Usage: python analyze_v3f.py MAIN.csv [--lat LAT.csv] [--long LONG.csv] [--diag f1_diag.csv] [--fix f1_fix.json] [--types f1_types.json] [--out DIR]
       python analyze_v3f.py --selftest
MAIN.csv.first.csv (side file written by nq_v3f.py) is read automatically. Outputs DIR/cells.csv, DIR/hypotheses.md.
No p-value is reported for any 'k of n cells' or pooled rule (CRITIC3_v8: the order-statistic p is invalid); only descriptive per-comparison exact McNemar p (unadjusted) appears in detail lines."""
import csv, os, sys, json, math, argparse, random, tempfile, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v3d import Data, paired, wilson, mcnemar_p, kofm, pn
import diag_f1

PRIM = (0.84, 0.88, 0.90, 0.92)                                  # F1 primary cells: bnq n=200
LAT = [("qwh", N, h) for N in (30, 40) for h in (0.30, 0.35, 0.40, 0.45)] + [("qcp", N, f) for N in (30, 40) for f in (0.38, 0.42, 0.46, 0.50)]
K5_LOW, K5_HI = (12.0, 14.0, 16.0, 18.0), (20.0, 20.5)
NOISE_SOLVER = {0.3: "ls_nbk_n30", 0.4: "ls_nbk_n40", 0.5: "ls_nbk_n50", 0.567: "ls_nbk", 0.65: "ls_nbk_n65"}
BNQ6 = [(n, p) for n in (100, 200) for p in (0.84, 0.88, 0.92)]
MIN_POOL = 60                                                    # F3: paired (cell, seed) instances needed (of 80)
THR = dict(f1_min_t=0.30, f1_within=0.05, f1b_first=0.10, f2a_solved=0.95, f2b_solved=0.50, f3a_slack=0.10, f3b_gap=0.15, f4_gap=0.5, f5_gap=0.5, f5_within=0.10)
EPS = 1e-9

def p2(b, c): return min(1.0, 2 * mcnemar_p(max(b, c), min(b, c)))
def fmt(x): return "NA" if x is None else f"{x:.2f}"
def side(path): return list(csv.DictReader(open(path))) if path and os.path.exists(path) else None
def times(D, cell, s): return {k: float(r["t"]) for k, r in D.I.get((cell[0], str(cell[1]), pn(cell[2]), s), {}).items() if r["status"] == "sat"}
def wl(k, n): lo, hi = wilson(k, n); return f"{k}/{n} [{lo:.2f},{hi:.2f}]"
def H(id_, role, rule, verdict, detail): return dict(id=id_, role=role, rule=rule, verdict=verdict, p=None, detail=detail)

# ---------------------------------------------------------------- F1
def f1(D, fixinfo, F, diag, types):
    rule = (f"PRIMARY. A fix exists (diag_f1 rule picked one) AND min solved time of ls_nb_fixed over bnq n=200, p in {PRIM} <= {THR['f1_min_t']} s AND "
            f"|solved(ls_nb_fixed) - solved(ls_nb)| <= {THR['f1_within']} in each of the 4 cells (point estimates). Descriptive verdict, no p-value")
    det = []
    if diag:
        S = diag_f1.summarize(diag)
        for k, v in sorted(S.items()): det.append(f"diag {k[0]}/{k[1]}: n={v['n']} first-call median {v['median']:.4f} s (min {v['min']:.4f}, max {v['max']:.4f}), 2nd-call median {v['median_second']:.4f}, compile events {v['median_compile_events']}, signature growth {v['median_sig_growth']}")
    if types: det.append("argument types identical across the n=8 warm-up board and the n=200 board: %s" % types.get("all_types_identical_across_boards"))
    if fixinfo is None: return H("F1", "PRIMARY", rule, "NOT EVALUABLE", det + ["no f1_fix.json"])
    det.append(f"diag fix rule result: fix={fixinfo.get('fix')}, status-quo (warm_real) median {fixinfo.get('baseline_median')}, reproduced {fixinfo.get('baseline_reproduced')}")
    if fixinfo.get("fix") is None: return H("F1", "PRIMARY", rule, "FALSIFIED", det + ["no registered fix reached the first-call threshold: no fix exists among same_call, sig, cache"])
    res, tmin = [], []
    for p in PRIM:
        c = ("bnq", 200, p); a = paired(D, c, "ls_nb_fixed", "ls_nb")
        if a is None: res.append(None); det.append(f"p={p}: not evaluable"); continue
        diff = a[0].mean() - a[1].mean(); t = times(D, c, "ls_nb_fixed"); tm = min(t.values()) if t else None; tmin += [tm] if tm is not None else []
        res.append(abs(diff) <= THR["f1_within"] + EPS)
        det.append(f"p={p}: ls_nb_fixed {wl(int(a[0].sum()), len(a[0]))} ls_nb {wl(int(a[1].sum()), len(a[1]))} diff {diff:+.2f} (only-fixed {a[2]}, only-ls_nb {a[3]}); min solved t fixed {fmt(tm)} s, ls_nb {fmt(min(times(D, c, 'ls_nb').values(), default=None))} s; within: {res[-1]}")
    if any(r is None for r in res) or not tmin: return H("F1", "PRIMARY", rule, "NOT EVALUABLE", det)
    mn = min(tmin); det.append(f"min solved time of ls_nb_fixed over the 4 cells: {mn:.4f} s")
    return H("F1", "PRIMARY", rule, "SUPPORTED" if mn <= THR["f1_min_t"] + EPS and all(res) else "FALSIFIED", det)

def f1b(F):
    rule = f"exploratory: median first-kernel-call sum of ls_nb_fixed bnq children <= {THR['f1b_first']} s (side file); ls_nb reported for comparison"
    if F is None: return H("F1b", "exploratory", rule, "NOT EVALUABLE", ["no side file"])
    g = lambda s: [float(r["first_s"]) for r in F if r["family"] == "bnq" and r["solver"] == s]
    a, b = g("ls_nb_fixed"), g("ls_nb")
    if not a: return H("F1b", "exploratory", rule, "NOT EVALUABLE", ["no ls_nb_fixed rows"])
    ma = statistics.median(a); det = [f"ls_nb_fixed: {len(a)} children, median {ma:.4f} s, max {max(a):.4f} s"] + ([f"ls_nb: {len(b)} children, median {statistics.median(b):.4f} s, max {max(b):.4f} s"] if b else [])
    return H("F1b", "exploratory", rule, "SUPPORTED" if ma <= THR["f1b_first"] + EPS else "FALSIFIED", det)

# ---------------------------------------------------------------- F2
def f2a(D):
    rule = f"exploratory: ls_nbk (noise 0.567) solves >= {THR['f2a_solved']} of the 20 instances in ALL 4 cells ksat5 n=200 r = 12, 14, 16, 18 (far below the 5-SAT threshold) = implementation is sane. No p-value"
    res, det = [], []
    for r in K5_LOW:
        c = ("ksat5", 200, r); v = D.vec(*c, "ls_nbk"); f = D.frac(*c, "ls_nbk")
        if f is None: res.append(None); det.append(f"r={r}: not evaluable"); continue
        res.append(f >= THR["f2a_solved"] - EPS)
        det.append(f"r={r}: ls_nbk {wl(sum(v.values()), len(v))}; probsat_nb {fmt(D.frac(*c, 'probsat_nb'))}; cadical195 {fmt(D.frac(*c, 'cadical195'))}; passes: {res[-1]}")
    return H("F2a", "exploratory", rule, kofm(res, 4), det)

def f2b(D):
    rule = (f"exploratory (descriptive; best-of-5 is selection-inflated): at r = 20 or 20.5 (ksat5 n=200) the best of the 5 WalkSAT noise settings {sorted(NOISE_SOLVER)} solves >= {THR['f2b_solved']} "
            "in >= 1 of the 2 cells = the v3e 0/60 was a noise-setting artefact. No p-value")
    res, det = [], []
    for r in K5_HI:
        c = ("ksat5", 200, r); fr = {z: D.frac(*c, s) for z, s in NOISE_SOLVER.items()}; ok = {z: f for z, f in fr.items() if f is not None}
        if not ok: res.append(None); det.append(f"r={r}: not evaluable"); continue
        zb = max(sorted(ok), key=lambda z: ok[z]); res.append(ok[zb] >= THR["f2b_solved"] - EPS)
        row = "; ".join(f"noise {z}: {wl(sum(D.vec(*c, NOISE_SOLVER[z]).values()), len(D.vec(*c, NOISE_SOLVER[z])))}" for z in sorted(ok))
        det.append(f"r={r}: {row}; best noise {zb} ({ok[zb]:.2f}); probsat_nb {fmt(D.frac(*c, 'probsat_nb'))}; cadical195 {fmt(D.frac(*c, 'cadical195'))}; passes: {res[-1]}")
    return H("F2b", "exploratory", rule, kofm(res, 1), det)

# ---------------------------------------------------------------- F3
def pool(Ds, s):
    """{(cell, seed): 0/1} for solver s over the 16 Latin cells; Ds = list of Data objects holding that solver"""
    out = {}
    for X in Ds:
        if X is None: continue
        for c in LAT:
            for sd, v in X.vec(*c, s, "sat").items(): out[(c, sd)] = v
    return out

def f3(Lat, Lng):
    S = {s: pool([Lat if s in ("ls_nb", "ls_nb_n10") else Lng], s) for s in ("ls_nb", "ls_nb_n10", "ls_py_n0", "ls_py")}
    def pr(a, b):
        ks = sorted(set(S[a]) & set(S[b]))
        if len(ks) < MIN_POOL: return None
        x = np.array([S[a][k] for k in ks]); y = np.array([S[b][k] for k in ks]); return len(ks), int(x.sum()), int(y.sum()), int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    rule_a = f"exploratory (descriptive, no p): over the 16 Latin cells x seeds 400-404 (80 paired instances): solved(ls_py_n0, 100 s) >= solved(ls_nb, 10 s) - {THR['f3a_slack']} x n (n = paired instances). SUPPORTED = the slow noise-0 search reaches the fast noise-0 level (noise is enough)"
    rule_b = f"exploratory (descriptive, no p): solved(ls_py_n0) - solved(ls_py, noise 0.1; both 100 s) >= {THR['f3b_gap']} x n. SUPPORTED = noise matters inside the slow implementation too"
    det = []; table = []
    for a, b, lab in (("ls_nb", "ls_py_n0", "fast noise0 vs slow noise0"), ("ls_nb_n10", "ls_py", "fast noise.1 vs slow noise.1"), ("ls_nb", "ls_nb_n10", "noise0 vs noise.1 (fast)"), ("ls_py_n0", "ls_py", "noise0 vs noise.1 (slow)")):
        r = pr(a, b)
        if r: table.append(f"{lab}: {a} {wl(r[1], r[0])}, {b} {wl(r[2], r[0])}; only-{a} {r[3]}, only-{b} {r[4]}; two-sided exact McNemar (unadjusted, descriptive) p = {p2(r[3], r[4]):.4f}")
        else: table.append(f"{lab}: not evaluable (< {MIN_POOL} paired instances)")
    ra, rb = pr("ls_nb", "ls_py_n0"), pr("ls_py_n0", "ls_py")
    va = "NOT EVALUABLE" if ra is None else "SUPPORTED" if ra[2] >= ra[1] - THR["f3a_slack"] * ra[0] - EPS else "FALSIFIED"
    vb = "NOT EVALUABLE" if rb is None else "SUPPORTED" if rb[1] - rb[2] >= THR["f3b_gap"] * rb[0] - EPS else "FALSIFIED"
    for c in LAT:
        row = f"{c}:"
        for s in ("ls_nb", "ls_nb_n10", "ls_py_n0", "ls_py"):
            k = [v for (cc, sd), v in S[s].items() if cc == c]; row += f" {s} {sum(k)}/{len(k)}"
        det.append(row)
    return [H("F3a", "exploratory", rule_a, va, table + det), H("F3b", "exploratory", rule_b, vb, [t for t in table if "slow)" in t])]

# ---------------------------------------------------------------- F4, F5
def gapk(D, n, ls, riv, k, gap, cells=(0.84, 0.88, 0.92)):
    res, det = [], []
    for p in cells:
        c = ("bnq", n, p); f = D.frac(*c, ls); fr = [D.frac(*c, s) for s in riv]
        if f is None or any(x is None for x in fr): res.append(None); det.append(f"n={n} p={p}: not evaluable"); continue
        best = max(fr); res.append(f - best >= gap - EPS)
        det.append(f"n={n} p={p}: {ls} {fmt(f)}; " + "; ".join(f"{s} {fmt(x)}" for s, x in zip(riv, fr)) + f"; {ls} - best rival {f - best:+.2f}; passes: {res[-1]}")
    return kofm(res, k), det

def f4(D):
    out = []
    for id_, n, ls, riv, lab in (("F4a", 200, "ls_nb", ("cpsat_strong", "cpsat_alt"), "one worker"), ("F4b", 200, "ls_nb", ("cpsat_strong_w4", "cpsat_alt_w4"), "four workers"),
                                 ("F4c", 100, "ls_nb", ("cpsat_strong", "cpsat_alt"), "one worker")):
        v, det = gapk(D, n, ls, riv, 2, THR["f4_gap"])
        out.append(H(id_, "exploratory", f"bnq n={n}, p in .84,.88,.92, {lab}: solved(ls_nb) - max(solved(cpsat_strong*), solved(cpsat_alt*)) >= +{THR['f4_gap']} in >= 2 of 3 cells (point estimates). Descriptive verdict, no p-value", v, det))
    det = []
    for n, p in BNQ6:
        for sa, sb in (("cpsat_alt", "cpsat_strong"), ("cpsat_alt_w4", "cpsat_strong_w4")):
            a = paired(D, ("bnq", n, p), sa, sb)
            det.append(f"n={n} p={p}: {sa} - {sb}: not evaluable" if a is None else f"n={n} p={p}: {sa} {wl(int(a[0].sum()), len(a[0]))} {sb} {wl(int(a[1].sum()), len(a[1]))}; only-alt {a[2]}, only-strong {a[3]}; two-sided exact McNemar (unadjusted, per cell) p = {p2(a[2], a[3]):.4f}")
    out.append(H("F4d", "exploratory (descriptive, no rule)", "per-cell comparison of the two CP-SAT models; no direction, no aggregate", "DESCRIPTIVE", det))
    return out

def f5(D):
    res, res2, det = [], [], []
    for n, p in BNQ6:
        c = ("bnq", n, p); a, b, z = D.frac(*c, "ls_py"), D.frac(*c, "cpsat_strong"), D.frac(*c, "ls_nb")
        if a is None or b is None or z is None: res.append(None); res2.append(None); det.append(f"n={n} p={p}: not evaluable"); continue
        res.append(a - b >= THR["f5_gap"] - EPS); res2.append(abs(a - z) <= THR["f5_within"] + EPS)
        tt = list(times(D, c, "ls_py").values()); v = D.vec(*c, "ls_py")
        det.append(f"n={n} p={p}: ls_py {wl(sum(v.values()), len(v))} (median solved t {fmt(statistics.median(tt) if tt else None)} s), ls_nb {fmt(z)}, cpsat_strong {fmt(b)}; ls_py - cpsat_strong {a - b:+.2f}: {res[-1]}; |ls_py - ls_nb| {abs(a - z):.2f}: {res2[-1]}")
    return [H("F5a", "exploratory", f"bnq n in 100, 200 x p in .84, .88, .92: solved(ls_py, interpreted) - solved(cpsat_strong) >= +{THR['f5_gap']} in >= 5 of 6 cells (point estimates). Descriptive verdict, no p-value", kofm(res, 5), det),
            H("F5b", "exploratory", f"|solved(ls_py) - solved(ls_nb)| <= {THR['f5_within']} in >= 4 of 6 cells", kofm(res2, 4), [])]

# ---------------------------------------------------------------- report
def report(D, Lat, Lng, F, fixinfo, diag, types, out):
    os.makedirs(out, exist_ok=True); bug, cnt, killed = D.checks()
    for X in (Lat, Lng):
        if X is not None: X.checks()
    with open(os.path.join(out, "cells.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "family", "size", "param", "solver", "n", "solved", "wilson_lo", "wilson_hi", "decided", "par10", "median_t_solved"])
        for tag, X in (("main", D), ("lat", Lat), ("long", Lng)):
            for (fam, size, param, sv), d in sorted((X.I if X else {}).items()):
                T = float(next(iter(d.values()))["T"]); n = len(d); k = sum(r["status"] == "sat" for r in d.values()); dec = sum(r["status"] in ("sat", "unsat") for r in d.values())
                par = np.mean([float(r["t"]) if r["status"] in ("sat", "unsat") else 10 * T for r in d.values()]); ts = [float(r["t"]) for r in d.values() if r["status"] == "sat"]
                lo, hi = wilson(k, n); w.writerow([tag, fam, size, param, sv, n, k, f"{lo:.3f}", f"{hi:.3f}", dec, f"{par:.2f}", f"{np.median(ts):.3f}" if ts else ""])
    Hs = [f1(D, fixinfo, F, diag, types), f1b(F), f2a(D), f2b(D), *f3(Lat, Lng), *f4(D), *f5(D)]
    Ls = ["# nq_v3f results (generated by analyze_v3f.py; verdict rules fixed in DESIGN_NOTE_V3F.md)", "",
          f"rows {len(D.rows)}; status counts {cnt}; killed {killed}; sat/unsat conflicts {len(bug)} {'**STOP: BUG**' if bug else ''}",
          "env: " + "; ".join(f"{e.get('cpu_model')} / ortools {e.get('ortools')} / numba {e.get('numba')} / stage {e.get('stage')}" for e in D.env), ""]
    for h in Hs: Ls += [f"## {h['id']} [{h['role']}] {h['verdict']}", f"Rule: {h['rule']}", *[f"- {x}" for x in h["detail"]], ""]
    open(os.path.join(out, "hypotheses.md"), "w").write("\n".join(Ls)); print("\n".join(Ls)); return Hs

# ---------------------------------------------------------------- selftest on fake data
def selftest():
    assert p2(5, 0) == 2 / 32 and p2(0, 0) == 1.0
    rg = random.Random(4); tmp = tempfile.mkdtemp(); HD = ["stage", "family", "size", "param", "seed", "solver", "workers", "T", "status", "t", "t_raw", "verified", "killed"]
    FH = []
    def mk(name): path = os.path.join(tmp, name); fh = open(path, "w", newline=""); FH.append(fh); W = csv.writer(fh); W.writerow(HD); return path, W
    def flush():
        for fh in FH: fh.flush()
    def put(W, fam, n, p, s, sv, pr, T=10, tlo=.01, thi=5, ok=None):
        ok = rg.random() < pr if ok is None else ok
        W.writerow(["fake", fam, n, p, s, sv, 4 if sv.endswith("_w4") else 1, T, "sat" if ok else "to", round(rg.uniform(tlo, thi), 3) if ok else T, "", 1 if ok else "", 0])
    pm, W = mk("main.csv")
    for s in range(20):
        for n, p in BNQ6 + [(200, 0.90)]:
            for sv, pr in (("ls_nb", 1.0), ("ls_nb_fixed", 1.0), ("ls_py", 1.0), ("cpsat_strong", 0.0 if n == 200 else .7), ("cpsat_alt", 0.0 if n == 200 else .6)): put(W, "bnq", n, p, s, sv, pr, tlo=.15 if sv == "ls_nb_fixed" else 1.0, thi=.9 if sv == "ls_nb_fixed" else 2.0)
            if p != 0.90:
                for sv in ("cpsat_strong_w4", "cpsat_alt_w4"): put(W, "bnq", n, p, s, sv, 0.1 if n == 200 else .9)
        for r in K5_LOW:
            for sv in ("ls_nbk", "probsat_nb", "cadical195"): put(W, "ksat5", 200, r, s, sv, 1.0)
        for r in K5_HI:
            for sv, pr in (("ls_nbk", 0.0), ("ls_nbk_n30", 0.8), ("ls_nbk_n40", 0.6), ("ls_nbk_n50", 0.2), ("ls_nbk_n65", 0.0), ("probsat_nb", 1.0), ("cadical195", 0.1)): put(W, "ksat5", 200, r, s, sv, pr)
    pl, WL = mk("lat.csv"); pg, WG = mk("long.csv")
    for s in range(5):
        for fam, n, p in LAT:
            solved = rg.random() < .8
            put(WL, fam, n, p, s, "ls_nb", 0, ok=solved); put(WL, fam, n, p, s, "ls_nb_n10", 0, ok=solved and rg.random() < .6)
            put(WG, fam, n, p, s, "ls_py_n0", 0, T=100, ok=solved or rg.random() < .1); put(WG, fam, n, p, s, "ls_py", 0, T=100, ok=solved and rg.random() < .4)
    flush()
    F = [dict(family="bnq", size="200", param=".9", seed="0", solver=sv, first_s=str(v), detail="") for sv, v in (("ls_nb", 1.1), ("ls_nb_fixed", .004)) for _ in range(10)]
    rows = []
    for cond, v in (("warm_real", 1.1), ("warm_same", 1.0), ("sig", .01), ("cache_cold", .02), ("cache_warm", .02)):
        rows += [dict(cond=cond, mode="fork", rep=i, t_build=0, t_seed=.0001, t_init1=v / 2, t_steps1=v / 2, t_init2=.001, t_steps2=.001, sig_init_before=1, sig_init_after=1, sig_steps_before=1, sig_steps_after=1, compile_events=0) for i in range(5)]
    S = diag_f1.summarize(rows); fixinfo = diag_f1.choose_fix(S); assert fixinfo["fix"] == "sig", fixinfo
    Hs = report(Data(pm), Data(pl), Data(pg), F, fixinfo, rows, {"all_types_identical_across_boards": True}, os.path.join(tmp, "out")); V = {h["id"]: h["verdict"] for h in Hs}
    assert V["F1"] == "SUPPORTED" and V["F1b"] == "SUPPORTED" and V["F2a"] == "SUPPORTED" and V["F2b"] == "SUPPORTED", V
    assert V["F4a"] == "SUPPORTED" and V["F4b"] == "SUPPORTED" and V["F4c"] == "FALSIFIED" and V["F5a"] == "FALSIFIED" and V["F5b"] == "SUPPORTED" and V["F4d"] == "DESCRIPTIVE", V
    assert V["F3a"] in ("SUPPORTED", "FALSIFIED") and V["F3b"] in ("SUPPORTED", "FALSIFIED"), V
    # F1 falsified when no fix, and when the fixed solved set differs by more than 0.05 in a cell
    assert f1(Data(pm), {"fix": None}, F, None, None)["verdict"] == "FALSIFIED" and f1(Data(pm), None, F, None, None)["verdict"] == "NOT EVALUABLE"
    pm2, W2 = mk("main2.csv"); pm3, W3 = mk("main3.csv")
    for s in range(20):
        for p in PRIM:
            put(W2, "bnq", 200, p, s, "ls_nb", 0, ok=True); put(W2, "bnq", 200, p, s, "ls_nb_fixed", 0, ok=(s < 18 if p == 0.84 else True), tlo=.1, thi=.5)
    for s in range(20):
        for p in PRIM: put(W3, "bnq", 200, p, s, "ls_nb", 0, ok=True); put(W3, "bnq", 200, p, s, "ls_nb_fixed", 0, ok=True, tlo=.4, thi=.9)
    flush()
    assert f1(Data(pm2), {"fix": "sig"}, F, None, None)["verdict"] == "FALSIFIED"          # 18/20 vs 20/20 = 0.10 > 0.05
    assert f1(Data(pm3), {"fix": "sig"}, F, None, None)["verdict"] == "FALSIFIED"          # min time > 0.30
    assert "F1 [PRIMARY] SUPPORTED" in open(os.path.join(tmp, "out", "hypotheses.md")).read()
    print("SELFTEST OK", V)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="?"); ap.add_argument("--lat"); ap.add_argument("--long"); ap.add_argument("--diag"); ap.add_argument("--fix"); ap.add_argument("--types")
    ap.add_argument("--out", default="analysis_v3f"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    else:
        ld = lambda p: json.load(open(p)) if p and os.path.exists(p) else None
        report(Data(a.csv), Data(a.lat) if a.lat else None, Data(a.long) if a.long else None, side(a.csv + ".first.csv"), ld(a.fix), side(a.diag), ld(a.types), a.out)
