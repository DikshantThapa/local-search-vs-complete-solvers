"""Analysis for nq_v3d (definitions = DESIGN_NOTE_V3D.md; constants here must match it).
Usage: python analyze_v3d.py CONFIRM.csv [--ext EXT.csv] [--out DIR]        |   python analyze_v3d.py --selftest
Outputs DIR/cells.csv, DIR/hypotheses.md. Plain python + numpy only (no scipy/pandas/matplotlib)."""
import csv, os, sys, json, math, argparse, random, tempfile
import numpy as np

MINN = 16            # completed instances per (cell, solver) needed to enter a test
NB = 10000           # bootstrap resamples (paired, percentile, numpy seed 0)
COMPLETE = ("cpsat_strong", "cpsat_tuned", "cadical195")
LSS = ("ls_nb", "ls_tabu_nb")
def pn(p):
    try: return f"{float(p):g}"
    except ValueError: return str(p)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))

def mcnemar_p(b, c):
    """one-sided exact: P(X >= b), X ~ Bin(b + c, 0.5); b = instances only the first solver solves."""
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [None] * len(ps); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

def tau_b(x, y):
    n0 = n1 = n2 = C = D = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            n0 += 1; dx = (x[i] > x[j]) - (x[i] < x[j]); dy = (y[i] > y[j]) - (y[i] < y[j])
            n1 += dx == 0; n2 += dy == 0
            if dx * dy > 0: C += 1
            elif dx * dy < 0: D += 1
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return float("nan") if den == 0 else (C - D) / den

def boot_ci(d, seed=0):
    d = np.asarray(d, float); rg = np.random.default_rng(seed); mu = d[rg.integers(0, len(d), (NB, len(d)))].mean(1)
    return float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))

class Data:
    def __init__(self, path):
        self.rows = list(csv.DictReader(open(path))); self.I = {}
        for r in self.rows: self.I.setdefault((r["family"], str(int(float(r["size"]))), pn(r["param"]), r["solver"]), {})[r["seed"]] = r
        self.env = json.load(open(path + ".env.json")) if os.path.exists(path + ".env.json") else []
    def checks(self):
        late = [r for r in self.rows if r["status"] in ("sat", "unsat") and float(r["t"]) > float(r["T"]) + 1e-6]
        assert not late, f"{len(late)} late answers left in CSV"
        by = {}
        for r in self.rows: by.setdefault((r["family"], r["size"], r["param"], r["seed"]), set()).add(r["status"])
        bug = [k for k, s in by.items() if "sat" in s and "unsat" in s]
        cnt = {}
        for r in self.rows: cnt[r["status"]] = cnt.get(r["status"], 0) + 1
        return bug, cnt, sum(int(r["killed"] or 0) for r in self.rows)
    def vec(self, fam, size, param, solver, what="sat"):
        d = self.I.get((fam, str(size), pn(param), solver), {})
        return {s: int(r["status"] == "sat" or (what == "decided" and r["status"] == "unsat")) for s, r in d.items()}
    def ok(self, *a):
        return len(self.vec(*a)) >= MINN
    def frac(self, *a):
        v = self.vec(*a); return sum(v.values()) / len(v) if len(v) >= MINN else None

def paired(D, cell, s1, s2):
    a, b = D.vec(*cell, s1), D.vec(*cell, s2); common = sorted(set(a) & set(b))
    if len(common) < MINN: return None
    x = np.array([a[s] for s in common]); y = np.array([b[s] for s in common])
    return x, y, int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())

def kofm(res, k):
    res = [None if r is None else bool(r) for r in res]
    if sum(r is True for r in res) >= k: return "SUPPORTED"
    return "NOT EVALUABLE" if sum(r is not False for r in res) >= k else "FALSIFIED"

def best(D, cell, names):
    f = {s: D.frac(*cell, s) for s in names}; f = {s: v for s, v in f.items() if v is not None}
    return max(sorted(f), key=lambda s: f[s]) if f else None

# ------------------------------------------------------------------ hypotheses
def d1(D):
    res, det = [], []
    for p in (0.84, 0.86, 0.88, 0.90):
        c = ("bnq", 200, p); pr = paired(D, c, "ls_nb", "cpsat_tuned")
        if pr is None: res.append(None); det.append(f"p={p}: not evaluable"); continue
        x, y, b, cc = pr; lo, hi = boot_ci(x - y); res.append(lo >= 0.5)
        ps = paired(D, c, "ls_nb", "cpsat_strong"); info = ""
        if ps: l2, _ = boot_ci(ps[0] - ps[1]); info = f"; info: vs cpsat_strong diff {np.mean(ps[0]-ps[1]):.2f}, lower {l2:.2f}"
        det.append(f"p={p}: ls_nb {x.mean():.2f} cpsat_tuned {y.mean():.2f} diff {np.mean(x-y):.2f} paired CI [{lo:.2f},{hi:.2f}] lower>=0.5: {res[-1]}{info}")
    return dict(id="D1", role="PRIMARY", rule="bnq n=200 p in .84,.86,.88,.90: paired 95% CI lower bound of solved(ls_nb)-solved(cpsat_tuned) >= 0.5 in >= 3 of 4 cells", verdict=kofm(res, 3), p=None, detail=det)

def d2(D):
    res, ps, det = [], [], []
    for r in (4.2, 4.26, 4.3):
        c = ("ksat", 350, r); pr = paired(D, c, "probsat_nb", "cadical195")
        if pr is None: res.append(None); det.append(f"r={r}: not evaluable"); continue
        x, y, b, cc = pr; diff = x.mean() - y.mean(); res.append(diff >= 0.10); ps.append(mcnemar_p(b, cc))
        w = D.vec(*c, "ls_nb"); det.append(f"r={r}: probsat_nb {x.mean():.2f} cadical195 {y.mean():.2f} diff {diff:+.2f} McNemar p {ps[-1]:.4f}; ls_nb(WalkSAT) {np.mean(list(w.values())) if w else float('nan'):.2f}")
    p = sorted(ps)[1] if len(ps) >= 2 else None
    return dict(id="D2", role="exploratory", rule="ksat n=350 r in 4.2,4.26,4.3: solved(probsat_nb) - solved(cadical195) >= +0.10 in >= 2 of 3 cells (SAT answers only)", verdict=kofm(res, 2), p=p, detail=det)

def d3(D):
    res, ps, det = [], [], []
    for r in (20.0, 20.5, 21.0):
        c = ("ksat5", 200, r); rival = best(D, c, ("cadical195", "cpsat_tuned"))
        pr = paired(D, c, "probsat_nb", rival) if rival else None
        if pr is None: res.append(None); det.append(f"r={r}: not evaluable"); continue
        x, y, b, cc = pr; diff = x.mean() - y.mean(); res.append(diff >= 0.20); ps.append(mcnemar_p(b, cc))
        det.append(f"r={r}: probsat_nb {x.mean():.2f} best rival ({rival}) {y.mean():.2f} diff {diff:+.2f} McNemar p {ps[-1]:.4f}")
    return dict(id="D3", role="exploratory", rule="ksat5 n=200 r in 20,20.5,21: solved(probsat_nb) - max(solved(cadical195), solved(cpsat_tuned)) >= +0.20 in >= 2 of 3 cells", verdict=kofm(res, 2), p=sorted(ps)[1] if len(ps) >= 2 else None, detail=det)

def d4(D):
    res, ps, det = [], [], []
    for h in (0.35, 0.40):
        c = ("qwh", 40, h); bc, bl = best(D, c, COMPLETE), best(D, c, LSS); pr = paired(D, c, bc, bl) if bc and bl else None
        if pr is None: res.append(None); det.append(f"h={h}: not evaluable"); continue
        x, y, b, cc = pr; gap = x.mean() - y.mean(); res.append(gap >= 0.20); ps.append(mcnemar_p(b, cc))
        det.append(f"h={h}: best complete {bc} {x.mean():.2f}; best LS {bl} {y.mean():.2f}; ls_nb {D.frac(*c,'ls_nb')}, ls_tabu_nb {D.frac(*c,'ls_tabu_nb')}; gap {gap:+.2f} McNemar p {ps[-1]:.4f}")
    for f_ in (0.38, 0.42, 0.46):   # descriptive only
        c = ("qcp", 40, f_)
        if best(D, c, COMPLETE): det.append(f"(descriptive) qcp N=40 f={f_}: complete {D.frac(*c, best(D, c, COMPLETE))}, ls_nb {D.frac(*c,'ls_nb')}, ls_tabu_nb {D.frac(*c,'ls_tabu_nb')}")
    return dict(id="D4", role="exploratory", rule="qwh N=40 h in .35,.40: solved(best complete) - solved(best of ls_nb, ls_tabu_nb) >= +0.20 in BOTH cells", verdict=kofm(res, 2), p=max(ps) if len(ps) == 2 else None, detail=det)

def d5(E):
    det, res = [], []
    if E is None: return dict(id="D5", role="exploratory", rule="outside benchmarks", verdict="NOT EVALUABLE", p=None, detail=["no ext CSV given"])
    for fam, sv, thr in (("ext_uf250", "probsat_nb", 0.9), ("ext_flat150", "probsat_nb", 0.8), ("ext_qg", None, None)):
        n_inst = {k[2] for k in E.I if k[0] == fam}
        for s in ("probsat_nb", "cadical195", "cpsat_strong", "cpsat_tuned"):
            k = sum(1 for p in n_inst for r in [E.I.get((fam, "0", p, s), {}).get("0")] if r and r["status"] == "sat"); n = len(n_inst); lo, hi = wilson(k, n)
            det.append(f"{fam} {s}: solved {k}/{n} Wilson [{lo:.2f},{hi:.2f}]")
            if sv == s: res.append((k / n >= thr) if n >= (16 if fam != "ext_qg" else 1) else None)
    return dict(id="D5", role="exploratory", rule="SATLIB uf250-1065 (first 20 files): probsat_nb solves >= 0.90; flat150-360 (first 20): probsat_nb solves >= 0.80; QG: descriptive only (no directional claim)", verdict=kofm(res, 2) if len(res) == 2 else "NOT EVALUABLE", p=None, detail=det)

def d6(D):
    taus, det = [], []; fams = sorted({k[0] for k in D.I if not k[0].startswith("ext_")})
    for fam in fams:
        cells = sorted({(k[1], k[2]) for k in D.I if k[0] == fam}); ft = []
        for size, param in cells:
            sols = sorted({k[3] for k in D.I if k[:3] == (fam, size, param)}); base = [s for s in sols if not s.startswith("cpsat")]
            if "cpsat_strong" not in sols or "cpsat_tuned" not in sols: continue
            fs = {s: D.frac(fam, size, param, s) for s in sols}
            if any(v is None for v in fs.values()): continue
            t = tau_b([fs[s] for s in base] + [fs["cpsat_strong"]], [fs[s] for s in base] + [fs["cpsat_tuned"]])
            if not math.isnan(t): ft.append(t)
            else: ft.append(None)
        ok = [t for t in ft if t is not None]; taus += ok
        det.append(f"{fam}: cells {len(ft)}, tau defined {len(ok)}, mean tau {np.mean(ok):.2f}" if ok else f"{fam}: no defined tau")
    allc = sum(1 for _ in taus)
    v = "NOT EVALUABLE" if not taus else ("SUPPORTED" if np.mean(taus) >= 0.8 else "FALSIFIED")
    det.append(f"all families: defined cells {len(taus)}, mean tau {np.mean(taus):.2f}" if taus else "no cells"); 
    return dict(id="D6", role="exploratory (descriptive, no p-value)", rule="per cell, Kendall tau-b of solved fractions over {non-CP-SAT solvers + cpsat_strong} vs {same + cpsat_tuned}; mean over cells where defined >= 0.80 (cells with a constant vector are undefined)", verdict=v, p=None, detail=det)

def report(D, E, out):
    os.makedirs(out, exist_ok=True); bug, cnt, killed = D.checks()
    with open(os.path.join(out, "cells.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "size", "param", "solver", "n", "solved", "wilson_lo", "wilson_hi", "decided", "par10", "median_t_solved"])
        for (fam, size, param, sv), d in sorted(D.I.items()):
            T = float(next(iter(d.values()))["T"]); n = len(d); k = sum(r["status"] == "sat" for r in d.values()); dec = sum(r["status"] in ("sat", "unsat") for r in d.values())
            par = np.mean([float(r["t"]) if r["status"] in ("sat", "unsat") else 10 * T for r in d.values()]); ts = [float(r["t"]) for r in d.values() if r["status"] == "sat"]
            lo, hi = wilson(k, n); w.writerow([fam, size, param, sv, n, k, f"{lo:.3f}", f"{hi:.3f}", dec, f"{par:.2f}", f"{np.median(ts):.3f}" if ts else ""])
    H = [d1(D), d2(D), d3(D), d4(D), d5(E), d6(D)]
    ex = [h for h in H if h["role"] == "exploratory" and h["p"] is not None]; adj = holm([h["p"] for h in ex]); 
    for h, a in zip(ex, adj): h["p_holm"] = a
    L = ["# nq_v3d results (generated by analyze_v3d.py; verdict rules fixed in DESIGN_NOTE_V3D.md)", "",
         f"rows {len(D.rows)}; status counts {cnt}; killed {killed}; sat/unsat conflicts {len(bug)} {'**STOP: BUG**' if bug else ''}",
         "env: " + "; ".join(f"{e.get('cpu_model')} / ortools {e.get('ortools')} / numba {e.get('numba')} / stage {e.get('stage')}" for e in D.env), ""]
    for h in H:
        v = h["verdict"]
        if "p_holm" in h and v == "SUPPORTED": v = "SUPPORTED (Holm p=%.3f)" % h["p_holm"] if h["p_holm"] < 0.05 else "RULE MET, NOT Holm-significant (p_holm=%.3f)" % h["p_holm"]
        L += [f"## {h['id']} [{h['role']}] {v}", f"Rule: {h['rule']}", f"Raw p (intersection-union): {h['p']}" if h["p"] is not None else "", *[f"- {x}" for x in h["detail"]], ""]
    open(os.path.join(out, "hypotheses.md"), "w").write("\n".join(L)); print("\n".join(L)); return H

# ------------------------------------------------------------------ selftest
def selftest():
    assert abs(tau_b([1, 2, 3], [1, 2, 3]) - 1) < 1e-9 and abs(tau_b([1, 2, 3], [3, 2, 1]) + 1) < 1e-9 and math.isnan(tau_b([1, 1, 1], [1, 2, 3]))
    assert abs(mcnemar_p(5, 0) - 1 / 32) < 1e-12 and mcnemar_p(0, 0) == 1.0
    lo, hi = wilson(0, 20); assert lo == 0 and 0.15 < hi < 0.17
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    rg = random.Random(1); tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "fake.csv")
    P = {"ls_nb": .97, "ls_tabu_nb": .6, "probsat_nb": .95, "cpsat_strong": .1, "cpsat_tuned": .15, "cadical195": .1}
    W = csv.writer(open(path, "w", newline="")); W.writerow(["stage", "family", "size", "param", "seed", "solver", "workers", "T", "status", "t", "t_raw", "verified", "killed"])
    cells = [("bnq", 200, p) for p in (.84, .86, .88, .9)] + [("qwh", 40, h) for h in (.35, .4)] + [("ksat", 350, r) for r in (4.2, 4.26, 4.3)] + [("ksat5", 200, r) for r in (20.0, 20.5, 21.0)]
    for fam, n, p in cells:
        for s in range(20):
            for sv in ("ls_nb", "ls_tabu_nb", "probsat_nb", "cpsat_strong", "cpsat_tuned", "cadical195"):
                if sv in ("ls_tabu_nb",) and fam != "qwh": continue
                if sv == "probsat_nb" and fam not in ("ksat", "ksat5"): continue
                pr = P[sv] if not (fam == "qwh" and sv in ("cpsat_strong", "cpsat_tuned", "cadical195")) else .9
                ok = rg.random() < pr; W.writerow(["fake", fam, n, p, s, sv, 1, 10, "sat" if ok else "to", round(rg.uniform(.1, 9), 3) if ok else 10, "", 1 if ok else "", 0])
    pe = os.path.join(tmp, "ext.csv"); W = csv.writer(open(pe, "w", newline="")); W.writerow(["stage", "family", "size", "param", "seed", "solver", "workers", "T", "status", "t", "t_raw", "verified", "killed"])
    for fam in ("ext_uf250", "ext_flat150", "ext_qg"):
        for i in range(20):
            for sv in ("probsat_nb", "cadical195", "cpsat_strong", "cpsat_tuned"): W.writerow(["fake", fam, 0, f"inst{i}", 0, sv, 1, 10, "sat", 1.0, "", 1, 0])
    H = report(Data(path), Data(pe), os.path.join(tmp, "out")); V = {h["id"]: h["verdict"] for h in H}
    assert V["D1"] == "SUPPORTED" and V["D2"] == "SUPPORTED" and V["D3"] == "SUPPORTED" and V["D5"] == "SUPPORTED", V
    assert V["D4"] == "FALSIFIED", V   # fake data: complete .9 vs ls_nb .97 -> no gap
    print("SELFTEST OK", V)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="?"); ap.add_argument("--ext"); ap.add_argument("--out", default="analysis_v3d"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    else: report(Data(a.csv), Data(a.ext) if a.ext else None, a.out)
