"""Analysis for nq_v3c confirm run. Definitions: DESIGN_NOTE_V3C.md sections 2-3 (fixed before the run).
Usage: python analyze_v3c.py confirm.csv [--out DIR]
Hypotheses C1 (PRIMARY) .. C5, Wilson/paired-bootstrap intervals, PAR10, ECDF figure for the headline cell. Numpy/pandas/matplotlib only."""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

KEY = ["family", "size", "param", "seed"]
COMPLETE = ["cpsat_strong", "cpsat_strong_w4", "cadical195"]
RNG = np.random.default_rng(0); NB = 10000; MIN_N = 16

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def load(path):
    d = pd.read_csv(path); d["status"] = d.status.astype(str)
    assert not (d.status.isin(["sat", "unsat", "bad"]) & (d.t > d["T"])).any(), "late answer left"
    d["solved"] = (d.status == "sat") & (d.verified == 1); d["decided"] = d.solved | (d.status == "unsat")
    d["par10"] = np.where(d.solved, d.t, 10 * d["T"]); d["param"] = d.param.astype(float)
    n = d.groupby(["family", "size", "param", "solver"]).size().groupby(["family", "size", "param"]).min().rename("m").reset_index()
    d = d.merge(n, on=["family", "size", "param"]); excl = int((n.m < MIN_N).sum()); return d[d.m >= MIN_N].drop(columns="m"), excl

def sel(d, fam=None, n=None, p=None):
    m = pd.Series(True, index=d.index)
    if fam: m &= d.family == fam
    if n: m &= d["size"] == n
    if p is not None: m &= np.isclose(d.param, p)
    return m

def paired(d, m, a, b):
    x = d[m & (d.solver == a)].set_index(KEY).solved.astype(float); y = d[m & (d.solver == b)].set_index(KEY).solved.astype(float)
    z = (x - y.reindex(x.index)).dropna().values
    if len(z) < 2: return np.nan, np.nan, np.nan, len(z)
    i = RNG.integers(0, len(z), (NB, len(z))); lo, hi = np.percentile(z[i].mean(axis=1), [2.5, 97.5]); return z.mean(), lo, hi, len(z)

def frac(d, fam, n, p, sv, col="solved"):
    g = d[sel(d, fam, n, p) & (d.solver == sv)][col]; return g.mean() if len(g) else np.nan

def labels(d):
    k = d.groupby(KEY).agg(sat=("solved", "any"), uns=("status", lambda s: (s == "unsat").any())).reset_index()
    c = k.groupby(["family", "size", "param"]).agg(sat=("sat", "mean"), uns=("uns", "mean")).reset_index()
    c["side"] = np.where(c.sat >= .9, "SAT", np.where(c.uns >= .9, "UNSAT", "transition")); return c

def best_complete(d, fam, n, p):
    v = {sv: frac(d, fam, n, p, sv) for sv in COMPLETE}; v = {k: x for k, x in v.items() if not np.isnan(x)}
    return (max(v.values()), max(v, key=v.get)) if v else (np.nan, "-")

def c1(d):
    o = ["## C1 (PRIMARY) bnq n=200, p in {.84,.86,.88,.90}: solved(ls_nb) - solved(cpsat_strong), paired 95% bootstrap CI; supported if lower bound >= 0.5 in >= 3 of 4 cells."]; ok = 0; k = 0
    for p in (.84, .86, .88, .90):
        e, lo, hi, n = paired(d, sel(d, "bnq", 200, p), "ls_nb", "cpsat_strong"); k += n >= 2; ok += int(n >= 2 and lo >= .5)
        a = d[sel(d, "bnq", 200, p) & (d.solver == "ls_nb")].solved; w = wilson(int(a.sum()), len(a))
        o.append(f"- p={p}: diff {e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={n}); ls_nb solved {a.mean():.2f} Wilson [{w[0]:.2f}, {w[1]:.2f}]; cpsat_strong {frac(d, 'bnq', 200, p, 'cpsat_strong'):.2f}")
    return o + [f"- VERDICT: {'supported' if ok >= 3 else 'falsified' if k == 4 else 'not evaluable'} ({ok}/4 cells)", ""]

def c2(d):
    o = ["## C2 (exploratory, core-matched) bnq n=200 cells where ls_nb_w4 solves every instance: cpsat_strong_w4 solves < 0.5, in >= 3 such cells."]; k = 0; ok = 0
    for p in sorted(d[sel(d, "bnq", 200)].param.unique()):
        l = d[sel(d, "bnq", 200, p) & (d.solver == "ls_nb_w4")].solved; w = d[sel(d, "bnq", 200, p) & (d.solver == "cpsat_strong_w4")].solved
        if len(l) and l.all() and len(w): k += 1; ok += int(w.mean() < .5); o.append(f"- p={p}: ls_nb_w4 {l.mean():.2f}, cpsat_strong_w4 {w.mean():.2f} (n={len(w)})")
    return o + [f"- VERDICT: {'not evaluable (<3 such cells)' if k < 3 else 'supported' if ok >= 3 else 'falsified'} ({ok} of {k} cells)", ""]

def c3(d, lab):
    o = ["## C3 (exploratory, replicates H8) SAT-side qwh/qcp cells at N in {30,40}: share with solved(ls_nb) >= 0.8*best complete is >= 0.75, per family."]; res = {}
    for fam in ("qwh", "qcp"):
        cs = lab[(lab.family == fam) & lab["size"].isin([30, 40]) & (lab.side == "SAT")]; hit = []
        for _, r in cs.iterrows():
            b, who = best_complete(d, fam, r["size"], r.param); l = frac(d, fam, r["size"], r.param, "ls_nb")
            hit.append(l >= .8 * b)
            if not hit[-1]: o.append(f"- FAIL {fam} N={r['size']} p={r.param}: ls_nb {l:.2f}, best complete {b:.2f} ({who})")
        res[fam] = (np.mean(hit) >= .75) if hit else None; o.append(f"- {fam}: {sum(hit)}/{len(hit)} cells hold")
    v = [x for x in res.values() if x is not None]
    return o + [f"- VERDICT: {'not evaluable' if len(v) < 2 else 'supported' if all(v) else 'partial' if any(v) else 'falsified'}", ""]

def c4(d, lab):
    o = ["## C4 (exploratory) qcp SAT-side cells at N=40: best complete solved - solved(ls_nb) >= 0.2 in at least one cell."]; gaps = []
    for p in lab[(lab.family == "qcp") & (lab["size"] == 40) & (lab.side == "SAT")].param:
        b, who = best_complete(d, "qcp", 40, p); l = frac(d, "qcp", 40, p, "ls_nb"); gaps.append(b - l); o.append(f"- qcp N=40 p={p}: best complete {b:.2f} ({who}), ls_nb {l:.2f}, gap {b - l:+.2f}")
    return o + [f"- VERDICT: {'not evaluable' if not gaps else 'supported (loss claim holds for this local search)' if max(gaps) >= .2 else 'falsified (no qcp N=40 cell with gap >= 0.2)'}", ""]

def c5(d):
    o = ["## C5 (exploratory) ksat n=350: ls_nb solved >= 0.9 at r=4.0 and cadical195 decided >= ls_nb solved at r in {4.26, 4.3}."]
    a = frac(d, "ksat", 350, 4.0, "ls_nb"); o.append(f"- r=4.0: ls_nb {a:.2f}"); ok = a >= .9
    for p in (4.26, 4.3):
        c = frac(d, "ksat", 350, p, "cadical195", "decided"); l = frac(d, "ksat", 350, p, "ls_nb"); o.append(f"- r={p}: cadical195 decided {c:.2f}, ls_nb solved {l:.2f}"); ok = ok and c >= l
    return o + [f"- VERDICT: {'supported' if ok else 'falsified'}", ""]

def tables(d, path):
    with open(path, "w") as f:
        for fam in ("bnq", "qwh", "qcp", "ksat"):
            g = d[d.family == fam]
            if g.empty: continue
            t = g.groupby(["size", "param", "solver"]).apply(lambda x: f"{x.solved.mean():.2f} [{wilson(int(x.solved.sum()), len(x))[0]:.2f},{wilson(int(x.solved.sum()), len(x))[1]:.2f}] par10 {x.par10.mean():.1f}", include_groups=False).unstack("solver")
            f.write(f"### {fam}: fraction solved [Wilson 95%] and PAR10\n\n{t.to_string()}\n\n")

def fig_ecdf(d, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, (fam, n, p) in zip(ax, (("bnq", 200, .88), ("qwh", 40, .45), ("ksat", 350, 4.2))):
        for sv in ("ls_nb", "ls_nb_w4", "cpsat_strong", "cpsat_strong_w4", "cadical195"):
            g = d[sel(d, fam, n, p) & (d.solver == sv)]
            if g.empty: continue
            ts = np.sort(g.t[g.solved].values); T = g["T"].iloc[0]; x = np.r_[0, ts, T]; y = np.r_[0, np.arange(1, len(ts) + 1), len(ts)] / len(g); a.step(x, y, where="post", label=sv)
        a.set_title(f"{fam} size {n} param {p}"); a.set_xlabel("time (s, budget-censored)"); a.set_ylabel("fraction solved"); a.set_ylim(-.02, 1.02)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out", default="analysis_v3c"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d, excl = load(a.csv); lab = labels(d); o = [f"# nq_v3c confirm hypotheses (definitions fixed in DESIGN_NOTE_V3C.md); cells excluded for < {MIN_N} instances: {excl}; rows {len(d)}; status {d.status.value_counts().to_dict()}; bad {int((d.status == 'bad').sum())}\n"]
    o += c1(d) + c2(d) + c3(d, lab) + c4(d, lab) + c5(d)
    open(f"{a.out}/hypotheses.md", "w").write("\n".join(o)); tables(d, f"{a.out}/tables.md"); fig_ecdf(d, f"{a.out}/fig_ecdf.png"); print("\n".join(o))

if __name__ == "__main__": main()
