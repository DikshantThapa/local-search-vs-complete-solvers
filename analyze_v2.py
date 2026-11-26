"""Analysis for nq_v2 results. Usage: python analyze_v2.py RESULTS.csv [W4.csv] [--out DIR]
Writes DIR/cells.csv, DIR/tables.md, DIR/hypotheses.md, DIR/fig_solved.png, DIR/fig_par2.png. Bootstrap: 10,000 resamples, seed 12345 (PREREG_NOTE.md)."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OKABE = {"ls": "#E69F00", "cpsat_naive": "#56B4E9", "cpsat_strong": "#0072B2", "glucose4": "#D55E00", "cadical195": "#009E73", "cpsat_strong_w4": "#CC79A7"}
MARK = {"ls": "o", "cpsat_naive": "s", "cpsat_strong": "D", "glucose4": "^", "cadical195": "v", "cpsat_strong_w4": "P"}
FAM = {"bnq": "Blocked n-Queens (p blocked)", "qwh": "Quasigroup with holes (holes)", "qcp": "Quasigroup completion (preassigned)", "ksat": "Random 3-SAT (clauses/variables)"}
NB, RNG = 10000, np.random.default_rng(12345)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10, "figure.dpi": 150})

def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    late = d.status.isin(["sat", "unsat"]) & (d.t > d["T"])  # strict budget: an answer after T counts as a timeout (runs were killed at T + GRACE)
    d.loc[late, "status"] = "to"; d.loc[late, "verified"] = 0
    print("late answers reclassified as timeouts:", int(late.sum()))
    d["solved"] = (d.status == "sat") & (d.verified == 1); d["decided"] = d.solved | (d.status == "unsat")
    d["par2s"] = np.where(d.solved, d.t, 2 * d["T"]); d["par2d"] = np.where(d.decided, d.t, 2 * d["T"])
    return d

def boot_ci(x, f=np.mean):
    x = np.asarray(x, float); i = RNG.integers(0, len(x), (NB, len(x))); b = f(x[i], axis=1) if f is np.mean else np.apply_along_axis(f, 1, x[i])
    return np.percentile(b, [2.5, 97.5])

def cells(d):
    rows = []
    for (fam, n, p, sv), g in d.groupby(["family", "size", "param", "solver"]):
        r = dict(family=fam, size=n, param=p, solver=sv, n=len(g), solved=g.solved.mean(), decided=g.decided.mean(), par2s=g.par2s.mean(), par2d=g.par2d.mean(),
                 med_t_solved=g.t[g.solved].median() if g.solved.any() else np.nan, bad=int((g.status == "bad").sum()), err=int(g.status.str.startswith("err").sum()))
        r["solved_lo"], r["solved_hi"] = boot_ci(g.solved.values) if len(g) > 1 else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)

def label_cells(d):
    """SAT-side: >=90% of instances solved by some solver; UNSAT-side: >=90% proven unsat; else transition."""
    k = d.groupby(["family", "size", "param", "seed"]).agg(sat=("solved", "any"), uns=("status", lambda s: (s == "unsat").any())).reset_index()
    c = k.groupby(["family", "size", "param"]).agg(sat=("sat", "mean"), uns=("uns", "mean")).reset_index()
    c["side"] = np.where(c.sat >= .9, "SAT", np.where(c.uns >= .9, "UNSAT", "transition")); return c

def paired(d, sel, a, b, col="solved"):
    """paired bootstrap of mean(col_a - col_b) over instances in selection `sel` (DataFrame mask of cells)."""
    x = d[sel & (d.solver == a)].set_index(["family", "size", "param", "seed"])[col].astype(float); y = d[sel & (d.solver == b)].set_index(["family", "size", "param", "seed"])[col].astype(float)
    z = (x - y.reindex(x.index)).dropna().values
    if len(z) == 0: return np.nan, np.nan, np.nan, 0
    lo, hi = boot_ci(z); return z.mean(), lo, hi, len(z)

def hypotheses(d, lab, cl):
    out = ["# Hypothesis checks (thresholds from PREREG_NOTE.md; TODO-RESULTS if cells are missing)\n"]; m = lambda fam, n=None: (d.family == fam) & ((d.size == n) if n else True)
    def cell(fam, n, p): return (d.family == fam) & (d["size"] == n) & (d.param == p)
    # H1
    out.append("## H1 bnq n=200, p in .84-.90: solved(ls) - solved(cpsat_strong), paired 95% CI; supported if lower bound >= 0.5 in at least 3 of 4 cells")
    ok = 0
    for p in (.84, .86, .88, .90):
        e, lo, hi, k = paired(d, cell("bnq", 200, p), "ls", "cpsat_strong"); ok += lo >= .5; out.append(f"- p={p}: {e:.2f} [{lo:.2f}, {hi:.2f}] (n={k})")
    out.append(f"- verdict: {'supported' if ok >= 3 else 'falsified or no data'} ({ok}/4 cells)\n")
    # H2
    out.append("## H2 family-specific LS advantage: ls vs best complete solver on SAT-side cells at largest size")
    for fam, n in (("qwh", 40), ("qcp", 40), ("ksat", 350)):
        c = cl[(cl.family == fam) & (cl["size"] == n)]; sat = lab[(lab.family == fam) & (lab["size"] == n) & (lab.side == "SAT")].param.tolist()
        for p in sat:
            s = c[c.param == p].set_index("solver").solved; best = s.drop("ls", errors="ignore").max(); out.append(f"- {fam} n={n} p={p}: ls {s.get('ls', np.nan):.2f}, best complete {best:.2f}")
    out.append("")
    # H3
    out.append("## H3 encoding: pooled solved(cpsat_strong) - solved(cpsat_naive) over SAT-side and transition cells; supported if >= 0.1 for bnq and latin (qwh+qcp) and |.| < 0.1 for ksat")
    for name, fams in (("bnq", ["bnq"]), ("latin", ["qwh", "qcp"]), ("ksat", ["ksat"])):
        keep = lab[lab.family.isin(fams)]; sel = d.family.isin(fams) & d.set_index(["family", "size", "param"]).index.isin(keep.set_index(["family", "size", "param"]).index)
        e, lo, hi, k = paired(d, sel, "cpsat_strong", "cpsat_naive"); out.append(f"- {name}: {e:.3f} [{lo:.3f}, {hi:.3f}] (n={k})")
    out.append("")
    # H4
    out.append("## H4 UNSAT side: decided fraction by solver, largest sizes (bnq n=100,200 p>=.94; ksat n=250 ratio>=4.4)")
    for fam, n, pm in (("bnq", 100, .94), ("bnq", 200, .94), ("ksat", 250, 4.4)):
        c = cl[(cl.family == fam) & (cl["size"] == n) & (cl.param >= pm)].groupby("solver").decided.mean(); out.append(f"- {fam} n={n} param>={pm}: " + ", ".join(f"{k} {v:.2f}" for k, v in c.items()))
    out.append("")
    # H5
    out.append("## H5 hardness peak: argmax of PAR2(decision) of the best complete solver vs. cell where SAT fraction crosses 0.5, largest size")
    for fam in FAM:
        n = d[d.family == fam]["size"].max(); c = cl[(cl.family == fam) & (cl["size"] == n) & cl.solver.isin(["cpsat_strong", "glucose4", "cadical195"])]
        if c.empty: continue
        best = c.pivot(index="param", columns="solver", values="par2d").min(axis=1); sat = lab[(lab.family == fam) & (lab["size"] == n)].set_index("param").sat
        cross = sat.index[np.argmin(np.abs(sat.values - .5))]; out.append(f"- {fam} n={n}: peak at {best.idxmax()}, SAT-fraction crosses 0.5 near {cross} (grid index distance {abs(list(best.index).index(best.idxmax()) - list(best.index).index(cross))})")
    out.append("")
    # H6
    if "cpsat_strong_w4" in set(d.solver):
        out.append("## H6 4 workers, bnq n=200 cells where ls solves all instances (5 seeds): solved(cpsat_strong_w4)")
        for p in sorted(d[d.family == "bnq"].param.unique()):
            g = d[cell("bnq", 200, p) & (d.seed < 5)]; l = g[g.solver == "ls"].solved; w = g[g.solver == "cpsat_strong_w4"].solved
            if len(l) and l.all() and len(w): out.append(f"- p={p}: w4 solved {w.mean():.2f} (n={len(w)})")
        out.append("")
    # H7
    c = cl.pivot_table(index=["family", "size", "param"], columns="solver", values="decided"); 
    if {"cadical195", "glucose4"} <= set(c.columns):
        frac = (c.cadical195 >= c.glucose4).mean(); out.append(f"## H7 exploratory: cadical195 decided >= glucose4 in {frac:.0%} of cells (threshold 75%)\n")
    return "\n".join(out)

def fig_solved(cl, out):
    fams = [f for f in FAM if f in set(cl.family)]; sizes = {f: sorted(cl[cl.family == f]["size"].unique()) for f in fams}
    fig, ax = plt.subplots(len(fams), 3, figsize=(15, 3.6 * len(fams)), squeeze=False)
    for i, f in enumerate(fams):
        for j, n in enumerate(sizes[f][:3]):
            a = ax[i, j]
            for sv, g in cl[(cl.family == f) & (cl["size"] == n)].groupby("solver"):
                g = g.sort_values("param"); a.plot(g.param, g.solved, marker=MARK[sv], color=OKABE[sv], label=sv, lw=1.6, ms=6); a.fill_between(g.param, g.solved_lo, g.solved_hi, color=OKABE[sv], alpha=.12)
            a.set_ylim(-.03, 1.03); a.set_title(f"{FAM[f].split(' (')[0]}, size {n}"); a.set_xlabel(FAM[f].split("(")[1].rstrip(")")); a.set_ylabel("fraction solved")
    for i, f in enumerate(fams):
        for j in range(len(sizes[f][:3]), 3): ax[i, j].axis("off")
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out, dpi=200); plt.close(fig)

def fig_par2(cl, out):
    fams = [f for f in FAM if f in set(cl.family)]; fig, ax = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.8), squeeze=False)
    for a, f in zip(ax[0], fams):
        n = cl[cl.family == f]["size"].max()
        for sv, g in cl[(cl.family == f) & (cl["size"] == n)].groupby("solver"):
            g = g.sort_values("param"); a.plot(g.param, g.par2d, marker=MARK[sv], color=OKABE[sv], label=sv, lw=1.6)
        a.set_title(f"{f}, size {n}"); a.set_xlabel(FAM[f].split("(")[1].rstrip(")")); a.set_ylabel("PAR2 (s), decision")
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; od = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "analysis_v2"
    if "--out" in sys.argv: args.remove(od)
    os.makedirs(od, exist_ok=True); d = load(args); cl = cells(d); lab = label_cells(d[d.solver != "cpsat_strong_w4"])
    cl.to_csv(f"{od}/cells.csv", index=False)
    with open(f"{od}/tables.md", "w") as f:
        f.write(f"rows {len(d)}; status counts {d.status.value_counts().to_dict()}; bad {int((d.status == 'bad').sum())}; err {int(d.status.str.startswith('err').sum())}\n\n")
        for fam in FAM:
            c = cl[cl.family == fam]
            if c.empty: continue
            t = c.assign(cellv=lambda x: x.solved.map("{:.2f}".format) + " (" + x.med_t_solved.map(lambda v: "-" if np.isnan(v) else f"{v:.2f}s") + ")").pivot(index=["size", "param"], columns="solver", values="cellv")
            f.write(f"### {fam}: fraction solved (median time of solved)\n\n{t.to_markdown()}\n\n")
        f.write("### cell labels (SAT / UNSAT side)\n\n" + lab.round(2).to_markdown(index=False) + "\n")
    open(f"{od}/hypotheses.md", "w").write(hypotheses(d, lab, cl)); fig_solved(cl, f"{od}/fig_solved.png"); fig_par2(cl, f"{od}/fig_par2.png"); print("wrote", od)

if __name__ == "__main__": main()
