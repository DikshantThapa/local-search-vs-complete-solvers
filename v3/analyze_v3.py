"""Analysis for nq_v3 (definitions fixed in DESIGN_NOTE_V3.md; do not change after the first non-smoke run).
Usage: python analyze_v3.py --main10 main10.csv [--long60 long60.csv] [--long300 long300.csv] [--out DIR]
Reads the CSV(s) and the <csv>.env.json next to each. The strict budget is already applied by the harness (late answers are stored as 'to');
this script only asserts that no answer with t > T is left. Bootstrap: 10,000 resamples, numpy seed 12345, percentile CI, paired over instances.
Writes DIR/environment.md, tables_<stage>.md, cells_<stage>.csv, hypotheses.md, fig_solved_main10.png, fig_par2_main10.png, fig_long_<stage>.png, fig_h8.png."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

COLOR = {"ls_nb": "#E69F00", "ls_py": "#999999", "cpsat_naive": "#56B4E9", "cpsat_strong": "#0072B2", "glucose4": "#D55E00", "cadical195": "#009E73", "cpsat_strong_w4": "#CC79A7"}
MARK = {"ls_nb": "o", "ls_py": "x", "cpsat_naive": "s", "cpsat_strong": "D", "glucose4": "^", "cadical195": "v", "cpsat_strong_w4": "P"}
FAM = {"bnq": "Blocked n-Queens (p blocked)", "qwh": "Quasigroup with holes (holes)", "qcp": "Quasigroup completion (preassigned)", "ksat": "Random 3-SAT (clauses/variables)"}
COMPLETE = ["cpsat_naive", "cpsat_strong", "cpsat_strong_w4", "glucose4", "cadical195"]
NB = 10000; RNG = np.random.default_rng(12345)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10, "figure.dpi": 150})
KEY = ["family", "size", "param", "seed"]

def load(path):
    d = pd.read_csv(path); d["status"] = d.status.astype(str)
    assert not (d.status.isin(["sat", "unsat", "bad"]) & (d.t > d["T"])).any(), "late answer left in CSV: harness strict rule violated"
    d["solved"] = (d.status == "sat") & (d.verified == 1); d["decided"] = d.solved | (d.status == "unsat")
    d["par2s"] = np.where(d.solved, d.t, 2 * d["T"]); d["par2d"] = np.where(d.decided, d.t, 2 * d["T"])
    d["param"] = d.param.astype(float); return d

def boot_ci(x):
    x = np.asarray(x, float); i = RNG.integers(0, len(x), (NB, len(x))); return np.percentile(x[i].mean(axis=1), [2.5, 97.5])

def cells(d):
    rows = []
    for (fam, n, p, sv), g in d.groupby(["family", "size", "param", "solver"]):
        r = dict(family=fam, size=n, param=p, solver=sv, n=len(g), solved=g.solved.mean(), decided=g.decided.mean(), par2s=g.par2s.mean(), par2d=g.par2d.mean(),
                 med_t_solved=g.t[g.solved].median() if g.solved.any() else np.nan, bad=int((g.status == "bad").sum()), err=int(g.status.str.startswith("err").sum()))
        r["solved_lo"], r["solved_hi"] = boot_ci(g.solved.values) if len(g) > 1 else (np.nan, np.nan); rows.append(r)
    return pd.DataFrame(rows)

def label_cells(d):
    """SAT-side: >= 90% of instances solved by some solver; UNSAT-side: >= 90% proven UNSAT by some solver; else transition. sat = share solved by any solver."""
    k = d.groupby(KEY).agg(sat=("solved", "any"), uns=("status", lambda s: (s == "unsat").any())).reset_index()
    c = k.groupby(["family", "size", "param"]).agg(sat=("sat", "mean"), uns=("uns", "mean"), n=("sat", "size")).reset_index()
    c["side"] = np.where(c.sat >= .9, "SAT", np.where(c.uns >= .9, "UNSAT", "transition")); return c

def sel(d, fam=None, n=None, p=None):
    m = pd.Series(True, index=d.index)
    if fam is not None: m &= d.family == fam
    if n is not None: m &= d["size"] == n
    if p is not None: m &= np.isclose(d.param, p)
    return m

def paired(d, m, a, b, col="solved"):
    x = d[m & (d.solver == a)].set_index(KEY)[col].astype(float); y = d[m & (d.solver == b)].set_index(KEY)[col].astype(float)
    z = (x - y.reindex(x.index)).dropna().values
    if len(z) < 2: return np.nan, np.nan, np.nan, len(z)
    lo, hi = boot_ci(z); return z.mean(), lo, hi, len(z)

def best_complete(cl, fam, n, p):
    g = cl[(cl.family == fam) & (cl["size"] == n) & np.isclose(cl.param, p) & cl.solver.isin(COMPLETE)]
    return (g.solved.max() if len(g) else np.nan), (g.loc[g.solved.idxmax(), "solver"] if len(g) else "-")

def fmt(e, lo, hi, k): return "no data" if k < 2 else f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={k})"

# ---------------------------------------------------------------- hypotheses (thresholds: DESIGN_NOTE_V3.md section 3)
def h1(d):
    o = ["## H1  bnq n=200, p in {.84,.86,.88,.90}: solved(ls_nb) - solved(cpsat_strong), paired 95% CI. Supported if lower bound >= 0.5 in at least 3 of 4 cells."]; ok = 0; k = 0
    for sv in ("ls_nb", "ls_py"):
        for p in (.84, .86, .88, .90):
            e, lo, hi, n = paired(d, sel(d, "bnq", 200, p), sv, "cpsat_strong"); o.append(f"- {sv} p={p}: {fmt(e, lo, hi, n)}")
            if sv == "ls_nb": ok += (lo >= .5) if n >= 2 else 0; k += n >= 2
    return o + [f"- VERDICT (ls_nb): {'supported' if ok >= 3 else 'falsified' if k == 4 else 'not evaluable (cells missing)'} ({ok}/4 cells)", ""]

def h2(d, cl, lab):
    o = ["## H2  family-specific LS advantage, on ls_py (v2 continuity). Falsified if ls_py is within 0.2 of the best complete solver on all SAT-side qwh (or all qcp) N=40 cells, or solves < 0.9 on ksat n=350 r=4.0. ls_nb shown for information (H8 is its test)."]
    res = {}
    for fam, n in (("qwh", 40), ("qcp", 40), ("ksat", 350)):
        sat = lab[(lab.family == fam) & (lab["size"] == n) & (lab.side == "SAT")].param.tolist(); gaps = []
        for p in sat:
            if fam == "ksat" and p > 4.2: continue
            b, _ = best_complete(cl, fam, n, p); s = cl[(cl.family == fam) & (cl["size"] == n) & np.isclose(cl.param, p)].set_index("solver").solved
            o.append(f"- {fam} n={n} p={p}: ls_py {s.get('ls_py', np.nan):.2f}, ls_nb {s.get('ls_nb', np.nan):.2f}, best complete {b:.2f}"); gaps.append(b - s.get("ls_py", np.nan))
        res[fam] = gaps
    lat = [any(g >= .2 for g in res[f]) if res[f] else None for f in ("qwh", "qcp")]
    k40 = cl[(cl.family == "ksat") & (cl["size"] == 350) & np.isclose(cl.param, 4.0) & (cl.solver == "ls_py")].solved
    o.append(f"- latin part (gap >= 0.2 in >= 1 SAT-side cell, qwh, qcp): {lat}; ksat n=350 r=4.0 ls_py solved: {k40.iloc[0] if len(k40) else 'no data'}")
    ok = all(x is True for x in lat) and len(k40) > 0 and k40.iloc[0] >= .9; ev = all(x is not None for x in lat) and len(k40) > 0
    return o + [f"- VERDICT: {'supported' if ok else 'falsified' if ev else 'not evaluable'}", ""]

def h3(d, lab):
    o = ["## H3  encoding: pooled solved(cpsat_strong) - solved(cpsat_naive) over SAT-side + transition cells (UNSAT-side cells excluded, labels from this run). Supported if bnq lower CI >= 0.1, latin (qwh+qcp) lower CI >= 0.1, and ksat CI inside (-0.1, 0.1)."]; v = {}
    for name, fams in (("bnq", ["bnq"]), ("latin", ["qwh", "qcp"]), ("ksat", ["ksat"])):
        keep = lab[lab.family.isin(fams) & (lab.side != "UNSAT")]; idx = keep.set_index(["family", "size", "param"]).index
        m = d.family.isin(fams) & d.set_index(["family", "size", "param"]).index.isin(idx)
        e, lo, hi, n = paired(d, m, "cpsat_strong", "cpsat_naive"); v[name] = (lo, hi, n); o.append(f"- {name}: {fmt(e, lo, hi, n)}; cells used {len(keep)}")
    ev = all(x[2] >= 2 for x in v.values()); ok = ev and v["bnq"][0] >= .1 and v["latin"][0] >= .1 and v["ksat"][0] > -.1 and v["ksat"][1] < .1
    return o + [f"- VERDICT: {'supported' if ok else 'falsified' if ev else 'not evaluable'}", ""]

def h4(cl):
    o = ["## H4  UNSAT side. ksat n=250, ratio >= 4.4: decided(cadical195) >= decided(cpsat_strong) + 0.1; bnq n=100 and 200, p >= .94: decided(cpsat_strong) >= decided(cadical195). Only cells where some complete solver decides >= 20%; supported if it holds in every such cell."]; res = []
    for fam, n, pm, rule in (("ksat", 250, 4.4, lambda c, s: c >= s + .1), ("bnq", 100, .94, lambda c, s: s >= c), ("bnq", 200, .94, lambda c, s: s >= c)):
        for p in sorted(cl[(cl.family == fam) & (cl["size"] == n) & (cl.param >= pm - 1e-9)].param.unique()):
            g = cl[(cl.family == fam) & (cl["size"] == n) & np.isclose(cl.param, p) & cl.solver.isin(COMPLETE)].set_index("solver").decided
            if g.max() < .2: o.append(f"- {fam} n={n} p={p}: skipped (max decided {g.max():.2f} < 0.2)"); continue
            c, s = g.get("cadical195", np.nan), g.get("cpsat_strong", np.nan); ok = bool(rule(c, s)); res.append(ok); o.append(f"- {fam} n={n} p={p}: cadical195 {c:.2f}, cpsat_strong {s:.2f} -> {'holds' if ok else 'fails'}")
    return o + [f"- VERDICT: {'supported' if res and all(res) else 'falsified' if res else 'not evaluable'}", ""]

def h5(cl, d):
    o = ["## H5  hardness peak. Per family at the largest size: s(param) = share of instances solved by some solver. A crossing exists only if min s < 0.5 <= max s over the grid; otherwise NOT EVALUABLE (qwh is always SAT by construction: declared not evaluable in advance). Peak = argmax over param of min over complete solvers of PAR2(decision); NOT EVALUABLE if that max is >= 0.95*2T in more than 2 cells (plateau). Crossing cell = grid point with s closest to 0.5. Supported if |peak index - crossing index| <= 1 (ksat n=350: peak in {4.2, 4.26, 4.3}); H5 as a whole supported if every evaluable family is supported and >= 2 families are evaluable."]; res = []
    for fam in FAM:
        g = d[d.family == fam]
        if g.empty: continue
        n = g["size"].max(); gg = g[g["size"] == n]
        s = gg.groupby(KEY).solved.any().groupby(["family", "size", "param"]).mean().droplevel([0, 1]); two = 2 * gg["T"].iloc[0]
        c = cl[(cl.family == fam) & (cl["size"] == n) & cl.solver.isin(COMPLETE)].pivot(index="param", columns="solver", values="par2d").min(axis=1)
        plateau = (c >= .95 * two).sum() > 2
        if fam == "qwh" or not (s.min() < .5 <= s.max()) or plateau:
            why = "always SAT by construction" if fam == "qwh" else "no 0.5 crossing in grid" if not (s.min() < .5 <= s.max()) else "PAR2 plateau at 2T"
            o.append(f"- {fam} n={n}: NOT EVALUABLE ({why}); s range [{s.min():.2f}, {s.max():.2f}], peak {c.idxmax()}"); continue
        ps = list(s.index); cross = (s - .5).abs().idxmin(); peak = c.idxmax(); dist = abs(ps.index(peak) - ps.index(cross)); res.append(dist <= 1)
        o.append(f"- {fam} n={n}: peak {peak}, crossing {cross}, distance {dist} -> {'holds' if dist <= 1 else 'fails'}")
    return o + [f"- VERDICT: {'supported' if len(res) >= 2 and all(res) else 'falsified' if res and not all(res) else 'not evaluable'} ({len(res)} families evaluable)", ""]

def h6(d, stage):
    o = [f"## H6 ({stage})  bnq n=200 cells where ls_nb solves every instance: cpsat_strong_w4 solves < 50%. Falsified if w4 solves >= 50% in two or more such cells."]; bad = 0; k = 0
    for p in sorted(d[(d.family == "bnq") & (d["size"] == 200)].param.unique()):
        m = sel(d, "bnq", 200, p); l = d[m & (d.solver == "ls_nb")].solved; w = d[m & (d.solver == "cpsat_strong_w4")].solved
        if len(l) and l.all() and len(w): k += 1; bad += w.mean() >= .5; o.append(f"- p={p}: ls_nb {l.mean():.2f} (n={len(l)}), w4 solved {w.mean():.2f} (n={len(w)})")
    return o + [f"- VERDICT: {'falsified' if bad >= 2 else 'supported' if k else 'not evaluable (no such cell or no w4 data)'} ({bad} of {k} cells with w4 >= 50%)", ""]

def h7(cl):
    c = cl.pivot_table(index=["family", "size", "param"], columns="solver", values="decided")
    if not {"cadical195", "glucose4"} <= set(c.columns): return ["## H7  no glucose4 data", ""]
    f = (c.cadical195 >= c.glucose4).mean()
    return ["## H7 (exploratory)  cadical195 decided >= glucose4 in >= 75% of cells (ties count).", f"- {f:.0%} of {len(c)} cells", f"- VERDICT: {'supported' if f >= .75 else 'falsified'}", ""]

def h8(d, cl, lab, stage):
    o = [f"## H8 ({stage})  tuned compiled LS closes the Latin-square gap. Over SAT-side cells of a family at N in {{30,40}}: share of cells with solved(ls_nb) >= 0.8 * best complete solved is >= 0.75. Supported if true for both qwh and qcp; partial if one; falsified if neither."]; ok = {}
    for fam in ("qwh", "qcp"):
        cs = lab[(lab.family == fam) & lab["size"].isin([30, 40]) & (lab.side == "SAT")]; hit = []
        for _, r in cs.iterrows():
            b, who = best_complete(cl, fam, r["size"], r.param); l = cl[(cl.family == fam) & (cl["size"] == r["size"]) & np.isclose(cl.param, r.param) & (cl.solver == "ls_nb")].solved
            if not len(l): continue
            hit.append(l.iloc[0] >= .8 * b); o.append(f"- {fam} N={r['size']} p={r.param}: ls_nb {l.iloc[0]:.2f}, best complete {b:.2f} ({who}) -> {'holds' if hit[-1] else 'fails'}")
        ok[fam] = (np.mean(hit) >= .75) if hit else None; o.append(f"- {fam}: {sum(hit)}/{len(hit)} SAT-side cells hold")
    v = [x for x in ok.values() if x is not None]
    out = "not evaluable" if len(v) < 2 else "supported" if all(v) else "partial (one family)" if any(v) else "falsified"
    return o + [f"- VERDICT: {out}", ""]

# ---------------------------------------------------------------- tables / figures
def tables(d, cl, lab, path):
    with open(path, "w") as f:
        f.write(f"rows {len(d)}; status {d.status.value_counts().to_dict()}; bad {int((d.status == 'bad').sum())}; err {int(d.status.str.startswith('err').sum())}; killed {int(d.killed.sum())}\n\n")
        f.write(f"instances per cell: {d.groupby(['family', 'size', 'param', 'solver']).size().describe()[['min', 'max']].to_dict()}\n\n")
        for fam in FAM:
            c = cl[cl.family == fam]
            if c.empty: continue
            t = c.assign(v=lambda x: x.solved.map("{:.2f}".format) + " (" + x.med_t_solved.map(lambda v: "-" if np.isnan(v) else f"{v:.2f}s") + ")").pivot(index=["size", "param"], columns="solver", values="v")
            f.write(f"### {fam}: fraction solved (median time of solved runs)\n\n{t.to_markdown()}\n\n")
        f.write("### cell labels\n\n" + lab.round(2).to_markdown(index=False) + "\n")

def fig_main(cl, out1, out2):
    fams = [f for f in FAM if f in set(cl.family)]; sizes = {f: sorted(cl[cl.family == f]["size"].unique()) for f in fams}
    fig, ax = plt.subplots(len(fams), 3, figsize=(15, 3.6 * len(fams)), squeeze=False)
    for i, f in enumerate(fams):
        for j in range(3):
            if j >= len(sizes[f]): ax[i, j].axis("off"); continue
            a = ax[i, j]; n = sizes[f][j]
            for sv, g in cl[(cl.family == f) & (cl["size"] == n)].groupby("solver"):
                g = g.sort_values("param"); a.plot(g.param, g.solved, marker=MARK[sv], color=COLOR[sv], label=sv, lw=1.6, ms=6); a.fill_between(g.param, g.solved_lo, g.solved_hi, color=COLOR[sv], alpha=.12)
            a.set_ylim(-.03, 1.03); a.set_title(f"{FAM[f].split(' (')[0]}, size {n}"); a.set_xlabel(FAM[f].split("(")[1].rstrip(")")); a.set_ylabel("fraction solved")
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(.5, 1.0)); fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out1, dpi=200); plt.close(fig)
    fig, ax = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.8), squeeze=False)
    for a, f in zip(ax[0], fams):
        n = cl[cl.family == f]["size"].max()
        for sv, g in cl[(cl.family == f) & (cl["size"] == n)].groupby("solver"):
            g = g.sort_values("param"); a.plot(g.param, g.par2d, marker=MARK[sv], color=COLOR[sv], label=sv, lw=1.6)
        a.set_title(f"{f}, size {n}"); a.set_xlabel(FAM[f].split("(")[1].rstrip(")")); a.set_ylabel("PAR2 (s), decision")
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(out2, dpi=200); plt.close(fig)

def fig_long(cl, T, out):
    c = cl.assign(cell=lambda x: x.family + " " + x["size"].astype(str) + " " + x.param.astype(str)); cs = list(dict.fromkeys(c.cell)); svs = [s for s in COLOR if s in set(c.solver)]
    fig, ax = plt.subplots(figsize=(max(8, .55 * len(cs) * len(svs) / 2), 4.2)); w = .8 / len(svs)
    for k, sv in enumerate(svs):
        g = c[c.solver == sv].set_index("cell").reindex(cs); ax.bar(np.arange(len(cs)) + k * w, g.solved, w, color=COLOR[sv], label=sv)
    ax.set_xticks(np.arange(len(cs)) + .4); ax.set_xticklabels(cs, rotation=70, ha="right", fontsize=8); ax.set_ylabel("fraction solved"); ax.set_title(f"T = {T:g} s"); ax.legend(ncol=4); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def fig_h8(cl, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), squeeze=False)
    for a, f in zip(ax[0], ("qwh", "qcp")):
        n = cl[cl.family == f]["size"].max()
        for sv in ("ls_nb", "ls_py", "cpsat_strong", "cadical195"):
            g = cl[(cl.family == f) & (cl["size"] == n) & (cl.solver == sv)].sort_values("param")
            if len(g): a.plot(g.param, g.solved, marker=MARK[sv], color=COLOR[sv], label=sv, lw=1.6)
        a.set_title(f"{f}, N={n}"); a.set_xlabel(FAM[f].split("(")[1].rstrip(")")); a.set_ylabel("fraction solved"); a.set_ylim(-.03, 1.03)
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def env_table(paths):
    rows = []
    for st, p in paths.items():
        f = p + ".env.json"
        if not os.path.exists(f): rows.append({"stage": st, "note": "env.json missing"}); continue
        for i, e in enumerate(json.load(open(f))):
            rows.append({"stage": st, "start": i, **{k: e.get(k) for k in ("python", "ortools", "python-sat", "numpy", "numba", "cpu_model", "cpu_count_usable", "ram_gb", "os", "T", "seeds", "jobs", "start_utc")}})
    return pd.DataFrame(rows)

MIN_N = {"main10": 8, "long60": 4, "long300": 3}

def testable(d, stage):
    """drop cells where any solver has fewer than MIN_N[stage] completed instances (truncated run); tables still show them."""
    n = d.groupby(["family", "size", "param", "solver"]).size().groupby(["family", "size", "param"]).min().rename("m").reset_index()
    k = d.merge(n, on=["family", "size", "param"]); print(f"{stage}: {int((n.m < MIN_N[stage]).sum())} cells excluded from tests (< {MIN_N[stage]} instances)"); return k[k.m >= MIN_N[stage]].drop(columns="m")

def main():
    ap = argparse.ArgumentParser(); [ap.add_argument(f"--{s}") for s in ("main10", "long60", "long300")]; ap.add_argument("--out", default="analysis_v3"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    paths = {s: getattr(a, s) for s in ("main10", "long60", "long300") if getattr(a, s)}; D = {s: load(p) for s, p in paths.items()}; CL, LAB = {}, {}
    for s, d in D.items():
        CL[s] = cells(d); LAB[s] = label_cells(d); CL[s].to_csv(f"{a.out}/cells_{s}.csv", index=False); tables(D[s], CL[s], LAB[s], f"{a.out}/tables_{s}.md")
    open(f"{a.out}/environment.md", "w").write(env_table(paths).to_markdown(index=False))
    T_ = {s: t for s, t in ((s, testable(d, s)) for s, d in D.items()) if len(t)}; TCL = {s: cells(T_[s]) for s in T_}; TLAB = {s: label_cells(T_[s]) for s in T_}
    o = ["# nq_v3 hypothesis checks (definitions: DESIGN_NOTE_V3.md; every verdict is reported, none dropped)\n"]
    if "main10" in T_:
        d, cl, lab = T_["main10"], TCL["main10"], TLAB["main10"]; o += ["# Stage main10 (T=10 s)\n"] + h1(d) + h2(d, cl, lab) + h3(d, lab) + h4(cl) + h5(cl, d) + h7(cl) + h8(d, cl, lab, "main10")
        fig_main(CL["main10"], f"{a.out}/fig_solved_main10.png", f"{a.out}/fig_par2_main10.png"); fig_h8(CL["main10"], f"{a.out}/fig_h8.png")
    for s in ("long60", "long300"):
        if s in T_: o += [f"# Stage {s} (T={D[s]['T'].iloc[0]:g} s)\n"] + h6(T_[s], s) + (h8(T_[s], TCL[s], TLAB[s], s) if s == "long60" else []); fig_long(CL[s], D[s]["T"].iloc[0], f"{a.out}/fig_long_{s}.png")
    open(f"{a.out}/hypotheses.md", "w").write("\n".join(o)); print("\n".join(o)); print("wrote", a.out)

if __name__ == "__main__": main()
