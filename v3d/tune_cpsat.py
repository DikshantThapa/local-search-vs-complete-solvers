"""Tuning for v3d, run BEFORE the confirmatory stage. Tuning seeds 2000-2002 only (outside every stage: v3 used 0-19, v3c 100-119, v3d 200-219, v3 LS tuning 1000-1002).
Part 'cpsat': 8 documented single-parameter CP-SAT variants (nq_v3d.CP_VARIANTS) per family on 3 cells x 3 seeds, T_tune = 5 s, strong encoding, 1 worker.
  Winner = most solved runs, tie -> lower PAR2, tie -> 'default'. A non-default winner must beat default by >= 1 solved run or >= 10% PAR2, else 'default' is kept.
Part 'ls': tabu tenure {5,10,20,40} x stall {50,200} (units N steps) for ls_tabu_nb on qwh/qcp tuning cells. probSAT constants and ls_nb constants are NOT re-tuned here.
Writes tuned_cpsat.json, tuned_ls.json and tune_log.csv beside this file.  Usage: python tune_cpsat.py [--part cpsat|ls|all] [--jobs 4] [--T 5]"""
import argparse, csv, json, os, sys, time, multiprocessing as mp
import nq_v3d as d
m = d.m
CELLS = {"bnq": [(100, .92), (200, .88), (200, .94)], "qwh": [(30, .40), (40, .35), (40, .45)], "qcp": [(30, .42), (40, .42), (40, .46)],
         "ksat": [(250, 4.2), (350, 4.26), (350, 4.3)], "ksat5": [(150, 21.0), (200, 20.5), (200, 21.5)]}
LS_CELLS = [("qwh", 40, .35), ("qwh", 40, .40), ("qcp", 30, .42), ("qcp", 40, .42)]
SEEDS = (2000, 2001, 2002)
_cache = {}

def one(job):
    kind, fam, n, p, seed, spec, T = job
    key = (fam, n, p, seed)
    if key not in _cache: _cache.clear(); _cache[key] = d.instance(*key)
    inst = _cache[key]; t0 = time.monotonic()
    if kind == "cp": st, sol = d.cp_with(fam, inst, T, seed, d.CP_VARIANTS[spec], t0)
    else: st, sol = d.lstabu_latin(inst, T, seed * 7919 + 1, t0, tenure=spec[0], stall=spec[1])
    t = time.monotonic() - t0; ok = st == "sat" and t <= T and bool(m.VERIFY[fam](inst, sol))
    return job, int(ok), min(t, T) if ok else 2 * T

def run(jobs, nproc):
    jobs = sorted(jobs, key=lambda j: j[1:5])   # same instance consecutive per worker chunk
    with mp.get_context("fork").Pool(nproc) as pool: return pool.map(one, jobs, chunksize=1)

def validate_variants():
    from ortools.sat.python import cp_model
    from google.protobuf import text_format
    ok = {}
    for k, v in d.CP_VARIANTS.items():
        try: text_format.Merge(v, cp_model.CpSolver().parameters); ok[k] = v
        except Exception as e: print("variant dropped (not accepted by this ortools):", k, e, flush=True)
    return ok

def tune_cp(a, log):
    d.CP_VARIANTS = validate_variants(); m.warm_up(); out = {}
    for fam, cells in CELLS.items():
        res = run([("cp", fam, n, p, s, v, a.T) for n, p in cells for s in SEEDS for v in d.CP_VARIANTS], a.jobs); agg = {}
        for (kind, f, n, p, s, v, T), ok, t in res:
            x = agg.setdefault(v, [0, 0.0]); x[0] += ok; x[1] += t; log.writerow(["cp", f, n, p, s, v, ok, round(t, 3)])
        best = min(agg, key=lambda v: (-agg[v][0], agg[v][1], v != "default"))
        if best != "default" and not (agg[best][0] >= agg["default"][0] + 1 or agg[best][1] <= 0.9 * agg["default"][1]): best = "default"
        out[fam] = {"variant": best, "params": d.CP_VARIANTS[best], "table": {v: {"solved": x[0], "par2_sum": round(x[1], 2)} for v, x in agg.items()}}
        print(fam, "->", best, {v: x[0] for v, x in agg.items()}, flush=True)
    out["_meta"] = {"T_tune": a.T, "seeds": SEEDS, "cells": CELLS, "runs_per_family": len(next(iter(CELLS.values()))) * len(SEEDS) * len(d.CP_VARIANTS)}
    json.dump(out, open(os.path.join(d.HERE, "tuned_cpsat.json"), "w"), indent=1)

def tune_ls(a, log):
    m.warm_up(); grid = [(t, s) for t in (5, 10, 20, 40) for s in (50, 200)]
    res = run([("ls", f, n, p, s, g, a.T) for f, n, p in LS_CELLS for s in SEEDS for g in grid], a.jobs); agg = {}
    for (kind, f, n, p, s, g, T), ok, t in res:
        x = agg.setdefault(g, [0, 0.0]); x[0] += ok; x[1] += t; log.writerow(["ls", f, n, p, s, f"{g[0]}/{g[1]}", ok, round(t, 3)])
    best = min(agg, key=lambda g: (-agg[g][0], agg[g][1], g != (10, 50)))
    json.dump({"tabu_tenure": best[0], "tabu_stall": best[1], "table": {f"{g[0]}/{g[1]}": {"solved": x[0], "par2_sum": round(x[1], 2)} for g, x in agg.items()},
               "T_tune": a.T, "seeds": SEEDS, "cells": LS_CELLS}, open(os.path.join(d.HERE, "tuned_ls.json"), "w"), indent=1)
    print("ls_tabu_nb ->", best, {f"{g[0]}/{g[1]}": x[0] for g, x in agg.items()}, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all", choices=("cpsat", "ls", "all"))
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--T", type=float, default=5.0); a = ap.parse_args()
    f = open(os.path.join(d.HERE, "tune_log.csv"), "a", newline=""); log = csv.writer(f)
    if os.path.getsize(f.name) == 0: log.writerow(["part", "family", "size", "param", "seed", "setting", "solved", "t_or_2T"])
    t = time.time()
    if a.part in ("cpsat", "all"): tune_cp(a, log); f.flush()
    if a.part in ("ls", "all"): tune_ls(a, log); f.flush()
    print(f"tuning done in {time.time() - t:.0f} s", flush=True)
