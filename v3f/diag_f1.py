"""diag_f1: F1 start-up cost diagnosis (about 10 CPU minutes, serial, run BEFORE nq_v3f.py main3f; counts as the first non-smoke run of v3f).
Question: why does the first call of bnq_init / bnq_steps in a forked child cost about 1.1 s although the parent warmed up on a real-size board?
Every condition is a fresh parent process (so the numba cache env var and the kernel variant are set before numba loads). The parent warms up, measures
one control pass itself, then runs REPS forked children (the harness way) and SPAWNS fresh interpreters. Each process times, around each kernel call:
nb_seed, bnq_init, bnq_steps (first call) and again init/steps (second call), and counts numba compile events and dispatcher signatures before/after.
One row per process is appended to the side file f1_diag.csv. Then the fix rule (fixed in DESIGN_NOTE_V3F.md section 4) writes f1_fix.json.
Conditions (kern = kernel variant, warm = warm-up in the parent, cache = NUMBA_CACHE_DIR set):
  base_none  lazy kernels, no warm-up            warm_tiny  lazy, n=8 warm-up (nq_v3 warm_up)      warm_real  lazy, n=200 warm-up via the real loop (v3e)
  warm_same  lazy, warm-up = direct calls with the exact argument types of the real call (candidate fix 'same_call')
  sig        explicit eager signatures (candidate 'sig')       cache_cold / cache_warm   cache=True + NUMBA_CACHE_DIR (candidate 'cache'; cold = empty dir)
Usage: python diag_f1.py [--reps 5] [--spawns 3] [--outdir .]       python diag_f1.py --selftest      (pure python: summary + fix rule on fake rows)"""
import os, sys, csv, json, time, random, shutil, statistics, subprocess, argparse, contextlib, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["cond", "mode", "rep", "pid", "ppid", "t_build", "t_seed", "t_init1", "t_steps1", "t_init2", "t_steps2",
          "sig_init_before", "sig_init_after", "sig_steps_before", "sig_steps_after", "compile_events", "numba", "note"]
COND = {"base_none": dict(kern="lazy", warm="none", cache=False, spawn=True), "warm_tiny": dict(kern="lazy", warm="tiny", cache=False, spawn=False),
        "warm_real": dict(kern="lazy", warm="real", cache=False, spawn=False), "warm_same": dict(kern="lazy", warm="same", cache=False, spawn=False),
        "sig": dict(kern="sig", warm="none", cache=False, spawn=True), "cache_cold": dict(kern="cache", warm="real", cache=True, spawn=False),
        "cache_warm": dict(kern="cache", warm="real", cache=True, spawn=True)}
ORDER = ["base_none", "warm_tiny", "warm_real", "warm_same", "sig", "cache_cold", "cache_warm"]     # cache_cold must precede cache_warm
CANDIDATES = [("same_call", ("warm_same",)), ("sig", ("sig",)), ("cache", ("cache_cold", "cache_warm"))]   # preference order of the fix rule
THR_FIRST = 0.10        # fix rule: median first-call sum (nb_seed + init + steps) of the forked children <= 0.10 s (all listed conditions)
MIN_CHILDREN = 5
SIG_INIT = "void(int64[:,::1], int64[::1], int64[::1], int64[::1], int64[::1])"
SIG_STEPS = "int64(int64[:,::1], int64[::1], int64[::1], int64[::1], int64[::1], float64, int64)"

# ---------------------------------------------------------------- pure-python part: summary and fix rule (also used by analyze_v3f.py)
def first_sum(r): return float(r["t_seed"]) + float(r["t_init1"]) + float(r["t_steps1"])

def summarize(rows):
    S = {}
    for r in rows:
        S.setdefault((r["cond"], r["mode"]), []).append(r)
    out = {}
    for k, rs in S.items():
        fs = [first_sum(r) for r in rs]; ce = [int(float(r["compile_events"])) for r in rs]
        grow = [int(float(r["sig_init_after"])) - int(float(r["sig_init_before"])) + int(float(r["sig_steps_after"])) - int(float(r["sig_steps_before"])) for r in rs]
        out[k] = dict(n=len(rs), median=statistics.median(fs), min=min(fs), max=max(fs), median_second=statistics.median(float(r["t_init2"]) + float(r["t_steps2"]) for r in rs),
                      median_compile_events=statistics.median(ce), median_sig_growth=statistics.median(grow), median_build=statistics.median(float(r["t_build"]) for r in rs))
    return out

def choose_fix(summary):
    """rule: the FIRST candidate (order same_call, sig, cache) whose forked-child median first-call sum is <= THR_FIRST in every one of its conditions (>= MIN_CHILDREN children each)
    is the fix; none -> fix = None. 'baseline_reproduced' = warm_real median > THR_FIRST (status quo is slow), reported, does not change the rule."""
    base = summary.get(("warm_real", "fork")); res = dict(fix=None, thr=THR_FIRST, baseline_median=base["median"] if base else None,
                                                           baseline_reproduced=bool(base and base["median"] > THR_FIRST), medians={}, rule="first of same_call, sig, cache with fork-child median first-call sum <= %.2f s" % THR_FIRST)
    for k, v in summary.items(): res["medians"]["%s/%s" % k] = round(v["median"], 4)
    for name, conds in CANDIDATES:
        ok = all(summary.get((c, "fork")) and summary[(c, "fork")]["n"] >= MIN_CHILDREN and summary[(c, "fork")]["median"] <= THR_FIRST for c in conds)
        if ok and res["fix"] is None: res["fix"] = name
    return res

def read_rows(path): return list(csv.DictReader(open(path)))

def selftest():
    def fake(cond, mode, v, n=5, ce=0, grow=0):
        return [dict(cond=cond, mode=mode, rep=i, t_build=0, t_seed=0.0001, t_init1=v / 2, t_steps1=v / 2, t_init2=0.001, t_steps2=0.002, sig_init_before=1, sig_init_after=1 + grow,
                     sig_steps_before=1, sig_steps_after=1, compile_events=ce) for i in range(n)]
    rows = fake("warm_real", "fork", 1.1, ce=2, grow=1) + fake("warm_same", "fork", 1.05) + fake("sig", "fork", 0.01) + fake("cache_cold", "fork", 0.01) + fake("cache_warm", "fork", 0.2)
    r = choose_fix(summarize(rows)); assert r["fix"] == "sig" and r["baseline_reproduced"], r
    rows2 = rows + fake("warm_same", "fork", 0.02, n=5)       # a faster same_call in the pool pushes the median down only if it dominates
    assert choose_fix(summarize(fake("warm_real", "fork", 1.1) + fake("warm_same", "fork", 0.01) + fake("sig", "fork", 0.01)))["fix"] == "same_call"
    assert choose_fix(summarize(fake("warm_real", "fork", 1.1) + fake("warm_same", "fork", 0.5) + fake("sig", "fork", 0.4, n=3)))["fix"] is None   # too few children / too slow
    assert choose_fix(summarize(fake("warm_real", "fork", 1.1) + fake("cache_cold", "fork", 0.01) + fake("cache_warm", "fork", 0.01)))["fix"] == "cache"
    assert choose_fix(summarize(fake("warm_real", "fork", 0.01) + fake("warm_same", "fork", 0.01)))["baseline_reproduced"] is False
    print("diag_f1 SELFTEST OK")

# ---------------------------------------------------------------- numba part (imports inside functions: this file is importable without numba)
def compile_events():
    try:
        from numba.core import event
        cm = event.install_recorder("numba:compile")
        return cm if hasattr(cm, "__enter__") else contextlib.nullcontext(None)
    except Exception: return contextlib.nullcontext(None)

def count_events(rec):
    try: return sum(1 for _, ev in rec.buffer if ev.is_start) if rec is not None else -1
    except Exception: return -1

def kernels(kern):
    """(init, steps, seed) kernels of the variant; eager compile for 'sig' happens here (its cost is the t_build column)"""
    from numba import njit
    import nq_v3 as m
    if kern == "sig": return njit(SIG_INIT)(m.bnq_init.py_func), njit(SIG_STEPS)(m.bnq_steps.py_func), m.nb_seed
    if kern == "cache": return njit(cache=True)(m.bnq_init.py_func), njit(cache=True)(m.bnq_steps.py_func), m.nb_seed
    return m.bnq_init, m.bnq_steps, m.nb_seed

def board(rep, n=200, p=0.9):
    import nq_v3 as m
    return m.gen_bnq(n, p, random.Random(f"diag-{rep}"))

def arrays(B):
    import numpy as np
    n = len(B); return B.astype(np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(2 * n, np.int64), np.zeros(2 * n, np.int64)

def run_loop(K, B, T, seed=0):
    """lsnb_bnq of nq_v3.py with the kernels K (the 'real' warm-up path)"""
    import numpy as np, nq_v3 as m
    init, steps, nbs = K; n = len(B); H = B.astype(np.int64); nbs(seed); pos = np.zeros(n, np.int64); col = np.zeros(n, np.int64); d1 = np.zeros(2 * n, np.int64); d2 = np.zeros(2 * n, np.int64)
    t_end = time.monotonic() + T; noise = m.NB_NOISE["bnq"]
    while time.monotonic() < t_end:
        init(H, pos, col, d1, d2); left = m.NB_RESTART["bnq"]
        while left > 0:
            if steps(H, pos, col, d1, d2, noise, 500): return
            left -= 500
            if time.monotonic() > t_end: return

def warm(K, how):
    import numpy as np, nq_v3 as m
    r = random.Random(1)
    if how == "tiny": run_loop(K, m.gen_bnq(8, .3, r), 0.5)
    elif how == "real": run_loop(K, m.gen_bnq(200, .9, r), 0.3)
    elif how == "same":       # direct calls, argument types exactly as in the real call: int64 C arrays, float noise, int steps
        H, pos, col, d1, d2 = arrays(m.gen_bnq(200, .9, r)); K[2](0); K[0](H, pos, col, d1, d2); K[1](H, pos, col, d1, d2, m.NB_NOISE["bnq"], 500)

def measure(K, B, cond, mode, rep, t_build, note=""):
    import numba
    init, steps, nbs = K; H, pos, col, d1, d2 = arrays(B); si = len(getattr(init, "signatures", ())); ss = len(getattr(steps, "signatures", ()))
    pc = time.perf_counter
    with compile_events() as rec:
        t = pc(); nbs(5); t_seed = pc() - t
        t = pc(); init(H, pos, col, d1, d2); t_init1 = pc() - t
        t = pc(); steps(H, pos, col, d1, d2, 0.0, 500); t_steps1 = pc() - t
    ce = count_events(rec); si2 = len(getattr(init, "signatures", ())); ss2 = len(getattr(steps, "signatures", ()))
    t = pc(); init(H, pos, col, d1, d2); t_init2 = pc() - t
    t = pc(); steps(H, pos, col, d1, d2, 0.0, 500); t_steps2 = pc() - t
    return dict(cond=cond, mode=mode, rep=rep, pid=os.getpid(), ppid=os.getppid(), t_build=round(t_build, 5), t_seed=round(t_seed, 5), t_init1=round(t_init1, 5), t_steps1=round(t_steps1, 5),
                t_init2=round(t_init2, 5), t_steps2=round(t_steps2, 5), sig_init_before=si, sig_init_after=si2, sig_steps_before=ss, sig_steps_after=ss2, compile_events=ce, numba=numba.__version__, note=note)

def emit(path, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    line = ",".join(str(row.get(k, "")) for k in FIELDS) + "\n"
    if new: os.write(fd, (",".join(FIELDS) + "\n").encode())
    os.write(fd, line.encode()); os.close(fd)

def fork_child(K, cond, rep, out):
    emit(out, measure(K, board(rep), cond, "fork", rep, 0.0))

def parent(cond, reps, spawns, out):
    import multiprocessing as mp
    spec = COND[cond]; t = time.perf_counter(); K = kernels(spec["kern"]); tb = time.perf_counter() - t
    t = time.perf_counter(); warm(K, spec["warm"]); tw = time.perf_counter() - t
    emit(out, measure(K, board(99), cond, "parent", 0, tb, note=f"warm_s={tw:.3f}"))      # control: the parent's own first measured call after its warm-up
    ctx = mp.get_context("fork")
    for rep in range(reps):
        p = ctx.Process(target=fork_child, args=(K, cond, rep, out)); p.start(); p.join()
    if spec["spawn"]:
        for rep in range(spawns): subprocess.run([sys.executable, os.path.abspath(__file__), "--child", cond, str(rep), out], check=True)

def spawn_child(cond, rep, out):
    spec = COND[cond]; t = time.perf_counter(); K = kernels(spec["kern"]); tb = time.perf_counter() - t       # fresh interpreter, no warm-up
    emit(out, measure(K, board(rep), cond, "spawn", rep, tb))

def types_check(out):
    """argument types of the REAL call (lsnb_bnq on n=200 and on the n=8 warm-up board) as seen by numba, via spies on the two kernels"""
    import numba, nq_v3 as m
    seen = []; oi, os_ = m.bnq_init, m.bnq_steps
    m.bnq_init = lambda *a: (seen.append(("init", len(a[0]), tuple(str(numba.typeof(x)) for x in a))), oi(*a))[1]
    m.bnq_steps = lambda *a: (seen.append(("steps", len(a[0]), tuple(str(numba.typeof(x)) for x in a))), os_(*a))[1]
    r = random.Random(1); m.lsnb_bnq(m.gen_bnq(8, .3, r), 0.2, 0); m.lsnb_bnq(m.gen_bnq(200, .9, r), 0.2, 0)
    m.bnq_init, m.bnq_steps = oi, os_
    by = {}
    for kind, n, ty in seen: by.setdefault(kind, set()).add(ty)
    res = {k: sorted(map(list, v)) for k, v in by.items()}; res["all_types_identical_across_boards"] = all(len(v) == 1 for v in by.values())
    json.dump(res, open(out, "w"), indent=1); print("types:", json.dumps(res), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--spawns", type=int, default=3); ap.add_argument("--outdir", default=".")
    ap.add_argument("--child", nargs=3, metavar=("COND", "REP", "OUT")); ap.add_argument("--parent", nargs=2, metavar=("COND", "OUT")); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--types", action="store_true")
    a = ap.parse_args(); sys.path.insert(0, HERE)
    if a.selftest: return selftest()
    if a.child: return spawn_child(a.child[0], int(a.child[1]), a.child[2])
    if a.parent: return parent(a.parent[0], a.reps, a.spawns, a.parent[1])
    if a.types: return types_check(os.path.join(a.outdir, "f1_types.json"))
    out = os.path.join(a.outdir, "f1_diag.csv"); cache = os.path.join(os.path.abspath(a.outdir), "f1_numba_cache"); shutil.rmtree(cache, ignore_errors=True)
    if os.path.exists(out): sys.exit(out + " exists: diagnosis is run once (move it away to repeat)")
    t0 = time.time(); subprocess.run([sys.executable, os.path.abspath(__file__), "--types", "--outdir", a.outdir], check=True)
    for cond in ORDER:
        env = dict(os.environ); env.pop("NUMBA_CACHE_DIR", None)
        if COND[cond]["cache"]: env["NUMBA_CACHE_DIR"] = cache
        subprocess.run([sys.executable, os.path.abspath(__file__), "--parent", cond, out, "--reps", str(a.reps), "--spawns", str(a.spawns)], env=env, check=True)
        print(f"condition {cond} done, {time.time() - t0:.0f} s", flush=True)
    S = summarize(read_rows(out)); res = choose_fix(S); res["numba_cache_dir_for_cache_fix"] = "set by nq_v3f.py (HERE/numba_cache)"
    for k, v in sorted(S.items()): print(f"{k[0]:11s} {k[1]:6s} n={v['n']:2d} first-call median {v['median']:.4f} s (min {v['min']:.4f}, max {v['max']:.4f}); second-call median {v['median_second']:.4f}; "
                                          f"compile events {v['median_compile_events']}; signature growth {v['median_sig_growth']}; build {v['median_build']:.3f}")
    json.dump(res, open(os.path.join(a.outdir, "f1_fix.json"), "w"), indent=1); print("F1 fix rule result:", json.dumps({k: res[k] for k in ("fix", "baseline_median", "baseline_reproduced")}), f"{time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
