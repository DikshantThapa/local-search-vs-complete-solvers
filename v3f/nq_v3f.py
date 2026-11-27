"""nq_v3f: fifth and LAST small pre-registered stage on FRESH seeds (400-419). Wraps nq_v3.py, nq_v3d.py, nq_v3e.py (all unchanged, copied beside this file).
Design: DESIGN_NOTE_V3F.md. Questions F1-F5 (CRITIC3_v8 rejection reasons 2-4):
 F1 start-up cost: diag_f1.py (run BEFORE the stage) picks a fix by a fixed rule and writes f1_fix.json; solver 'ls_nb_fixed' = ls_nb with that fix.
 F2 5-SAT WalkSAT sanity: ls_nbk at r 12-18 and a noise sweep (ls_nbk_n30/n40/n50/n65; plain ls_nbk = 0.567) at r 20, 20.5.
 F3 noise x speed: 'ls_py_n0' = ls_py (lspy_latin) with noise 0.0, nothing else changed; ls_py (noise 0.1), ls_nb, ls_nb_n10 on the same instances.
 F4 CP-SAT second bnq model 'cpsat_alt' (full-domain integer per row, blocked cells forbidden by AddElement), also with 4 workers (_w4).
 F5 interpreted ls_py on bnq.
Usage: python nq_v3f.py OUT.csv --stage main3f|lat3f|long3f|smoke3f [--jobs 4] [--hours H] [--seeds K] [--solvers a,b]"""
import sys, os, json, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
FIX_PATH = os.environ.get("NQ_F1_FIX", os.path.join(HERE, "f1_fix.json"))
FIXINFO = json.load(open(FIX_PATH)) if os.path.exists(FIX_PATH) else {}
FIX = FIXINFO.get("fix")                       # None | "same_call" | "sig" | "cache"
if FIX == "cache": os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "numba_cache"))   # must be set before numba is imported
import numpy as np
from numba import njit
import nq_v3e as e
import nq_v3d as d
import nq_v3 as m

d.OFFSET.update({"main3f": 400, "lat3f": 400, "long3f": 400, "smoke3f": 5000})   # instance seed = index + OFFSET; lat3f and long3f share instances
SOLVERS = ("ls_nb", "ls_nb_fixed", "ls_py", "ls_py_n0", "ls_nb_n10", "ls_nbk", "ls_nbk_n30", "ls_nbk_n40", "ls_nbk_n50", "ls_nbk_n65", "probsat_nb", "cadical195",
           "cpsat_strong", "cpsat_alt", "cpsat_strong_w4", "cpsat_alt_w4")
NOISE_SWEEP = ("ls_nbk_n30", "ls_nbk_n40", "ls_nbk_n50", "ls_nbk_n65")       # plain ls_nbk is the 0.567 point of the sweep

# ---------------------------------------------------------------- grids
BNQ_P = (0.84, 0.88, 0.92)
BNQ = [("bnq", n, p) for n in (100, 200) for p in BNQ_P] + [("bnq", 200, 0.90)]           # .90 only at n=200: the 4th F1 primary cell
K5_LOW = (12.0, 14.0, 16.0, 18.0); K5_HI = (20.0, 20.5)
K5 = [("ksat5", 200, r) for r in K5_LOW + K5_HI]
LAT = e.LAT                                                                                  # the 16 v3e Latin cells
CELLS = {"main3f": BNQ + K5, "lat3f": LAT, "long3f": LAT,
         "smoke3f": [("bnq", 20, 0.5), ("qwh", 8, 0.4), ("qcp", 8, 0.30), ("ksat5", 40, 12.0), ("ksat5", 40, 20.0)]}

def cell_solvers(stage, fam, n, p):
    if stage == "long3f": return ("ls_py_n0", "ls_py")
    if stage == "lat3f": return ("ls_nb", "ls_nb_n10")
    if stage == "smoke3f":
        if fam == "bnq": return ("ls_nb", "ls_py", "cpsat_strong", "cpsat_alt", "cpsat_alt_w4") + (("ls_nb_fixed",) if FIX else ())
        if fam in ("qwh", "qcp"): return ("ls_nb", "ls_nb_n10", "ls_py", "ls_py_n0")
        return ("ls_nbk", "probsat_nb", "cadical195") + (NOISE_SWEEP if p >= 20 else ())
    if fam == "bnq":
        w4 = ("cpsat_strong_w4", "cpsat_alt_w4") if p != 0.90 else ()
        return ("ls_nb",) + (("ls_nb_fixed",) if FIX else ()) + ("ls_py", "cpsat_strong", "cpsat_alt") + w4
    return ("ls_nbk", "probsat_nb", "cadical195") + (NOISE_SWEEP if p >= 20 else ())

m.STAGES.update({"main3f": ({}, 20, 10.0, SOLVERS), "lat3f": ({}, 5, 10.0, SOLVERS), "long3f": ({}, 5, 100.0, SOLVERS), "smoke3f": ({}, 2, 2.0, SOLVERS)})

def tasks_for(stage, seeds, solvers):
    """1-worker runs first (seed outermost, balanced truncation), then the 4-worker CP-SAT runs alone (as nq_v3)."""
    out = []
    for grp in (False, True):
        for s in range(seeds):
            for f, n, p in CELLS[stage]: out += [(f, n, p, s, sv) for sv in cell_solvers(stage, f, n, p) if sv in solvers and sv.endswith("_w4") == grp]
    return out
m.tasks_for = tasks_for

# ---------------------------------------------------------------- F1: ls_nb_fixed = ls_nb with the kernel set chosen by diag_f1.py (rule in the design note)
SIG_INIT = "void(int64[:,::1], int64[::1], int64[::1], int64[::1], int64[::1])"
SIG_STEPS = "int64(int64[:,::1], int64[::1], int64[::1], int64[::1], int64[::1], float64, int64)"
def _disp(f):
    """the numba dispatcher behind nq_v3e's timing wrapper (or the dispatcher itself)"""
    if hasattr(f, "py_func"): return f
    return next(c.cell_contents for c in f.__closure__ if hasattr(c.cell_contents, "py_func"))

def build_kernels():
    if FIX in ("sig", "cache"):
        pi, ps = _disp(m.bnq_init).py_func, _disp(m.bnq_steps).py_func
        ki, ks = (njit(SIG_INIT)(pi), njit(SIG_STEPS)(ps)) if FIX == "sig" else (njit(cache=True)(pi), njit(cache=True)(ps))
        ns = {"bnq_init": ki, "bnq_steps": ks}; e._wrap(ns, "bnq_init"); e._wrap(ns, "bnq_steps")     # same first-call timing as ls_nb
        return ns["bnq_init"], ns["bnq_steps"]
    return m.bnq_init, m.bnq_steps                      # same_call: the very same dispatchers, only the parent warm-up differs
KERN = build_kernels() if FIX else None

def lsnb_bnq_fixed(B, T, seed, t0):
    """lsnb_bnq of nq_v3.py line for line, with the kernels of KERN"""
    init, steps = KERN; noise = m.NB_NOISE["bnq"]
    n = len(B); t_end = t0 + T; H = B.astype(np.int64); m.nb_seed(seed)
    pos = np.zeros(n, np.int64); col = np.zeros(n, np.int64); d1 = np.zeros(2 * n, np.int64); d2 = np.zeros(2 * n, np.int64)
    while time.monotonic() < t_end:
        init(H, pos, col, d1, d2); left = m.NB_RESTART["bnq"]
        while left > 0:
            if steps(H, pos, col, d1, d2, noise, 500): return "sat", pos.tolist()
            left -= 500
            if time.monotonic() > t_end: return "to", None
    return "to", None

def exact_call_warm(rg):
    """fix 'same_call': call the kernels with exactly the argument types of the real call (int64 C arrays, float noise, int steps) at n = 50, 100, 200"""
    for n in (50, 100, 200):
        H = m.gen_bnq(n, .9, rg).astype(np.int64); pos = np.zeros(n, np.int64); col = np.zeros(n, np.int64); d1 = np.zeros(2 * n, np.int64); d2 = np.zeros(2 * n, np.int64)
        m.nb_seed(0); m.bnq_init(H, pos, col, d1, d2); m.bnq_steps(H, pos, col, d1, d2, m.NB_NOISE["bnq"], 500)

# ---------------------------------------------------------------- F4: second CP-SAT model of blocked n-Queens
def cp_alt(B, T, seed, workers, t0):
    """Model B: one integer per row with the FULL domain 0..n-1; blocked cells are excluded by an element constraint blocked_r[q_r] = 0 (not by the domain);
    AllDifferent on q, q + r, q - r. Differs from cpsat_strong (domain holes, same AllDifferents) in how blocked cells enter the model."""
    cm = m.cp_model; n = len(B)
    if any(B[r].all() for r in range(n)): return "unsat", None
    mod = cm.CpModel(); zero, one = mod.NewConstant(0), mod.NewConstant(1)
    q = [mod.NewIntVar(0, n - 1, f"q{r}") for r in range(n)]
    for r in range(n): mod.AddElement(q[r], [one if B[r, c] else zero for c in range(n)], zero)
    mod.AddAllDifferent(q); mod.AddAllDifferent([q[r] + r for r in range(n)]); mod.AddAllDifferent([q[r] - r for r in range(n)])
    return m.run_cp(mod, T, seed, workers, t0, lambda sv: [sv.Value(x) for x in q])

# ---------------------------------------------------------------- dispatch
_prev = m.solve
def solve(fam, inst, solver, T, seed, workers):
    t0 = time.monotonic(); sd = seed * 7919 + 1
    if solver == "ls_py_n0":      # ls_py (lspy_latin) with noise 0.0; rng and everything else exactly as nq_v3.solve
        st, sol = m.lspy_latin(inst, T, random.Random(sd), noise=0.0)
    elif solver == "ls_nb_fixed":
        if KERN is None: raise RuntimeError("ls_nb_fixed requested but f1_fix.json has no fix")
        e.FIRST.clear(); st, sol = lsnb_bnq_fixed(inst, T, sd, t0)
        e.log_first(solver, sum(e.FIRST.values()), ";".join(f"{k}={v:.4f}" for k, v in e.FIRST.items()))
    elif solver.startswith("ls_nbk_n"):
        e.FIRST.clear(); st, sol = e.walksat_k(inst, T, sd, t0, noise=int(solver[8:]) / 100)
        e.log_first(solver, sum(e.FIRST.values()), ";".join(f"{k}={v:.4f}" for k, v in e.FIRST.items()))
    elif solver.startswith("cpsat_alt"): st, sol = cp_alt(inst, T, seed, workers, t0)
    else: return _prev(fam, inst, solver, T, seed, workers)
    return st, sol, time.monotonic() - t0
m.solve = solve

def warm_up():
    """compile the kernels of the families in this stage on real-size instances through the real solve() path, before any fork"""
    r = random.Random(1); stage = STAGE[0] or ""; big = not stage.startswith("smoke"); fams = {f for f, _, _ in CELLS.get(stage, [])}
    jobs = []
    if "bnq" in fams: jobs.append(("bnq", m.gen_bnq(200 if big else 20, .9 if big else .5, r), ("ls_nb",) + (("ls_nb_fixed",) if KERN else ())))
    if stage in ("lat3f", "smoke3f") and "qwh" in fams: jobs.append(("qwh", m.gen_qwh(40 if big else 8, .4, r), ("ls_nb", "ls_nb_n10")))
    if stage in ("lat3f", "smoke3f") and "qcp" in fams: jobs.append(("qcp", m.gen_qcp(40 if big else 8, .42 if big else .3, r), ("ls_nb", "ls_nb_n10")))
    if "ksat5" in fams: jobs.append(("ksat5", d.gen_ksat5(200 if big else 40, 20.5 if big else 12.0, r), ("ls_nbk", "probsat_nb", "ls_nbk_n30")))
    for fam, x, svs in jobs:
        for sv in svs:
            t = time.monotonic(); solve(fam, x, sv, 0.3, 0, 1)
            print(f"warm-up {fam} {sv}: {time.monotonic() - t:.2f} s, first calls {dict((k, round(v, 3)) for k, v in e.FIRST.items())}", flush=True)
    if FIX == "same_call" and "bnq" in fams: exact_call_warm(r)
m.warm_up = warm_up

_env0 = m.environment
def environment(a):
    x = _env0(a); x.update({"v3f_fix": FIXINFO, "v3f_offset": d.OFFSET.get(a.stage), "v3f_first_file": a.out + ".first.csv", "v3f_numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR")}); return x
m.environment = environment

STAGE = [None]
if __name__ == "__main__":
    for i, x in enumerate(sys.argv):
        if x == "--stage" and i + 1 < len(sys.argv): STAGE[0] = d.STAGE[0] = e.STAGE_NAME[0] = sys.argv[i + 1]
    if STAGE[0] == "main3f" and not os.path.exists(FIX_PATH): sys.exit("f1_fix.json missing: run diag_f1.py first (pre-registration order)")
    out = sys.argv[1]; e.LOG["fd"] = os.open(out + ".first.csv", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    if os.fstat(e.LOG["fd"]).st_size == 0: os.write(e.LOG["fd"], b"family,size,param,seed,solver,first_s,detail\n")
    e.LOG["on"] = True; m.main()
