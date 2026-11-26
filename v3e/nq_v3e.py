"""nq_v3e: fourth pre-registered stage on FRESH seeds (300-319). Wraps nq_v3.py and nq_v3d.py (both unchanged, copied beside this file).
Design: DESIGN_NOTE_V3E.md. Answers CRITIC2_v6 reasons 2, 3, 5:
 E1 start-up cost: warm_up compiles every kernel on a REAL-SIZE instance of every family through the real solve() path before forking,
    and the first-call time of every kernel in each child is logged to OUT.first.csv (residual per-run overhead).
 E2 native baseline: solver 'ls_c' = nq_ls_c.c (same min-conflicts algorithm as lsnb_bnq), run as a subprocess that reads the board from
    a temp file; the timed region includes writing the file, process start, parsing the answer.
 E3 ablation: 'ls_nb_n10' = ls_nb with noise 0.1 on Latin squares (ls_py's setting); stage long3e = ls_py with T = 100 s.
 E4 'ls_nbk' = WalkSAT/SKC for any clause length (needed for 5-SAT; ls_nb is hard-wired to k = 3), probsat_nb as in v3d.
 E5 CP-SAT tuning on bnq (tune_v3e.py) -> tuned_cpsat_bnq.json; cpsat_tuned is run on bnq n=200 only, and only if a variant won.
Usage: python nq_v3e.py OUT.csv --stage confirm3e|long3e|smoke3e [--jobs 4] [--hours H] [--seeds K] [--solvers a,b]"""
import sys, os, json, time, random, subprocess, tempfile, hashlib, shutil
import numpy as np
from numba import njit
import nq_v3d as d
import nq_v3 as m
from nq_v3 import rnd, nb_seed

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("NQ_LS_C_BIN", os.path.join(HERE, "nq_ls_c"))
TUNED_BNQ_PATH = os.environ.get("NQ_TUNED_BNQ", os.path.join(HERE, "tuned_cpsat_bnq.json"))
TUNED_BNQ = json.load(open(TUNED_BNQ_PATH)) if os.path.exists(TUNED_BNQ_PATH) else {}
HAS_SIGNAL = bool(TUNED_BNQ.get("bnq", {}).get("params"))
if HAS_SIGNAL: d.TUNED_CP["bnq"] = TUNED_BNQ["bnq"]      # d.solve reads TUNED_CP[fam]["params"]
d.OFFSET.update({"confirm3e": 300, "long3e": 300, "smoke3e": 5000})   # instance seed = index + OFFSET; tuning 2000-2002 is separate
SOLVERS = ("ls_nb", "ls_c", "ls_nb_n10", "ls_nbk", "probsat_nb", "ls_py", "cpsat_strong", "cpsat_tuned", "cadical195")

# ---------------------------------------------------------------- grids and per-cell solver lists
E1_N, E1_P = (50, 100, 200), (0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96)
LAT = [("qwh", N, h) for N in (30, 40) for h in (0.30, 0.35, 0.40, 0.45)] + [("qcp", N, f) for N in (30, 40) for f in (0.38, 0.42, 0.46, 0.50)]
E4 = [("ksat5", 200, r) for r in (20.0, 20.5, 21.0)] + [("ksat", n, r) for n in (1000, 2000) for r in (4.0, 4.1, 4.2)]
CELLS = {"confirm3e": [("bnq", n, p) for n in E1_N for p in E1_P] + LAT + E4, "long3e": LAT,
         "smoke3e": [("bnq", 20, 0.5), ("qwh", 8, 0.4), ("qcp", 8, 0.30), ("ksat", 60, 3.5), ("ksat5", 40, 12.0)]}
def cell_solvers(stage, fam, n):
    if stage == "long3e": return ("ls_py",)
    if fam == "bnq": return ("ls_nb", "ls_c", "cpsat_strong", "cadical195") + (("cpsat_tuned",) if n == 200 and HAS_SIGNAL else ())
    if fam in ("qwh", "qcp"): return ("ls_nb", "ls_nb_n10", "cpsat_strong", "cadical195") + (("ls_py",) if stage == "smoke3e" else ())
    if fam == "ksat": return ("ls_nb", "probsat_nb", "cadical195")
    return ("ls_nbk", "probsat_nb", "cadical195")
m.STAGES.update({"confirm3e": ({}, 20, 10.0, SOLVERS), "long3e": ({}, 5, 100.0, SOLVERS), "smoke3e": ({}, 2, 2.0, SOLVERS)})
m.GEN.setdefault("ksat5", d.gen_ksat5)

def tasks_for(stage, seeds, solvers):   # seed outermost (balanced truncation)
    return [(f, n, p, s, sv) for s in range(seeds) for f, n, p in CELLS[stage] for sv in cell_solvers(stage, f, n) if sv in solvers]
m.tasks_for = tasks_for

# ---------------------------------------------------------------- first-call timing (E1) and cell identity
FIRST, CELL, LOG = {}, [None] * 4, {"fd": None, "on": False}
def _wrap(ns, name):
    f = ns[name]
    def w(*a):
        if name in FIRST: return f(*a)
        t = time.perf_counter(); r = f(*a); FIRST[name] = time.perf_counter() - t; return r
    ns[name] = w
for _n in ("nb_seed", "bnq_init", "bnq_steps", "latin_init", "latin_steps", "sat_init", "sat_steps"): _wrap(m.__dict__, _n)
for _n in ("nb_seed", "ps_init", "ps_steps"): _wrap(d.__dict__, _n)

_rng0 = m.rng_for
def rng_for(fam, n, p, seed): CELL[:] = [fam, n, p, seed]; return _rng0(fam, n, p, seed)   # parent sets it before forking each child
m.rng_for = rng_for

def log_first(solver, first_s, detail):
    if LOG["fd"] is None or not LOG["on"] or CELL[0] is None: return
    os.write(LOG["fd"], (",".join(map(str, (CELL[0], CELL[1], CELL[2], CELL[3], solver, f"{first_s:.5f}", detail))) + "\n").encode())

# ---------------------------------------------------------------- E4: WalkSAT/SKC for any clause length (ragged clauses)
@njit
def wk_steps(lits, coff, a, nt, uns, where, occ_off, occ, nuns, noise, brk, flips):
    """WalkSAT/SKC as sat_steps in nq_v3.py but clause length L is read from coff: random unsatisfied clause; free move if some
    variable has break 0 (random among them); else with prob `noise` a random variable; else min break (random ties)."""
    for _ in range(flips):
        if nuns == 0: return 1, nuns
        j = uns[rnd(nuns)]; s = coff[j]; L = coff[j + 1] - s; mn = 1 << 30
        for k in range(L):
            v = abs(lits[s + k]); t = 2 * v + (0 if a[v] else 1); b = 0
            for q in range(occ_off[t], occ_off[t + 1]):
                if nt[occ[q]] == 1: b += 1
            brk[k] = b
            if b < mn: mn = b
        c = 0; ch = 0
        if mn == 0:
            for k in range(L):
                if brk[k] == 0:
                    c += 1
                    if rnd(c) == 0: ch = k
        elif np.random.random() < noise: ch = rnd(L)
        else:
            for k in range(L):
                if brk[k] == mn:
                    c += 1
                    if rnd(c) == 0: ch = k
        v = abs(lits[s + ch]); old = 2 * v + (0 if a[v] else 1); new = old ^ 1; a[v] = not a[v]
        for q in range(occ_off[old], occ_off[old + 1]):
            jj = occ[q]; nt[jj] -= 1
            if nt[jj] == 0: uns[nuns] = jj; where[jj] = nuns; nuns += 1
        for q in range(occ_off[new], occ_off[new + 1]):
            jj = occ[q]; nt[jj] += 1
            if nt[jj] == 1:
                k2 = where[jj]; nuns -= 1; last = uns[nuns]
                uns[k2] = last; where[last] = k2; where[jj] = -1
    return 0, nuns
_wrap(globals(), "wk_steps")

def walksat_k(C, T, seed, t0=None, noise=None):
    noise = m.NB_NOISE["ksat"] if noise is None else noise     # same constant as ls_nb on 3-SAT (0.567); not tuned for k = 5 (stated limit)
    t_end = (t0 or time.monotonic()) + T; nb_seed(seed)
    lens = np.fromiter((len(c) for c in C), np.int64, len(C)); coff = np.zeros(len(C) + 1, np.int64); coff[1:] = np.cumsum(lens)
    lits = np.fromiter((l for c in C for l in c), np.int64, int(coff[-1])); n = int(np.abs(lits).max())
    li = 2 * np.abs(lits) + (lits < 0); cid = np.repeat(np.arange(len(C), dtype=np.int64), lens)
    off = np.zeros(2 * n + 3, np.int64); np.add.at(off, li + 1, 1); off = np.cumsum(off); occ = cid[np.argsort(li, kind="stable")]
    a = np.zeros(n + 1, np.bool_); a[1:] = np.random.default_rng(seed).random(n) < .5
    nt = np.zeros(len(C), np.int64); uns = np.zeros(len(C), np.int64); where = np.zeros(len(C), np.int64)
    nuns = d.ps_init(lits, coff, a, nt, uns, where); brk = np.zeros(int(lens.max()), np.int64)
    while True:
        ok, nuns = wk_steps(lits, coff, a, nt, uns, where, off, occ, nuns, noise, brk, 20000)
        if ok: return "sat", a
        if time.monotonic() > t_end: return "to", None

# ---------------------------------------------------------------- E2: native C local search as a subprocess
def run_ls_c(B, T, sd, t0):
    """Timed region (t0 set by the caller) = write board file + process start + run + parse. Internal budget = remaining time - 0.02 s."""
    n = len(B); rows = (B.astype(np.uint8) + 48); fd, path = tempfile.mkstemp(prefix="bnq_", suffix=".txt")
    try:
        with os.fdopen(fd, "wb") as f: f.write(f"{n}\n".encode() + b"\n".join(r.tobytes() for r in rows) + b"\n")
        rem = max(0.01, T - (time.monotonic() - t0) - 0.02); tw = time.monotonic()
        try: pr = subprocess.run([BIN, path, str(sd), f"{rem:.4f}"], capture_output=True, text=True, timeout=rem + 0.5)
        except subprocess.TimeoutExpired: return "to", None, None
        wall = time.monotonic() - tw
    finally:
        try: os.unlink(path)
        except OSError: pass
    if pr.returncode != 0: raise RuntimeError("nq_ls_c rc=%d %s" % (pr.returncode, pr.stderr[:200]))
    tok = pr.stdout.split()
    if tok[0] == "TO": return "to", None, (wall, float(tok[1]))
    return "sat", [int(x) for x in tok[2:2 + n]], (wall, float(tok[1]))

# ---------------------------------------------------------------- dispatch
_prev = m.solve
def solve(fam, inst, solver, T, seed, workers):
    FIRST.clear(); t0 = time.monotonic(); sd = seed * 7919 + 1
    if solver == "ls_c":
        st, sol, wi = run_ls_c(inst, T, sd, t0)
        if wi: log_first(solver, wi[0] - wi[1], f"wall={wi[0]:.4f};internal={wi[1]:.4f}")
        return st, sol, time.monotonic() - t0
    if solver == "ls_nb_n10": st, sol = m.lsnb_latin(inst, T, sd, noise=0.1, t0=t0, fam=fam)
    elif solver == "ls_nbk": st, sol = walksat_k(inst, T, sd, t0)
    else:
        r = _prev(fam, inst, solver, T, seed, workers)
        if solver in ("ls_nb", "probsat_nb") and FIRST: log_first(solver, sum(FIRST.values()), ";".join(f"{k}={v:.4f}" for k, v in FIRST.items()))
        return r
    log_first(solver, sum(FIRST.values()), ";".join(f"{k}={v:.4f}" for k, v in FIRST.items()))
    return st, sol, time.monotonic() - t0
m.solve = solve

# ---------------------------------------------------------------- E1: warm-up on real-size instances through the real solve() path
def ensure_bin():
    if os.path.exists(BIN): return True
    src = os.path.join(HERE, "nq_ls_c.c")
    if shutil.which("gcc") and os.path.exists(src): return subprocess.run(["gcc", "-O3", "-march=native", "-o", BIN, src]).returncode == 0
    return False

def warm_up():
    r = random.Random(1); stage = STAGE_NAME[0] or ""
    big = not stage.startswith("smoke")
    inst = [("bnq", m.gen_bnq(200 if big else 20, .9 if big else .5, r), ("ls_nb", "ls_c")),
            ("qwh", m.gen_qwh(40 if big else 8, .4, r), ("ls_nb", "ls_nb_n10")), ("qcp", m.gen_qcp(40 if big else 8, .42 if big else .3, r), ("ls_nb", "ls_nb_n10")),
            ("ksat", m.gen_ksat(2000 if big else 60, 4.2 if big else 3.5, r), ("ls_nb", "probsat_nb")),
            ("ksat5", d.gen_ksat5(200 if big else 40, 20.5 if big else 12.0, r), ("ls_nbk", "probsat_nb"))]
    for fam, x, svs in inst:
        for sv in svs:
            if sv == "ls_c" and not ensure_bin(): print("WARM-UP: ls_c skipped, binary missing", flush=True); continue
            t = time.monotonic(); solve(fam, x, sv, 0.3, 0, 1); print(f"warm-up {fam} {sv}: {time.monotonic() - t:.2f} s, first calls {dict((k, round(v, 3)) for k, v in FIRST.items())}", flush=True)
m.warm_up = warm_up

_env0 = m.environment
def environment(a):
    e = _env0(a); gcc = shutil.which("gcc")
    try: gv = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0] if gcc else None
    except Exception: gv = None
    e.update({"v3e_tuned_bnq": TUNED_BNQ, "v3e_cpsat_tuned_active": HAS_SIGNAL, "v3e_gcc": gv, "v3e_ls_c_sha256": hashlib.sha256(open(BIN, "rb").read()).hexdigest() if os.path.exists(BIN) else None,
              "v3e_offset": d.OFFSET.get(a.stage), "v3e_first_file": a.out + ".first.csv"})
    return e
m.environment = environment

STAGE_NAME = [None]
if __name__ == "__main__":
    for i, x in enumerate(sys.argv):
        if x == "--stage" and i + 1 < len(sys.argv): STAGE_NAME[0] = d.STAGE[0] = sys.argv[i + 1]
    if STAGE_NAME[0] == "confirm3e" and not os.path.exists(TUNED_BNQ_PATH):
        sys.exit("tuned_cpsat_bnq.json missing: run tune_v3e.py first (pre-registration order)")
    if STAGE_NAME[0] in ("confirm3e", "smoke3e") and "--solvers" not in sys.argv and not ensure_bin():
        sys.exit("ls_c binary missing and gcc unavailable/failed: install gcc or pass --solvers without ls_c (then E2 is not evaluable)")
    out = sys.argv[1]; LOG["fd"] = os.open(out + ".first.csv", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    if os.fstat(LOG["fd"]).st_size == 0: os.write(LOG["fd"], b"family,size,param,seed,solver,first_s,detail\n")
    LOG["on"] = True; m.main()
