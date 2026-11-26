"""nq_v3d: third pre-registered stage on FRESH seeds, wraps nq_v3.py (unchanged, copied beside this file).
Design: DESIGN_NOTE_V3D.md. Adds: ksat5 family; solvers probsat_nb (probSAT, Balint and Schoening 2012, numba), ls_tabu_nb (tabu
row-permutation LS for qwh/qcp), cpsat_tuned (CP-SAT with the parameter text chosen by tune_cpsat.py, tuned_cpsat.json);
structured outside benchmarks (SATLIB, loaded from ext_data/ after `python nq_v3d.py --fetch`); per-family solver lists.
Usage: python nq_v3d.py OUT.csv --stage confirm3d|ext3d|smoke3 [--jobs 4] [--hours H] [--seeds K] [--solvers a,b]
       python nq_v3d.py --fetch          (download + unpack SATLIB sets, write ext_manifest.json; needs internet)"""
import sys, os, json, time, glob, hashlib, random
import numpy as np
from numba import njit
import nq_v3 as m
from nq_v3 import rnd, nb_seed, latin_init

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED_CP_PATH = os.environ.get("NQ_TUNED_CP", os.path.join(HERE, "tuned_cpsat.json"))
TUNED_LS_PATH = os.environ.get("NQ_TUNED_LS", os.path.join(HERE, "tuned_ls.json"))
EXT_DIR = os.environ.get("NQ_EXT_DIR", os.path.join(HERE, "ext_data"))
SATLIB = "https://www.cs.ubc.ca/~hoos/SATLIB/"   # hrefs verified by reading the page (2026-11-26); archive contents NOT verified
EXT_SETS = {"ext_uf250": ("Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz", 20),
            "ext_flat150": ("Benchmarks/SAT/GCP/flat150-360.tar.gz", 20),
            "ext_qg": ("Benchmarks/SAT/QG/QG.tar.gz", 22)}
PROBSAT = {"poly": (2.06, 0.9), "exp5": 3.7}   # published constants (3-SAT poly; 5-SAT exp), taken from the brief/memory, paper not re-opened
OFFSET = {"confirm3d": 200, "smoke3": 5000, "ext3d": 0}   # instance seed = index + OFFSET (confirmatory 200-219; tuning 2000-2002; smoke 5000+)

# ---------------------------------------------------------------- families, grids, per-family solvers
def gen_ksat5(n, ratio, rng):
    return [[v * rng.choice((-1, 1)) for v in rng.sample(range(1, n + 1), 5)] for _ in range(round(ratio * n))]

def parse_dimacs(path):
    cl, cur = [], []
    for line in open(path):
        if not line.strip() or line[0] in "cp": continue
        if line.startswith("%"): break
        for tok in line.split():
            x = int(tok)
            if x:
                cur.append(x); continue
            s = set(cur); cur = []
            if s and not any(-l in s for l in s): cl.append(sorted(s, key=abs))   # tautologies dropped, duplicate literals merged
    return cl

def ext_manifest():
    p = os.path.join(EXT_DIR, "ext_manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def load_ext(fam, stem):
    return parse_dimacs(os.path.join(EXT_DIR, fam, stem + ".cnf"))

def fetch():
    import urllib.request, tarfile
    os.makedirs(EXT_DIR, exist_ok=True); man = {}
    for fam, (href, k) in EXT_SETS.items():
        d = os.path.join(EXT_DIR, fam); os.makedirs(d, exist_ok=True); arc = os.path.join(EXT_DIR, fam + ".tar.gz")
        urllib.request.urlretrieve(SATLIB + href, arc)
        sha = hashlib.sha256(open(arc, "rb").read()).hexdigest()
        with tarfile.open(arc) as tf:   # untrusted: only regular .cnf members, flattened to basenames
            for mem in tf.getmembers():
                if mem.isfile() and mem.name.endswith(".cnf"):
                    open(os.path.join(d, os.path.basename(mem.name)), "wb").write(tf.extractfile(mem).read())
        stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.cnf")))[:k]
        man[fam] = {"url": SATLIB + href, "sha256": sha, "files": stems}; print(fam, len(stems), "files", sha[:12], flush=True)
    json.dump(man, open(os.path.join(EXT_DIR, "ext_manifest.json"), "w"), indent=1)

GRID_D = {"bnq": ((100, 200), (0.84, 0.86, 0.88, 0.90, 0.92, 0.94)), "qwh": ((30, 40), (0.35, 0.40, 0.45, 0.55)),
          "qcp": ((30, 40), (0.38, 0.42, 0.46)), "ksat": ((250, 350), (4.0, 4.1, 4.2, 4.26, 4.3, 4.4)),
          "ksat5": ((100, 150, 200), (19.0, 20.0, 20.5, 21.0, 21.1, 21.5, 22.0, 24.0))}
SMOKE_D = {"bnq": ((20,), (0.5,)), "qwh": ((8,), (0.4,)), "qcp": ((8,), (0.30,)), "ksat": ((60,), (3.5,)), "ksat5": ((40,), (12.0,))}
FAM_SOLVERS = {"bnq": ("ls_nb", "cpsat_strong", "cpsat_tuned", "cadical195"),
               "qwh": ("ls_nb", "ls_tabu_nb", "cpsat_strong", "cpsat_tuned", "cadical195"),
               "qcp": ("ls_nb", "ls_tabu_nb", "cpsat_strong", "cpsat_tuned", "cadical195"),
               "ksat": ("probsat_nb", "ls_nb", "cpsat_strong", "cpsat_tuned", "cadical195"),
               "ksat5": ("probsat_nb", "cpsat_strong", "cpsat_tuned", "cadical195")}
for f in EXT_SETS: FAM_SOLVERS[f] = ("probsat_nb", "cpsat_strong", "cpsat_tuned", "cadical195")
ALL_SOLVERS = tuple(dict.fromkeys(s for v in FAM_SOLVERS.values() for s in v))
EXT_GRID = {f: ((0,), tuple(v["files"])) for f, v in ext_manifest().items()}
m.STAGES.update({"confirm3d": (GRID_D, 20, 10.0, ALL_SOLVERS), "smoke3": (SMOKE_D, 2, 2.0, ALL_SOLVERS), "ext3d": (EXT_GRID, 1, 10.0, ALL_SOLVERS)})
m.GEN["ksat5"] = gen_ksat5; m.VERIFY["ksat5"] = m.verify_ksat; m.CNF["ksat5"] = m.cnf_ksat; m.CP["ksat5"] = m.cp_ksat
for f in EXT_SETS:
    m.GEN[f] = (lambda ff: lambda n, p, rng: load_ext(ff, p))(f); m.VERIFY[f] = m.verify_ksat; m.CNF[f] = m.cnf_ksat; m.CP[f] = m.cp_ksat

def tasks_for(stage, seeds, solvers):   # seed outermost (balanced truncation); per-family solver lists
    out = []
    for s in range(seeds):
        for fam, (sizes, params) in m.STAGES[stage][0].items():
            for n in sizes:
                for p in params: out += [(fam, n, p, s, sv) for sv in FAM_SOLVERS[fam] if sv in solvers]
    return out
m.tasks_for = tasks_for

_orig_rng = m.rng_for
STAGE = [None]
def rng_for(fam, n, p, seed): return _orig_rng(fam, n, p, seed + OFFSET.get(STAGE[0], 0))
m.rng_for = rng_for
def instance(fam, n, p, abs_seed):
    """generator with the ABSOLUTE seed (used by tune_cpsat.py with seeds 2000-2002)."""
    return m.GEN[fam](n, p, _orig_rng(fam, n, p, abs_seed))

# ---------------------------------------------------------------- tuned parameters
def _load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default
TUNED_CP = _load(TUNED_CP_PATH, {}); TUNED_LS = _load(TUNED_LS_PATH, {"tabu_tenure": 10, "tabu_stall": 50})
CP_VARIANTS = {"default": "", "lin0": "linearization_level: 0", "lin2": "linearization_level: 2",
               "portfolio": "search_branching: PORTFOLIO_SEARCH", "fixed": "search_branching: FIXED_SEARCH",
               "sym0": "symmetry_level: 0", "sym2": "symmetry_level: 2", "nopresolve": "cp_model_presolve: false"}
ACTIVE = {}

def run_cp(mm, T, seed, workers, t0, read):   # nq_v3.run_cp plus optional parameter text (set only by cpsat_tuned / tuning)
    from ortools.sat.python import cp_model
    from google.protobuf import text_format
    sv = cp_model.CpSolver(); P = sv.parameters; P.num_workers = workers; P.random_seed = seed
    if ACTIVE.get("txt"): text_format.Merge(ACTIVE["txt"], P)
    P.max_time_in_seconds = max(0.01, T - (time.monotonic() - t0)); st = sv.Solve(mm)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE): return "sat", read(sv)
    return ("unsat" if st == cp_model.INFEASIBLE else "to"), None
m.run_cp = run_cp

def cp_with(fam, inst, T, seed, txt, t0):
    ACTIVE["txt"] = txt
    return m.CP[fam](inst, T, seed, 1, True, t0)

# ---------------------------------------------------------------- probSAT (ragged clauses, numba)
@njit
def ps_init(lits, coff, a, nt, uns, where):
    st = 0
    for j in range(len(coff) - 1):
        c = 0
        for q in range(coff[j], coff[j + 1]):
            l = lits[q]
            if (l > 0) == a[abs(l)]: c += 1
        nt[j] = c; where[j] = -1
        if c == 0: uns[st] = j; where[j] = st; st += 1
    return st

@njit
def ps_steps(lits, coff, a, nt, uns, where, occ_off, occ, nuns, tab, probs, flips):
    """probSAT: random unsatisfied clause; flip var v with probability f(break(v)) / sum, f from table (break-only, no make)."""
    nt_ = len(tab)
    for _ in range(flips):
        if nuns == 0: return 1, nuns
        j = uns[rnd(nuns)]; s = coff[j]; L = coff[j + 1] - s; tot = 0.0
        for k in range(L):
            v = abs(lits[s + k]); t = 2 * v + (0 if a[v] else 1); b = 0
            for q in range(occ_off[t], occ_off[t + 1]):
                if nt[occ[q]] == 1: b += 1
            if b >= nt_: b = nt_ - 1
            probs[k] = tab[b]; tot += tab[b]
        x = np.random.random() * tot; ch = L - 1; acc = 0.0
        for k in range(L):
            acc += probs[k]
            if x < acc: ch = k; break
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

def prob_table(maxlen, size=256):
    b = np.arange(size, dtype=np.float64)
    if maxlen <= 3: cb, eps = PROBSAT["poly"]; return (eps + b) ** (-cb)
    return PROBSAT["exp5"] ** (-b)   # k >= 4 (ksat5 and long-clause structured sets): exponential, cb 3.7

def probsat_nb(C, T, seed, t0=None):
    t_end = (t0 or time.monotonic()) + T; nb_seed(seed)
    lens = np.fromiter((len(c) for c in C), np.int64, len(C)); coff = np.zeros(len(C) + 1, np.int64); coff[1:] = np.cumsum(lens)
    lits = np.fromiter((l for c in C for l in c), np.int64, int(coff[-1])); n = int(np.abs(lits).max())
    li = 2 * np.abs(lits) + (lits < 0); cid = np.repeat(np.arange(len(C), dtype=np.int64), lens)
    off = np.zeros(2 * n + 3, np.int64); np.add.at(off, li + 1, 1); off = np.cumsum(off); occ = cid[np.argsort(li, kind="stable")]
    a = np.zeros(n + 1, np.bool_); a[1:] = np.random.default_rng(seed).random(n) < .5
    nt = np.zeros(len(C), np.int64); uns = np.zeros(len(C), np.int64); where = np.zeros(len(C), np.int64)
    nuns = ps_init(lits, coff, a, nt, uns, where); mx = int(lens.max()); tab = prob_table(mx); probs = np.zeros(mx, np.float64)
    while True:
        ok, nuns = ps_steps(lits, coff, a, nt, uns, where, off, occ, nuns, tab, probs, 10000)
        if ok: return "sat", a
        if time.monotonic() > t_end: return "to", None

# ---------------------------------------------------------------- tabu row-permutation LS for Latin squares
@njit
def tabu_steps(hole, X, cnt, cost, tabu, st, tenure, stall_lim, steps):
    """Same representation and cost as ls_nb (rows are permutations, cost = column duplicates). Each step takes the best swap
    (conflicting hole cell, other hole of its row) over ALL conflicting cells; ties random; a swap is tabu if it puts a value
    back into a cell it left less than tenure+rnd steps ago, unless it beats the best cost of this restart (aspiration).
    st = [iteration, best cost, steps since improvement]. Returns 1 solved, 2 restart wanted (stall or no admissible move), 0 else."""
    N = len(X)
    for _ in range(steps):
        if cost[0] == 0: return 1
        st[0] += 1; it = st[0]; bestd = 1 << 30; nt = 0; br = 0; b1 = 0; b2 = 0
        for r in range(N):
            for c1 in range(N):
                if not hole[r, c1]: continue
                a = X[r, c1]
                if cnt[c1, a] <= 1: continue
                for c2 in range(N):
                    if c2 == c1 or not hole[r, c2]: continue
                    b = X[r, c2]
                    d = (1 if cnt[c1, b] >= 1 else 0) + (1 if cnt[c2, a] >= 1 else 0) - (1 if cnt[c2, b] > 1 else 0) - (1 if cnt[c1, a] > 1 else 0)
                    if (tabu[r, c1, b] > it or tabu[r, c2, a] > it) and cost[0] + d >= st[1]: continue
                    if d < bestd: bestd = d; nt = 1; br = r; b1 = c1; b2 = c2
                    elif d == bestd:
                        nt += 1
                        if rnd(nt) == 0: br = r; b1 = c1; b2 = c2
        if nt == 0: return 2
        a = X[br, b1]; b = X[br, b2]
        cnt[b1, a] -= 1; cnt[b1, b] += 1; cnt[b2, b] -= 1; cnt[b2, a] += 1; X[br, b1] = b; X[br, b2] = a; cost[0] += bestd
        ten = it + tenure + rnd(tenure // 2 + 1); tabu[br, b1, a] = ten; tabu[br, b2, b] = ten
        if cost[0] < st[1]: st[1] = cost[0]; st[2] = 0
        else:
            st[2] += 1
            if st[2] > stall_lim: return 2
    return 0

def lstabu_latin(G, T, seed, t0=None, tenure=None, stall=None):
    tenure = TUNED_LS["tabu_tenure"] if tenure is None else tenure; stall = TUNED_LS["tabu_stall"] if stall is None else stall
    N = len(G); t_end = (t0 or time.monotonic()) + T; nb_seed(seed); G = G.astype(np.int64); hole = G < 0
    X = np.zeros((N, N), np.int64); cnt = np.zeros((N, N), np.int64); cost = np.zeros(1, np.int64)
    tabu = np.zeros((N, N, N), np.int64); st = np.zeros(3, np.int64)
    while time.monotonic() < t_end:
        latin_init(G, hole, X, cnt, cost); tabu[:] = 0; st[0] = 0; st[1] = cost[0]; st[2] = 0
        while True:
            r = tabu_steps(hole, X, cnt, cost, tabu, st, tenure, stall * N, 200)
            if r == 1: return "sat", X.copy()
            if time.monotonic() > t_end: return "to", None
            if r == 2: break
    return "to", None

# ---------------------------------------------------------------- dispatch
_orig_solve = m.solve
def solve(fam, inst, solver, T, seed, workers):
    t0 = time.monotonic(); sd = seed * 7919 + 1
    if solver == "probsat_nb": st, sol = probsat_nb(inst, T, sd, t0)
    elif solver == "ls_tabu_nb": st, sol = lstabu_latin(inst, T, sd, t0)
    elif solver == "cpsat_tuned":
        key = "ksat" if fam.startswith("ext_") else fam
        st, sol = cp_with(fam, inst, T, seed, TUNED_CP.get(key, {}).get("params", ""), t0)
    else: return _orig_solve(fam, inst, solver, T, seed, workers)
    return st, sol, time.monotonic() - t0
m.solve = solve

_orig_warm = m.warm_up
def warm_up():
    _orig_warm(); r = random.Random(1)
    probsat_nb(gen_ksat5(20, 10.0, r), 0.3, 0); probsat_nb(m.gen_ksat(20, 3.0, r), 0.3, 0); lstabu_latin(m.gen_qwh(5, .4, r), 0.3, 0)
m.warm_up = warm_up

_orig_env = m.environment
def environment(a):
    e = _orig_env(a)
    e.update({"v3d_tuned_cpsat": TUNED_CP, "v3d_tuned_ls": TUNED_LS, "v3d_probsat": PROBSAT, "v3d_offset": OFFSET.get(a.stage), "v3d_ext": ext_manifest()})
    return e
m.environment = environment

if __name__ == "__main__":
    if "--fetch" in sys.argv: fetch(); sys.exit(0)
    for i, x in enumerate(sys.argv):
        if x == "--stage" and i + 1 < len(sys.argv): STAGE[0] = sys.argv[i + 1]
    if STAGE[0] in ("confirm3d", "ext3d") and not (TUNED_CP and os.path.exists(TUNED_LS_PATH)):
        sys.exit("tuned_cpsat.json / tuned_ls.json missing: run tune_cpsat.py first (pre-registration order)")
    if STAGE[0] == "ext3d" and not EXT_GRID: sys.exit("ext_manifest.json missing: run --fetch first")
    m.main()
