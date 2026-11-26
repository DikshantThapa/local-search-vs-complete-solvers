"""nq_v2: phase-transition comparison of local search, CP-SAT (two models) and CDCL (glucose4, cadical195)
on four random families with the same protocol:
  bnq  Blocked n-Queens        size n, param p = probability a cell is blocked
  qwh  quasigroup with holes   size N, param h = fraction of cells punched out of a random Latin square (always SAT)
  qcp  quasigroup completion   size N, param f = fraction of cells preassigned with forward checking (SAT or UNSAT)
  ksat random 3-SAT            size n vars, param r = clauses / variables
Each (instance, solver) run is a forked child process with a hard kill at T + GRACE, so every solver gets the same
wall-clock budget, including model building / encoding. Every reported solution is checked by an independent verifier.
One CSV row per run, appended and flushed immediately; rerunning with the same OUT resumes (finished keys are skipped).
Grid, seeds, budget and solver settings are fixed in nqueens/PREREG_NOTE.md; do not change them after the main run starts.
Usage: python nq_v2.py OUT.csv [--smoke] [--jobs 4] [--T 10] [--hours 11] [--stage main|w4]
"""
import argparse, csv, os, random, sys, time
import multiprocessing as mp
import numpy as np
from ortools.sat.python import cp_model
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

GRID = {
    "bnq": ((50, 100, 200), (0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96)),
    "qwh": ((20, 30, 40), (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70)),
    "qcp": ((20, 30, 40), (0.30, 0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.50)),
    "ksat": ((150, 250, 350), (3.8, 4.0, 4.1, 4.2, 4.26, 4.3, 4.4, 4.6)),
}
SEEDS = 20          # instances per cell (main stage)
SEEDS_W4 = 5        # instances per cell for the 4-worker CP-SAT check
SMOKE = {"bnq": ((12, 30), (0.5, 0.9)), "qwh": ((8,), (0.4,)), "qcp": ((8,), (0.42,)), "ksat": ((40,), (4.26,))}
SOLVERS = ("ls", "cpsat_naive", "cpsat_strong", "glucose4", "cadical195")
GRACE = 3.0
FIELDS = ["family", "size", "param", "seed", "solver", "workers", "T", "status", "t", "verified", "killed"]

# ---------------------------------------------------------------- instances
def rng_for(fam, size, param, seed):
    return random.Random(f"{fam}-{size}-{param}-{seed}")  # str seeds are hashed with sha512: stable across runs

def gen_bnq(n, p, rng):
    return np.array([[rng.random() < p for _ in range(n)] for _ in range(n)])  # True = blocked

def jm_latin(N, rng, iters=None):
    """Jacobson-Matthews Markov chain on the incidence cube, started from the cyclic square.
    The cube is stored sparsely as three line maps (each line has at most three non-zero entries).
    ponytail: N**3 moves, proper and improper both counted; mixing time is not proven, raise iters if uniformity matters."""
    iters = iters or N ** 3
    CS = [[{} for _ in range(N)] for _ in range(N)]; RS = [[{} for _ in range(N)] for _ in range(N)]; RC = [[{} for _ in range(N)] for _ in range(N)]
    def add(r, c, s, k):
        for D, key in ((CS[c][s], r), (RS[r][s], c), (RC[r][c], s)):
            v = D.get(key, 0) + k
            if v: D[key] = v
            else: del D[key]
    ones = lambda D: [k for k, v in D.items() if v == 1]
    for r in range(N):
        for c in range(N): add(r, c, (r + c) % N, 1)
    imp, i = None, 0
    while i < iters or imp is not None:
        if imp is None:
            while True:
                r, c, s = rng.randrange(N), rng.randrange(N), rng.randrange(N)
                if s not in RC[r][c]: break
            r2, c2, s2 = ones(CS[c][s])[0], ones(RS[r][s])[0], ones(RC[r][c])[0]
        else:
            r, c, s = imp
            r2, c2, s2 = rng.choice(ones(CS[c][s])), rng.choice(ones(RS[r][s])), rng.choice(ones(RC[r][c]))
        i += 1
        for a, b, d, k in ((r, c, s, 1), (r, c2, s2, 1), (r2, c, s2, 1), (r2, c2, s, 1), (r, c, s2, -1), (r, c2, s, -1), (r2, c, s, -1), (r2, c2, s2, -1)):
            add(a, b, d, k)
        imp = (r2, c2, s2) if RC[r2][c2].get(s2, 0) < 0 else None
    assert all(len(RC[r][c]) == 1 for r in range(N) for c in range(N))
    L = np.array([[ones(RC[r][c])[0] for c in range(N)] for r in range(N)])
    assert verify_latin(-np.ones((N, N), int), L)
    return L

def gen_qwh(N, h, rng):
    G = jm_latin(N, rng).copy(); cells = [(r, c) for r in range(N) for c in range(N)]
    for r, c in rng.sample(cells, round(h * N * N)): G[r, c] = -1
    return G

def gen_qcp(N, f, rng):
    """Random cells get random values; a value is rejected if forward checking empties a domain in its row or column."""
    G = -np.ones((N, N), int); rowhas = np.zeros((N, N), bool); colhas = np.zeros((N, N), bool)
    cells = [(r, c) for r in range(N) for c in range(N)]; rng.shuffle(cells); k = round(f * N * N); placed = 0
    for r, c in cells:
        if placed == k: break
        vals = [v for v in range(N) if not rowhas[r, v] and not colhas[c, v]]; rng.shuffle(vals)
        for v in vals:
            er = [c2 for c2 in range(N) if G[r, c2] < 0 and c2 != c]; ec = [r2 for r2 in range(N) if G[r2, c] < 0 and r2 != r]
            A = ~(rowhas[r][None, :] | colhas[er]) if er else np.zeros((0, N), bool)
            B = ~(colhas[c][None, :] | rowhas[ec]) if ec else np.zeros((0, N), bool)
            if ((A.sum(1) == 1) & A[:, v]).any() or ((B.sum(1) == 1) & B[:, v]).any(): continue
            G[r, c] = v; rowhas[r, v] = colhas[c, v] = True; placed += 1; break
    return G  # ponytail: if no value passes the check the cell stays empty, so a few instances may have < k preassigned

def gen_ksat(n, ratio, rng):
    return [[v * rng.choice((-1, 1)) for v in rng.sample(range(1, n + 1), 3)] for _ in range(round(ratio * n))]

GEN = {"bnq": gen_bnq, "qwh": gen_qwh, "qcp": gen_qcp, "ksat": gen_ksat}

# ---------------------------------------------------------------- verifiers
def verify_bnq(B, q):
    n = len(B); q = [int(x) for x in q]
    return len(q) == n and all(0 <= q[r] < n and not B[r, q[r]] for r in range(n)) and len(set(q)) == n \
        and len({q[r] + r for r in range(n)}) == n and len({q[r] - r for r in range(n)}) == n

def verify_latin(G, X):
    N = len(G); X = np.asarray(X); full = set(range(N))
    return X.shape == (N, N) and all((G[G >= 0] == X[G >= 0])) and all(set(X[r]) == full for r in range(N)) and all(set(X[:, c]) == full for c in range(N))

def verify_ksat(C, a):
    return all(any((l > 0) == bool(a[abs(l)]) for l in cl) for cl in C)

VERIFY = {"bnq": verify_bnq, "qwh": verify_latin, "qcp": verify_latin, "ksat": verify_ksat}

def latin_domains(G):
    N = len(G); rowhas = [set(G[r][G[r] >= 0]) for r in range(N)]; colhas = [set(G[:, c][G[:, c] >= 0]) for c in range(N)]
    return {(r, c): [v for v in range(N) if v not in rowhas[r] and v not in colhas[c]] for r in range(N) for c in range(N) if G[r, c] < 0}, rowhas, colhas

# ---------------------------------------------------------------- local search
def ls_bnq(B, T, rng):
    """Min-conflicts (Minton et al. 1992) with greedy start, blocked-cell penalty 5, restart every 20000 moves (as in v1)."""
    n = len(B); t_end = time.monotonic() + T; H = B.astype(int); ar = np.arange(n); nr = np.random.default_rng(rng.randrange(2 ** 31))
    while time.monotonic() < t_end:
        pos = np.zeros(n, int); col = np.zeros(n, int); d1 = np.zeros(2 * n, int); d2 = np.zeros(2 * n, int)
        def put(r, c, s=1): pos[r] = c; col[c] += s; d1[r - c + n] += s; d2[r + c] += s
        for r in range(n):
            v = col + d1[r - ar + n] + d2[r + ar] + 5 * H[r]; put(r, int(nr.choice(np.flatnonzero(v == v.min()))))
        for step in range(20000):
            cf = col[pos] + d1[ar - pos + n] + d2[ar + pos] - 3 + H[ar, pos]; idx = np.flatnonzero(cf > 0)
            if len(idx) == 0: return "sat", pos.tolist()
            if step % 64 == 0 and time.monotonic() > t_end: return "to", None
            r = int(nr.choice(idx)); put(r, pos[r], -1)
            v = col + d1[r - ar + n] + d2[r + ar] + 5 * H[r]; put(r, int(nr.choice(np.flatnonzero(v == v.min()))))
    return "to", None

def ls_latin(G, T, rng, noise=0.1):
    """Rows are kept as permutations; a move swaps two empty cells of a row; cost = repeated values in columns.
    Min-conflicts on a random conflicted cell, random-walk swap with probability `noise`, restart every 50*N*N moves."""
    N = len(G); t_end = time.monotonic() + T; nr = np.random.default_rng(rng.randrange(2 ** 31)); hole = G < 0
    holes = [np.flatnonzero(hole[r]) for r in range(N)]; miss = [np.array(sorted(set(range(N)) - set(G[r][G[r] >= 0]))) for r in range(N)]
    cols = np.arange(N)[None, :]
    while time.monotonic() < t_end:
        X = G.copy()
        for r in range(N): X[r, holes[r]] = nr.permutation(miss[r])
        cnt = np.zeros((N, N), int)
        for c in range(N): np.add.at(cnt[c], X[:, c], 1)
        for step in range(50 * N * N):
            bad = np.flatnonzero((hole & (cnt[cols, X] > 1)).ravel())
            if len(bad) == 0: return "sat", X
            if step % 64 == 0 and time.monotonic() > t_end: return "to", None
            r, c1 = divmod(int(nr.choice(bad)), N); hs = holes[r][holes[r] != c1]
            if len(hs) == 0: continue
            a = X[r, c1]; b = X[r, hs]
            if nr.random() < noise: c2 = int(nr.choice(hs))
            else:
                d = (cnt[c1, b] >= 1).astype(int) + (cnt[hs, a] >= 1) - (cnt[hs, b] > 1) - int(cnt[c1, a] > 1)
                c2 = int(nr.choice(hs[d == d.min()]))
            b = X[r, c2]; cnt[c1, a] -= 1; cnt[c1, b] += 1; cnt[c2, b] -= 1; cnt[c2, a] += 1; X[r, c1], X[r, c2] = b, a
    return "to", None

def ls_ksat(C, T, rng, noise=0.567):
    """WalkSAT/SKC (Selman, Kautz and Cohen 1994), no restarts."""
    n = max(abs(l) for cl in C for l in cl); a = [False] + [rng.random() < 0.5 for _ in range(n)]
    li = lambda l: 2 * abs(l) + (l < 0); occ = [[] for _ in range(2 * n + 2)]
    for j, cl in enumerate(C):
        for l in cl: occ[li(l)].append(j)
    nt = [sum((l > 0) == a[abs(l)] for l in cl) for cl in C]; uns = [j for j in range(len(C)) if nt[j] == 0]; where = {j: k for k, j in enumerate(uns)}
    t_end = time.monotonic() + T; flips = 0
    while uns:
        flips += 1
        if flips % 1000 == 0 and time.monotonic() > t_end: return "to", None
        cl = C[uns[rng.randrange(len(uns))]]
        brk = [sum(nt[j] == 1 for j in occ[2 * abs(l) + (not a[abs(l)])]) for l in cl]
        if 0 in brk: v = abs(cl[rng.choice([k for k, b in enumerate(brk) if b == 0])])
        elif rng.random() < noise: v = abs(rng.choice(cl))
        else: m = min(brk); v = abs(cl[rng.choice([k for k, b in enumerate(brk) if b == m])])
        old = 2 * v + (not a[v]); new = old ^ 1; a[v] = not a[v]
        for j in occ[old]:
            nt[j] -= 1
            if nt[j] == 0: where[j] = len(uns); uns.append(j)
        for j in occ[new]:
            nt[j] += 1
            if nt[j] == 1:
                k = where.pop(j); last = uns.pop()
                if last != j: uns[k] = last; where[last] = k
    return "sat", a

# ---------------------------------------------------------------- CP-SAT
def run_cp(m, T, seed, workers, t0, read):
    sv = cp_model.CpSolver(); P = sv.parameters; P.num_workers = workers; P.random_seed = seed
    P.max_time_in_seconds = max(0.01, T - (time.monotonic() - t0)); st = sv.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE): return "sat", read(sv)
    return ("unsat" if st == cp_model.INFEASIBLE else "to"), None

def cp_bnq(B, T, seed, workers, strong, t0):
    n = len(B); m = cp_model.CpModel()
    if strong:  # one integer per row over its free columns, AllDifferent on q, q+r, q-r
        dom = [[c for c in range(n) if not B[r, c]] for r in range(n)]
        if any(not d for d in dom): return "unsat", None
        q = [m.NewIntVarFromDomain(cp_model.Domain.FromValues(dom[r]), f"q{r}") for r in range(n)]
        m.AddAllDifferent(q); m.AddAllDifferent([q[r] + r for r in range(n)]); m.AddAllDifferent([q[r] - r for r in range(n)])
        return run_cp(m, T, seed, workers, t0, lambda sv: [sv.Value(x) for x in q])
    x = {(r, c): m.NewBoolVar("") for r in range(n) for c in range(n) if not B[r, c]}  # Boolean cells, ExactlyOne / AtMostOne
    lines = {}
    for (r, c), v in x.items():
        for key in (("r", r), ("c", c), ("d", r - c), ("a", r + c)): lines.setdefault(key, []).append(v)
    if any(("r", r) not in lines for r in range(n)): return "unsat", None
    for key, L in lines.items(): (m.AddExactlyOne if key[0] == "r" else m.AddAtMostOne)(L)
    return run_cp(m, T, seed, workers, t0, lambda sv: [next(c for c in range(n) if (r, c) in x and sv.Value(x[r, c])) for r in range(n)])

def cp_latin(G, T, seed, workers, strong, t0):
    N = len(G); m = cp_model.CpModel(); dom, _, _ = latin_domains(G)
    if any(not d for d in dom.values()): return "unsat", None
    if strong:  # integer per empty cell, AllDifferent per row and column
        y = {k: m.NewIntVarFromDomain(cp_model.Domain.FromValues(d), "") for k, d in dom.items()}
        for i in range(N):
            for L in ([y[i, c] for c in range(N) if (i, c) in y], [y[r, i] for r in range(N) if (r, i) in y]):
                if len(L) > 1: m.AddAllDifferent(L)
        def read(sv):
            X = G.copy()
            for (r, c), v in y.items(): X[r, c] = sv.Value(v)
            return X
        return run_cp(m, T, seed, workers, t0, read)
    x = {(r, c, v): m.NewBoolVar("") for (r, c), d in dom.items() for v in d}  # Boolean (cell, value), ExactlyOne per cell / row-value / column-value
    groups = {}
    for (r, c, v), b in x.items():
        for key in (("cell", r, c), ("rv", r, v), ("cv", c, v)): groups.setdefault(key, []).append(b)
    for r in range(N):
        for v in range(N):
            if v not in set(G[r][G[r] >= 0]) and ("rv", r, v) not in groups: return "unsat", None
            if v not in set(G[:, r][G[:, r] >= 0]) and ("cv", r, v) not in groups: return "unsat", None
    for L in groups.values(): m.AddExactlyOne(L)
    def read(sv):
        X = G.copy()
        for (r, c, v), b in x.items():
            if sv.Value(b): X[r, c] = v
        return X
    return run_cp(m, T, seed, workers, t0, read)

def cp_ksat(C, T, seed, workers, strong, t0):
    n = max(abs(l) for cl in C for l in cl); m = cp_model.CpModel(); x = [None] + [m.NewBoolVar("") for _ in range(n)]
    lit = lambda l: x[l] if l > 0 else x[-l].Not()
    for cl in C:  # strong = native clause; naive = linear sum >= 1 (no global structure exists in random 3-SAT)
        if strong: m.AddBoolOr([lit(l) for l in cl])
        else: m.Add(sum(x[l] if l > 0 else 1 - x[-l] for l in cl) >= 1)
    return run_cp(m, T, seed, workers, t0, lambda sv: [False] + [bool(sv.Value(x[v])) for v in range(1, n + 1)])

CP = {"bnq": cp_bnq, "qwh": cp_latin, "qcp": cp_latin, "ksat": cp_ksat}

# ---------------------------------------------------------------- CNF + CDCL
def amo(vp, L):
    return CardEnc.atmost(L, 1, vpool=vp, encoding=EncType.pairwise if len(L) <= 6 else EncType.seqcounter).clauses if len(L) > 1 else []

def cnf_bnq(B):
    n = len(B); vp = IDPool(); x = {(r, c): vp.id(("x", r, c)) for r in range(n) for c in range(n) if not B[r, c]}
    lines = {}
    for (r, c), v in x.items():
        for key in (("r", r), ("c", c), ("d", r - c), ("a", r + c)): lines.setdefault(key, []).append(v)
    cl = [lines.get(("r", r), []) for r in range(n)]
    for L in lines.values(): cl += amo(vp, L)
    return cl, lambda mod: [next(c for c in range(n) if (r, c) in x and mod[x[r, c] - 1] > 0) for r in range(n)]

def cnf_latin(G):
    N = len(G); dom, rowhas, colhas = latin_domains(G); vp = IDPool(); x = {(r, c, v): vp.id((r, c, v)) for (r, c), d in dom.items() for v in d}
    groups = {}
    for (r, c, v), b in x.items():
        for key in (("cell", r, c), ("rv", r, v), ("cv", c, v)): groups.setdefault(key, []).append(b)
    cl = [groups.get(("cell", r, c), []) for (r, c) in dom]
    cl += [groups.get(("rv", r, v), []) for r in range(N) for v in range(N) if v not in rowhas[r]]
    cl += [groups.get(("cv", c, v), []) for c in range(N) for v in range(N) if v not in colhas[c]]
    for L in groups.values(): cl += amo(vp, L)
    def read(mod):
        X = G.copy()
        for (r, c, v), b in x.items():
            if mod[b - 1] > 0: X[r, c] = v
        return X
    return cl, read

def cnf_ksat(C):
    n = max(abs(l) for cl in C for l in cl)
    return C, lambda mod: [False] + [mod[v - 1] > 0 for v in range(1, n + 1)]

CNF = {"bnq": cnf_bnq, "qwh": cnf_latin, "qcp": cnf_latin, "ksat": cnf_ksat}

def run_sat(fam, inst, name):
    cl, read = CNF[fam](inst)
    if any(len(c) == 0 for c in cl): return "unsat", None
    with Solver(name=name, bootstrap_with=cl) as s:  # no interrupt needed: the parent kills the process at the deadline
        if s.solve():
            return "sat", read(s.get_model())
        return "unsat", None

# ---------------------------------------------------------------- one run (child process)
def solve(fam, inst, solver, T, seed, workers):
    t0 = time.monotonic(); rng = random.Random(seed * 7919 + 1)
    if solver == "ls": st, sol = {"bnq": ls_bnq, "qwh": ls_latin, "qcp": ls_latin, "ksat": ls_ksat}[fam](inst, T, rng)
    elif solver.startswith("cpsat"): st, sol = CP[fam](inst, T, seed, workers, "strong" in solver, t0)
    else: st, sol = run_sat(fam, inst, solver)
    return st, sol, time.monotonic() - t0

def child(conn, fam, inst, solver, T, seed, workers):
    try:
        st, sol, t = solve(fam, inst, solver, T, seed, workers); ver = ""
        if st == "sat":
            ver = int(bool(VERIFY[fam](inst, sol)))
            if not ver: st = "bad"
        conn.send((st, round(t, 4), ver))
    except Exception as e:
        conn.send(("err:" + type(e).__name__, -1, ""))

# ---------------------------------------------------------------- scheduler
def tasks_for(stage, grid, seeds):
    out = []
    for s in range(seeds):  # seed outermost: a truncated run still has balanced cells
        for fam, (sizes, params) in grid.items():
            for n in sizes:
                for p in params:
                    for sv in (SOLVERS if stage == "main" else ("cpsat_strong_w4",)):
                        out.append((fam, n, p, s, sv))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--hours", type=float, default=11.0); ap.add_argument("--stage", default="main", choices=("main", "w4"))
    a = ap.parse_args(); ctx = mp.get_context("fork"); t_start = time.monotonic()
    grid = SMOKE if a.smoke else GRID; seeds = 2 if a.smoke else (SEEDS if a.stage == "main" else SEEDS_W4)
    workers = 1 if a.stage == "main" else 4; jobs = a.jobs if a.stage == "main" else 1
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for r in csv.DictReader(f): done.add((r["family"], r["size"], r["param"], r["seed"], r["solver"]))
    todo = [t for t in tasks_for(a.stage, grid, seeds) if tuple(map(str, t)) not in done]
    print(f"{len(done)} rows present, {len(todo)} runs to do, stage={a.stage}, jobs={jobs}, T={a.T}", flush=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(FIELDS); f.flush()
    cache, running, n_done = {}, [], 0
    while todo or running:
        while todo and len(running) < jobs and time.monotonic() - t_start < a.hours * 3600:
            fam, n, p, s, sv = todo.pop(0); key = (fam, n, p, s)
            if key not in cache: cache.clear(); cache[key] = GEN[fam](n, p, rng_for(*key))
            rc, wc = ctx.Pipe(False); pr = ctx.Process(target=child, args=(wc, fam, cache[key], sv, a.T, s, workers)); pr.start(); wc.close()
            running.append((pr, rc, time.monotonic() + a.T + GRACE, (fam, n, p, s, sv)))
        if todo and time.monotonic() - t_start >= a.hours * 3600 and not running: print("hour budget reached", flush=True); break
        for item in list(running):
            pr, rc, dl, (fam, n, p, s, sv) = item; res = None
            if rc.poll():
                try: res = rc.recv() + (0,)
                except EOFError: res = ("err:EOF", -1, "", 0)
            elif not pr.is_alive(): res = ("err:died", -1, "", 0)
            elif time.monotonic() > dl: pr.kill(); res = ("to", a.T, "", 1)
            if res is None: continue
            pr.join(); rc.close(); running.remove(item); n_done += 1
            w.writerow([fam, n, p, s, sv, workers, a.T, *res]); f.flush()
            if n_done % 50 == 0: print(f"{n_done} runs, {time.monotonic() - t_start:.0f} s", flush=True)
        time.sleep(0.01)
    f.close(); print(f"DONE {n_done} runs in {time.monotonic() - t_start:.0f} s", flush=True)

if __name__ == "__main__":
    main()
