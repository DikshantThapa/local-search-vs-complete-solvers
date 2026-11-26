"""nq_v3: strengthened version of nq_v2 (design: nqueens/v3/DESIGN_NOTE_V3.md, fixed BEFORE any non-smoke run).
Changes against v2: (1) environment JSON written next to the CSV at run start; (2) strict budget: every run gets exactly T,
the parent kills at T + KILL_GRACE only to allow IPC, any answer with t > T is stored as status 'to' in the CSV itself;
(3) new compiled local search 'ls_nb' (numba) for all four families, old Python LS kept as 'ls_py'; (4) stages main10 / long60 / long300.
Families: bnq (Blocked n-Queens, p = blocked prob.), qwh (quasigroup with holes, h = hole fraction, always SAT),
qcp (quasigroup completion, f = preassigned fraction with forward checking), ksat (random 3-SAT, r = clauses/variables).
Each (instance, solver) run is a forked child; every reported solution is checked by the independent verifier.
CSV: one row per run, appended and flushed at once; rerunning with the same OUT resumes (finished keys are skipped).
Usage: python nq_v3.py OUT.csv --stage main10|long60|long300|smoke [--jobs 4] [--hours 10] [--seeds K] [--solvers a,b]
"""
import argparse, csv, json, os, platform, random, sys, time
import multiprocessing as mp
import numpy as np
import numba as nb
from numba import njit
from ortools.sat.python import cp_model
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

FULL_GRID = {
    "bnq": ((50, 100, 200), (0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96)),
    "qwh": ((20, 30, 40), (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70)),
    "qcp": ((20, 30, 40), (0.30, 0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.50)),
    "ksat": ((150, 250, 350), (3.8, 4.0, 4.1, 4.2, 4.26, 4.3, 4.4, 4.6)),
}
LONG60_GRID = {"bnq": ((100, 200), (0.84, 0.88, 0.92)), "qwh": ((30, 40), (0.35, 0.45, 0.55)),
               "qcp": ((30, 40), (0.35, 0.42, 0.50)), "ksat": ((250, 350), (4.1, 4.26, 4.4))}
LONG300_GRID = {"bnq": ((200,), (0.84, 0.88, 0.92)), "qwh": ((40,), (0.35, 0.45, 0.55)),
                "qcp": ((40,), (0.35, 0.42, 0.50)), "ksat": ((350,), (4.1, 4.26, 4.4))}
SMOKE_GRID = {"bnq": ((12, 30), (0.5, 0.9)), "qwh": ((8,), (0.4,)), "qcp": ((8,), (0.42,)), "ksat": ((40,), (3.5, 4.26))}
COMPLETE = ("cpsat_strong", "cpsat_strong_w4", "cadical195")
# stage -> (grid, seeds, T, solvers)
STAGES = {
    "main10": (FULL_GRID, 10, 10.0, ("ls_nb", "ls_py", "cpsat_naive", "cpsat_strong", "glucose4", "cadical195")),
    "long60": (LONG60_GRID, 5, 60.0, ("ls_nb", "cpsat_strong", "cpsat_strong_w4", "cadical195")),
    "long300": (LONG300_GRID, 3, 300.0, ("ls_nb", "cpsat_strong", "cpsat_strong_w4", "cadical195")),
    "smoke": (SMOKE_GRID, 2, 2.0, ("ls_nb", "ls_py", "cpsat_naive", "cpsat_strong", "cpsat_strong_w4", "glucose4", "cadical195")),
}
KILL_GRACE = 0.5    # parent kills a child at T + 0.5 s; this only allows IPC, answers with t > T are stored as 'to'
FIELDS = ["stage", "family", "size", "param", "seed", "solver", "workers", "T", "status", "t", "t_raw", "verified", "killed"]
# fixed LS constants (DESIGN_NOTE_V3.md section 4); tuned only on tuning seeds >= 1000, never on seeds used in stages
NB_NOISE = {"bnq": 0.0, "qwh": 0.0, "qcp": 0.0, "ksat": 0.567}
NB_RESTART = {"bnq": 20000, "latin": 50}   # bnq: moves per restart; latin: moves per restart = 50 * N * N

def environment(a):
    cpu = platform.processor() or platform.machine()
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"): cpu = l.split(":", 1)[1].strip(); break
    except OSError:
        if sys.platform == "darwin":
            import subprocess; cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
    try: ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError): ram = None
    import ortools, pysat
    return {"python": sys.version.split()[0], "ortools": ortools.__version__, "python-sat": getattr(pysat, "__version__", "?"), "numpy": np.__version__,
            "numba": nb.__version__, "cpu_model": cpu, "cpu_count_logical": os.cpu_count(),
            "cpu_count_usable": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
            "ram_gb": round(ram, 2) if ram else None, "os": platform.platform(), "machine": platform.machine(),
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv[1:], "stage": a.stage,
            "nb_noise": NB_NOISE, "nb_restart": NB_RESTART, "kill_grace_s": KILL_GRACE}

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
def lspy_bnq(B, T, rng):
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

def lspy_latin(G, T, rng, noise=0.1):
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

def lspy_ksat(C, T, rng, noise=0.567):
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

# ---------------------------------------------------------------- compiled local search (ls_nb)
# Kernels run a chunk of moves and return 1 if solved; Python only checks the clock between chunks (numba has no clock).
@njit
def nb_seed(s):
    np.random.seed(s)

@njit
def rnd(k):
    return int(np.random.random() * k)

@njit
def bnq_init(H, pos, col, d1, d2):
    n = len(pos); col[:] = 0; d1[:] = 0; d2[:] = 0
    for r in range(n):
        best = 1 << 30; k = 0; ch = 0
        for c in range(n):
            x = col[c] + d1[r - c + n] + d2[r + c] + 5 * H[r, c]
            if x < best: best = x; k = 1; ch = c
            elif x == best:
                k += 1
                if rnd(k) == 0: ch = c
        pos[r] = ch; col[ch] += 1; d1[r - ch + n] += 1; d2[r + ch] += 1

@njit
def bnq_steps(H, pos, col, d1, d2, noise, steps):
    """Min-conflicts: random conflicted queen, move it to a min-conflict column (blocked cell penalty 5); with prob `noise` a random column."""
    n = len(pos)
    for _ in range(steps):
        nb_ = 0; r = -1
        for i in range(n):
            if col[pos[i]] + d1[i - pos[i] + n] + d2[i + pos[i]] - 3 + H[i, pos[i]] > 0:
                nb_ += 1
                if rnd(nb_) == 0: r = i
        if nb_ == 0: return 1
        col[pos[r]] -= 1; d1[r - pos[r] + n] -= 1; d2[r + pos[r]] -= 1
        if np.random.random() < noise: ch = rnd(n)
        else:
            best = 1 << 30; k = 0; ch = 0
            for c in range(n):
                x = col[c] + d1[r - c + n] + d2[r + c] + 5 * H[r, c]
                if x < best: best = x; k = 1; ch = c
                elif x == best:
                    k += 1
                    if rnd(k) == 0: ch = c
        pos[r] = ch; col[ch] += 1; d1[r - ch + n] += 1; d2[r + ch] += 1
    return 0

@njit
def latin_init(G, hole, X, cnt, cost):
    N = len(G); cnt[:, :] = 0
    for r in range(N):
        seen = np.zeros(N, np.bool_)
        for c in range(N):
            if not hole[r, c]: seen[G[r, c]] = True
        miss = np.empty(N, np.int64); m = 0
        for v in range(N):
            if not seen[v]: miss[m] = v; m += 1
        p = np.random.permutation(m); k = 0
        for c in range(N):
            if hole[r, c]: X[r, c] = miss[p[k]]; k += 1
            else: X[r, c] = G[r, c]
    cost[0] = 0
    for r in range(N):
        for c in range(N): cnt[c, X[r, c]] += 1
    for c in range(N):
        for v in range(N):
            if cnt[c, v] > 1: cost[0] += cnt[c, v] - 1

@njit
def latin_steps(hole, X, cnt, cost, noise, steps):
    """Row-permutation min-conflicts: rows stay permutations; pick a random conflicted hole cell, swap it with the hole of the same row
    that lowers the column-duplicate cost most (ties random); with prob `noise` swap with a random hole of the row."""
    N = len(X); hs = np.empty(N, np.int64)
    for _ in range(steps):
        if cost[0] == 0: return 1
        nbad = 0
        for r in range(N):
            for c in range(N):
                if hole[r, c] and cnt[c, X[r, c]] > 1: nbad += 1
        k = rnd(nbad); r1 = 0; c1 = 0; i = 0
        for r in range(N):
            for c in range(N):
                if hole[r, c] and cnt[c, X[r, c]] > 1:
                    if i == k: r1 = r; c1 = c
                    i += 1
        nh = 0
        for c in range(N):
            if hole[r1, c] and c != c1: hs[nh] = c; nh += 1
        if nh == 0: continue
        a = X[r1, c1]
        if np.random.random() < noise: c2 = hs[rnd(nh)]
        else:
            best = 1 << 30; t = 0; c2 = hs[0]
            for j in range(nh):
                cc = hs[j]; b = X[r1, cc]
                d = (1 if cnt[c1, b] >= 1 else 0) + (1 if cnt[cc, a] >= 1 else 0) - (1 if cnt[cc, b] > 1 else 0) - (1 if cnt[c1, a] > 1 else 0)
                if d < best: best = d; t = 1; c2 = cc
                elif d == best:
                    t += 1
                    if rnd(t) == 0: c2 = cc
        b = X[r1, c2]
        d = (1 if cnt[c1, b] >= 1 else 0) + (1 if cnt[c2, a] >= 1 else 0) - (1 if cnt[c2, b] > 1 else 0) - (1 if cnt[c1, a] > 1 else 0)
        cnt[c1, a] -= 1; cnt[c1, b] += 1; cnt[c2, b] -= 1; cnt[c2, a] += 1; X[r1, c1] = b; X[r1, c2] = a; cost[0] += d
    return 0

@njit
def sat_init(Cm, a, nt, uns, where, occ_off, occ):
    m = len(Cm); st = 0
    for j in range(m):
        nt[j] = 0
        for k in range(3):
            l = Cm[j, k]
            if (l > 0) == a[abs(l)]: nt[j] += 1
        where[j] = -1
        if nt[j] == 0: uns[st] = j; where[j] = st; st += 1
    return st

@njit
def sat_steps(Cm, a, nt, uns, where, occ_off, occ, nuns, noise, flips):
    """WalkSAT/SKC: random unsatisfied clause; free move if some var has break 0, else random var with prob `noise`, else min break."""
    brk = np.zeros(3, np.int64)
    for _ in range(flips):
        if nuns == 0: return 1, nuns
        j = uns[rnd(nuns)]; zero = 0; mn = 1 << 30
        for k in range(3):
            v = abs(Cm[j, k]); t = 2 * v + (0 if a[v] else 1); b = 0
            for q in range(occ_off[t], occ_off[t + 1]):
                if nt[occ[q]] == 1: b += 1
            brk[k] = b
            if b < mn: mn = b
        c = 0; ch = 0
        if mn == 0:
            for k in range(3):
                if brk[k] == 0:
                    c += 1
                    if rnd(c) == 0: ch = k
        elif np.random.random() < noise: ch = rnd(3)
        else:
            for k in range(3):
                if brk[k] == mn:
                    c += 1
                    if rnd(c) == 0: ch = k
        v = abs(Cm[j, ch]); old = 2 * v + (0 if a[v] else 1); new = old ^ 1; a[v] = not a[v]
        for q in range(occ_off[old], occ_off[old + 1]):
            jj = occ[q]; nt[jj] -= 1
            if nt[jj] == 0: uns[nuns] = jj; where[jj] = nuns; nuns += 1
        for q in range(occ_off[new], occ_off[new + 1]):
            jj = occ[q]; nt[jj] += 1
            if nt[jj] == 1:
                k2 = where[jj]; nuns -= 1; last = uns[nuns]
                uns[k2] = last; where[last] = k2; where[jj] = -1
    return 0, nuns

def lsnb_bnq(B, T, seed, noise=None, t0=None):
    noise = NB_NOISE["bnq"] if noise is None else noise
    n = len(B); t_end = (t0 or time.monotonic()) + T; H = B.astype(np.int64); nb_seed(seed)
    pos = np.zeros(n, np.int64); col = np.zeros(n, np.int64); d1 = np.zeros(2 * n, np.int64); d2 = np.zeros(2 * n, np.int64)
    while time.monotonic() < t_end:
        bnq_init(H, pos, col, d1, d2); left = NB_RESTART["bnq"]
        while left > 0:
            if bnq_steps(H, pos, col, d1, d2, noise, 500): return "sat", pos.tolist()
            left -= 500
            if time.monotonic() > t_end: return "to", None
    return "to", None

def lsnb_latin(G, T, seed, noise=None, t0=None, fam="qwh"):
    noise = NB_NOISE[fam] if noise is None else noise
    N = len(G); t_end = (t0 or time.monotonic()) + T; nb_seed(seed); G = G.astype(np.int64); hole = G < 0
    X = np.zeros((N, N), np.int64); cnt = np.zeros((N, N), np.int64); cost = np.zeros(1, np.int64)
    while time.monotonic() < t_end:
        latin_init(G, hole, X, cnt, cost); left = NB_RESTART["latin"] * N * N; chunk = 2000
        while left > 0:
            if latin_steps(hole, X, cnt, cost, noise, chunk): return "sat", X.copy()
            left -= chunk
            if time.monotonic() > t_end: return "to", None
    return "to", None

def lsnb_ksat(C, T, seed, noise=None, t0=None):
    noise = NB_NOISE["ksat"] if noise is None else noise
    t_end = (t0 or time.monotonic()) + T; Cm = np.array(C, np.int64); n = int(np.abs(Cm).max()); m = len(Cm); nb_seed(seed)
    li = 2 * np.abs(Cm) + (Cm < 0); off = np.zeros(2 * n + 3, np.int64); np.add.at(off, li.ravel() + 1, 1); off = np.cumsum(off)
    order = np.argsort(li.ravel(), kind="stable"); occ = (order // 3).astype(np.int64)
    a = np.zeros(n + 1, np.bool_); a[1:] = np.random.default_rng(seed).random(n) < .5
    nt = np.zeros(m, np.int64); uns = np.zeros(m, np.int64); where = np.zeros(m, np.int64)
    nuns = sat_init(Cm, a, nt, uns, where, off, occ)
    while True:
        ok, nuns = sat_steps(Cm, a, nt, uns, where, off, occ, nuns, noise, 20000)
        if ok: return "sat", a
        if time.monotonic() > t_end: return "to", None

def warm_up():
    """compile all kernels before forking children (compile time is never charged to a run)."""
    r = random.Random(0); lsnb_bnq(gen_bnq(8, .3, r), 0.5, 0); G = gen_qwh(5, .4, r); lsnb_latin(G, 0.5, 0); lsnb_ksat(gen_ksat(20, 4.0, r), 0.5, 0)

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
    t0 = time.monotonic(); rng = random.Random(seed * 7919 + 1); sd = seed * 7919 + 1
    if solver == "ls_py": st, sol = {"bnq": lspy_bnq, "qwh": lspy_latin, "qcp": lspy_latin, "ksat": lspy_ksat}[fam](inst, T, rng)
    elif solver == "ls_nb":
        st, sol = lsnb_bnq(inst, T, sd, t0=t0) if fam == "bnq" else lsnb_ksat(inst, T, sd, t0=t0) if fam == "ksat" else lsnb_latin(inst, T, sd, t0=t0, fam=fam)
    elif solver.startswith("cpsat"): st, sol = CP[fam](inst, T, seed, workers, "strong" in solver, t0)
    else: st, sol = run_sat(fam, inst, solver)
    return st, sol, time.monotonic() - t0

def child(conn, fam, inst, solver, T, seed, workers):
    try:
        st, sol, t = solve(fam, inst, solver, T, seed, workers); ver = ""
        if st == "sat":
            ver = int(bool(VERIFY[fam](inst, sol)))
            if not ver: st = "bad"
        conn.send((st, t, ver))
    except Exception as e:
        conn.send(("err:" + type(e).__name__, -1, ""))

def strict(res, T):
    """strict budget: any answer (sat / unsat / bad) found after T is a timeout. returns CSV fields status, t, t_raw, verified, killed."""
    st, t, ver = res
    if st == "to" or (st in ("sat", "unsat", "bad") and t > T): return "to", T, round(t, 4), "", 0
    return st, round(t, 4), round(t, 4), ver, 0

# ---------------------------------------------------------------- scheduler
def tasks_for(stage, seeds, solvers):
    grid = STAGES[stage][0]; out = []
    for grp in (False, True):  # 1-worker runs first (4 in parallel), then the 4-worker CP-SAT runs alone
        for s in range(seeds):  # seed outermost: a truncated run still has balanced cells
            for fam, (sizes, params) in grid.items():
                for n in sizes:
                    for p in params:
                        out += [(fam, n, p, s, sv) for sv in solvers if sv.endswith("_w4") == grp]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--stage", required=True, choices=tuple(STAGES))
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=None); ap.add_argument("--solvers", default=None); ap.add_argument("--T", type=float, default=None)
    a = ap.parse_args(); ctx = mp.get_context("fork"); t_start = time.monotonic()
    _, seeds, T, solvers = STAGES[a.stage]; seeds = a.seeds or seeds; T = a.T or T; solvers = tuple(a.solvers.split(",")) if a.solvers else solvers
    envf = a.out + ".env.json"; envs = json.load(open(envf)) if os.path.exists(envf) else []
    envs.append({**environment(a), "T": T, "seeds": seeds, "solvers": solvers, "jobs": a.jobs}); json.dump(envs, open(envf, "w"), indent=1)
    warm_up(); done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for r in csv.DictReader(f): done.add((r["family"], r["size"], r["param"], r["seed"], r["solver"]))
    todo = [t for t in tasks_for(a.stage, seeds, solvers) if tuple(map(str, t)) not in done]
    print(f"{len(done)} rows present, {len(todo)} runs to do, stage={a.stage}, jobs={a.jobs}, T={T}", flush=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(FIELDS); f.flush()
    cache, running, n_done, used = {}, [], 0, 0; out_of_time = lambda: time.monotonic() - t_start >= a.hours * 3600
    while todo or running:
        while todo and not out_of_time():
            fam, n, p, s, sv = todo[0]; wk = 4 if sv.endswith("_w4") else 1
            if used + wk > a.jobs and running: break
            todo.pop(0); key = (fam, n, p, s)
            if key not in cache: cache.clear(); cache[key] = GEN[fam](n, p, rng_for(*key))
            rc, wc = ctx.Pipe(False); pr = ctx.Process(target=child, args=(wc, fam, cache[key], sv, T, s, wk)); pr.start(); wc.close()
            running.append((pr, rc, time.monotonic() + T + KILL_GRACE, (fam, n, p, s, sv), wk)); used += wk
        if todo and out_of_time() and not running: print("hour budget reached", flush=True); break
        for item in list(running):
            pr, rc, dl, (fam, n, p, s, sv), wk = item; res = None
            if rc.poll():
                try: res = strict(rc.recv(), T)
                except EOFError: res = ("err:EOF", -1, -1, "", 0)
            elif not pr.is_alive(): res = ("err:died", -1, -1, "", 0)
            elif time.monotonic() > dl: pr.kill(); res = ("to", T, "", "", 1)
            if res is None: continue
            pr.join(); rc.close(); running.remove(item); used -= wk; n_done += 1
            w.writerow([a.stage, fam, n, p, s, sv, wk, T, *res]); f.flush()
            if n_done % 50 == 0: print(f"{n_done} runs, {time.monotonic() - t_start:.0f} s", flush=True)
        time.sleep(0.01)
    f.close(); print(f"DONE {n_done} runs in {time.monotonic() - t_start:.0f} s", flush=True)

if __name__ == "__main__":
    main()
