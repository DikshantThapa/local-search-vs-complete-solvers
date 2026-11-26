"""Local checks without numba/ortools/pysat: (1) pure-python reference of nq_ls_c.c (same xorshift stream, same draws) must give the
SAME answer as the compiled binary on tiny boards; (2) with numba/ortools/pysat stubbed (njit = identity) the v3e solve() path runs
ls_nb, ls_nb_n10, ls_nbk, probsat_nb, ls_c on tiny instances and every answer passes the nq_v3 verifier; (3) task counts.
Usage: python3 selftest_local.py [path/to/nq_ls_c]   (exit 0 = all ok). On Kaggle (real numba) step 2 simply uses the real modules."""
import sys, os, random, subprocess, tempfile, types
import numpy as np
M64 = (1 << 64) - 1
class Ref:
    def __init__(self, seed):
        self.S = (((seed * 0x9E3779B97F4A7C15) + 0x1234567) & M64) | 1
        for _ in range(8): self.u()
    def u(self):
        S = self.S; S ^= S >> 12; S ^= (S << 25) & M64; S ^= S >> 27; self.S = S
        return (((S * 2685821657736338717) & M64) >> 11) / 9007199254740992.0
    def rnd(self, k): return int(self.u() * k)
def ref_ls(B, seed, max_restarts=30):
    n = len(B); R = Ref(seed)
    for _ in range(max_restarts):
        col = [0] * n; d1 = [0] * (2 * n); d2 = [0] * (2 * n); pos = [0] * n
        for r in range(n):
            best, k, ch = 1 << 30, 0, 0
            for c in range(n):
                x = col[c] + d1[r - c + n] + d2[r + c] + 5 * int(B[r][c])
                if x < best: best, k, ch = x, 1, c
                elif x == best:
                    k += 1
                    if R.rnd(k) == 0: ch = c
            pos[r] = ch; col[ch] += 1; d1[r - ch + n] += 1; d2[r + ch] += 1
        for step in range(20000):
            nb, r = 0, -1
            for i in range(n):
                if col[pos[i]] + d1[i - pos[i] + n] + d2[i + pos[i]] - 3 + int(B[i][pos[i]]) > 0:
                    nb += 1
                    if R.rnd(nb) == 0: r = i
            if nb == 0: return pos
            col[pos[r]] -= 1; d1[r - pos[r] + n] -= 1; d2[r + pos[r]] -= 1; R.u()
            best, k, ch = 1 << 30, 0, 0
            for c in range(n):
                x = col[c] + d1[r - c + n] + d2[r + c] + 5 * int(B[r][c])
                if x < best: best, k, ch = x, 1, c
                elif x == best:
                    k += 1
                    if R.rnd(k) == 0: ch = c
            pos[r] = ch; col[ch] += 1; d1[r - ch + n] += 1; d2[r + ch] += 1
    return None

def check_c(binp):
    rg = random.Random(5); ok = 0
    for n, p in ((8, .3), (10, .5), (12, .6), (16, .5), (20, .6)):
        for s in range(5):
            B = np.array([[rg.random() < p for _ in range(n)] for _ in range(n)])
            f = tempfile.mktemp(); open(f, "w").write(f"{n}\n" + "\n".join("".join("1" if x else "0" for x in row) for row in B) + "\n")
            out = subprocess.run([binp, f, str(s), "5"], capture_output=True, text=True).stdout.split(); os.unlink(f)
            if out[0] == "SAT":   # reference only on boards the binary solved (an unsolvable board would burn the whole restart cap)
                q = [int(x) for x in out[2:]]; ref = ref_ls(B, s)
                if ref is None: continue   # C needed more restarts than the reference cap: not compared
                assert ref == q, (n, p, s, ref, q); ok += 1
    print("C == python reference on", ok, "solved boards"); assert ok >= 10

def stubs():
    try: import numba, ortools, pysat; return False
    except ImportError: pass
    nb = types.ModuleType("numba"); nb.njit = lambda f=None, **k: f if f else (lambda g: g); nb.__version__ = "stub"; sys.modules["numba"] = nb
    for name in ("ortools", "ortools.sat", "ortools.sat.python", "ortools.sat.python.cp_model", "pysat", "pysat.solvers", "pysat.card", "pysat.formula"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["ortools.sat.python"].cp_model = sys.modules["ortools.sat.python.cp_model"]
    sys.modules["pysat.solvers"].Solver = object; sys.modules["pysat.card"].CardEnc = object; sys.modules["pysat.card"].EncType = object; sys.modules["pysat.formula"].IDPool = object
    sys.modules["ortools"].__version__ = "stub"; sys.modules["pysat"].__version__ = "stub"
    return True

def check_solve(binp):
    stubbed = stubs(); os.environ["NQ_LS_C_BIN"] = binp; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nq_v3e as e; m = e.m
    rg = random.Random(3); T = 20.0
    cases = [("bnq", m.gen_bnq(10, .4, rg), ("ls_nb", "ls_c")), ("qwh", m.gen_qwh(6, .4, rg), ("ls_nb", "ls_nb_n10")), ("qcp", m.gen_qcp(6, .3, rg), ("ls_nb", "ls_nb_n10")),
             ("ksat", m.gen_ksat(30, 3.0, rg), ("ls_nb", "probsat_nb")), ("ksat5", e.d.gen_ksat5(30, 8.0, rg), ("ls_nbk", "probsat_nb"))]
    for fam, x, svs in cases:
        for sv in svs:
            for seed in range(3):
                st, sol, t = e.solve(fam, x, sv, T, seed, 1)
                assert st == "sat" and m.VERIFY[fam](x, sol), (fam, sv, seed, st)
    print("solve() path ok for", sum(len(c[2]) for c in cases), "solver/family pairs (stubbed numba: %s)" % stubbed)
    for stage, k in (("confirm3e", None), ("long3e", None), ("smoke3e", None)):
        t = e.tasks_for(stage, m.STAGES[stage][1], m.STAGES[stage][3]); print(stage, "tasks", len(t), "per seed", len(t) // m.STAGES[stage][1])
    return e

if __name__ == "__main__":
    binp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nq_ls_c")
    assert os.path.exists(binp), "compile nq_ls_c.c first"
    check_c(binp); check_solve(binp); print("SELFTEST_LOCAL OK")
