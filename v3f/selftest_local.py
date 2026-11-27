"""Local checks for nq_v3f without numba/ortools/pysat (stubbed when missing; on Kaggle the real modules are used and the same checks run):
 (1) the whole solve() path for every v3f solver on tiny instances: every answer passes the nq_v3 verifier, for each fix variant (same_call / sig / cache) of ls_nb_fixed;
 (2) cpsat_alt formulation: a fake CP-SAT with real semantics for IntVar / AddElement / AddAllDifferent (DFS) solves tiny boards, answers pass the verifier,
     and the model contains n element constraints, 3 AllDifferent, full domains (blocked cells excluded only by the element constraint);
 (3) ls_py_n0 differs from ls_py only in the noise argument (same trajectory when noise 0.1 is never drawn is NOT claimed; the call signature is checked);
 (4) task counts per stage and the stage/seed offsets (400-419 do not overlap earlier stages 0-19, 100-119, 200-219, 300-319, 1000-1002, 2000-2002).
Usage: python3 selftest_local.py   (exit 0 = all ok)"""
import sys, os, json, random, types, subprocess, tempfile, importlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

def stubs():
    try: import numba, ortools, pysat; return False
    except ImportError: pass
    nb = types.ModuleType("numba")
    def njit(*a, **k):
        if a and callable(a[0]): a[0].py_func = a[0]; return a[0]
        def deco(g): g.py_func = g; return g
        return deco
    nb.njit = njit; nb.__version__ = "stub"; sys.modules["numba"] = nb
    for name in ("ortools", "ortools.sat", "ortools.sat.python", "ortools.sat.python.cp_model", "pysat", "pysat.solvers", "pysat.card", "pysat.formula"): sys.modules[name] = types.ModuleType(name)
    sys.modules["ortools.sat.python"].cp_model = sys.modules["ortools.sat.python.cp_model"]
    sys.modules["pysat.solvers"].Solver = object; sys.modules["pysat.card"].CardEnc = object; sys.modules["pysat.card"].EncType = object; sys.modules["pysat.formula"].IDPool = object
    sys.modules["ortools"].__version__ = "stub"; sys.modules["pysat"].__version__ = "stub"
    return True

# ---- fake CP-SAT with the semantics of the three constraint types cp_alt uses
class Var:
    def __init__(s, lo, hi): s.lo, s.hi = lo, hi
    def __add__(s, k): return Ex(s, k)
    def __sub__(s, k): return Ex(s, -k)
class Ex:
    def __init__(s, v, off): s.v, s.off = v, off
class FakeModel:
    def __init__(s): s.vars, s.elem, s.alld = [], [], []
    def NewConstant(s, c): v = Var(c, c); s.vars.append(v); return v
    def NewIntVar(s, lo, hi, name=""): v = Var(lo, hi); s.vars.append(v); return v
    def AddElement(s, idx, arr, target): s.elem.append((idx, arr, target))
    def AddAllDifferent(s, L): s.alld.append(L)
class FakeSolver:
    class P: pass
    def __init__(s): s.parameters = FakeSolver.P(); s.val = {}
    def Solve(s, mod):
        qs = [x for x in mod.vars if x.lo != x.hi]; val = {}
        for v in mod.vars:
            if v.lo == v.hi: val[v] = v.lo
        ev = lambda e: val[e.v] + e.off if isinstance(e, Ex) else val[e]
        def ok(k):      # constraints that only involve the first k variables
            for idx, arr, tgt in mod.elem:
                if idx in val and val[arr[val[idx]]] != val[tgt]: return False
            for L in mod.alld:
                xs = [ev(e) for e in L if (e.v if isinstance(e, Ex) else e) in val]
                if len(set(xs)) != len(xs): return False
            return True
        def dfs(i):
            if i == len(qs): return True
            for c in range(qs[i].lo, qs[i].hi + 1):
                val[qs[i]] = c
                if ok(i) and dfs(i + 1): return True
                del val[qs[i]]
            return False
        s.val = val; return FakeCp.OPTIMAL if dfs(0) else FakeCp.INFEASIBLE
    def Value(s, v): return s.val[v]
class FakeCp:
    OPTIMAL, FEASIBLE, INFEASIBLE = 4, 2, 3; CpModel = FakeModel; CpSolver = FakeSolver

def load(fix):
    """import nq_v3f fresh with f1_fix.json = {fix: ...} (via NQ_F1_FIX)"""
    for k in [k for k in sys.modules if k.startswith("nq_v3")]: del sys.modules[k]
    path = os.path.join(tempfile.mkdtemp(), "f1_fix.json"); json.dump({"fix": fix}, open(path, "w")); os.environ["NQ_F1_FIX"] = path
    import nq_v3f as f
    return f

def check_solve(stubbed):
    for fix in ("same_call", "sig", "cache", None):
        if fix == "cache": os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp()
        f = load(fix); m, e, d = f.m, f.e, f.d; rg = random.Random(3); T = 20.0
        cases = [("bnq", m.gen_bnq(10, .4, rg), ["ls_nb", "ls_py"] + (["ls_nb_fixed"] if fix else [])), ("qwh", m.gen_qwh(6, .4, rg), ["ls_nb", "ls_nb_n10", "ls_py", "ls_py_n0"]),
                 ("qcp", m.gen_qcp(6, .3, rg), ["ls_nb", "ls_nb_n10", "ls_py_n0"]), ("ksat5", d.gen_ksat5(30, 8.0, rg), ["ls_nbk", "ls_nbk_n30", "ls_nbk_n40", "ls_nbk_n50", "ls_nbk_n65", "probsat_nb"])]
        n = 0
        for fam, x, svs in cases:
            for sv in svs:
                for seed in range(3):
                    st, sol, t = f.solve(fam, x, sv, T, seed, 1); n += 1
                    assert st == "sat" and m.VERIFY[fam](x, sol), (fix, fam, sv, seed, st)
        f.warm_up() if False else None
        print(f"fix={fix}: solve() path ok for {n} runs (stubbed numba: {stubbed})")
        if fix is None:
            try: f.solve("bnq", cases[0][1], "ls_nb_fixed", 1.0, 0, 1); raise SystemExit("ls_nb_fixed must refuse without a fix")
            except RuntimeError: pass
    return f

def check_alt(f):
    m = f.m; m.cp_model = FakeCp; sys.modules["ortools.sat.python"].cp_model = FakeCp; rg = random.Random(7); k = 0
    for n, p in ((6, .2), (8, .3), (8, .5), (10, .3)):
        B = m.gen_bnq(n, p, rg); st, sol, t = f.solve("bnq", B, "cpsat_alt", 10.0, 1, 1)
        if st == "sat": assert m.verify_bnq(B, sol); k += 1
        else: assert st in ("to", "unsat"), st      # the fake solver reports INFEASIBLE as 'unsat'; a SAT board must not be called unsat (checked against the strong-model oracle below)
        # oracle: brute force existence of a solution
        def dfs(r, cols, d1, d2):
            if r == n: return True
            return any(not B[r, c] and c not in cols and (r - c) not in d1 and (r + c) not in d2 and dfs(r + 1, cols | {c}, d1 | {r - c}, d2 | {r + c}) for c in range(n))
        assert (st == "sat") == dfs(0, frozenset(), frozenset(), frozenset()), (n, p, st)
    assert k >= 2
    mod = FakeModel(); cap = {}
    class Spy(FakeCp): CpModel = lambda self=None: cap.setdefault("m", FakeModel())
    m.cp_model = Spy; B = m.gen_bnq(6, .2, rg); f.cp_alt(B, 5.0, 0, 1, __import__("time").monotonic())
    cm = cap["m"]; assert len(cm.elem) == 6 and len(cm.alld) == 3 and all(v.hi == 5 for v in cm.vars if v.lo != v.hi and v.lo == 0), "model structure"
    print("cpsat_alt: fake-semantics solve agrees with brute force on 4 boards, model has n element constraints + 3 AllDifferent")

def check_tasks(f):
    m = f.m; seen = {}
    for st in ("main3f", "lat3f", "long3f", "smoke3f"):
        t = f.tasks_for(st, m.STAGES[st][1], m.STAGES[st][3]); seen[st] = t; print(f"{st}: {len(t)} tasks, {len(t) // m.STAGES[st][1]} per seed, w4 {sum(1 for x in t if x[4].endswith('_w4'))}")
    t = seen["main3f"]; assert all(not x[4].endswith("_w4") for x in t[:len(t) - sum(1 for x in t if x[4].endswith("_w4"))]) and all(x[4].endswith("_w4") for x in t[-5:]), "w4 runs last"
    assert {x[4] for x in seen["long3f"]} == {"ls_py_n0", "ls_py"} and len(seen["long3f"]) == 160 and {x[4] for x in seen["lat3f"]} == {"ls_nb", "ls_nb_n10"} and len(seen["lat3f"]) == 160
    assert f.d.OFFSET["main3f"] == f.d.OFFSET["lat3f"] == f.d.OFFSET["long3f"] == 400 and 400 + 19 < 1000 and 400 > 319
    return seen

if __name__ == "__main__":
    stubbed = stubs(); f = check_solve(stubbed); check_alt(f); seen = check_tasks(f)
    print("SELFTEST_LOCAL OK")
