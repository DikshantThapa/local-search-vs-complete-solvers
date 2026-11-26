"""nq_v3c: confirmatory stage on FRESH instances, built on nq_v3.py (unchanged, copied beside this file).
Design: DESIGN_NOTE_V3C.md (committed with an external time stamp BEFORE the first run).
Adds (1) stage 'confirm': seeds 100-119 (never used for generation in v2/v3 main10/long and never for tuning, which used 1000-1002),
(2) solver 'ls_nb_w4': best-of-4 independent ls_nb workers (seeds 10*s+i), first verified SAT wins, one core each, so it is core-matched
to cpsat_strong_w4. Everything else (generators, verifier, strict budget, scheduler, env json) is nq_v3.py.
Usage: python nq_v3c.py OUT.csv --stage confirm [--jobs 4] [--hours H] [--seeds K]   (smoke2: tiny test)"""
import sys, time, multiprocessing as mp
import nq_v3 as m

STAGES_NEW = {
    "confirm": (m.FULL_GRID, 20, 10.0, ("ls_nb", "ls_nb_w4", "cpsat_strong", "cpsat_strong_w4", "cadical195")),
    "smoke2": (m.SMOKE_GRID, 2, 2.0, ("ls_nb", "ls_nb_w4", "cpsat_strong", "cadical195")),
}
m.STAGES.update(STAGES_NEW)
_orig_solve = m.solve
SEED0 = {"confirm": 100}   # instance seeds start here (the scheduler counts 0..K-1; we offset inside generator and solver via OFFSET)
OFFSET = 100

def _worker(conn, fam, inst, T, seed):
    try:
        st, sol, t = _orig_solve(fam, inst, "ls_nb", T, seed, 1)
        conn.send((st, sol, t))
    except Exception as e:
        conn.send(("err:" + type(e).__name__, None, -1))

def solve_best4(fam, inst, T, seed):
    t0 = time.monotonic(); ctx = mp.get_context("fork"); procs = []
    for i in range(4):
        rc, wc = ctx.Pipe(False); p = ctx.Process(target=_worker, args=(wc, fam, inst, T, 10 * seed + i), daemon=True)
        p.start(); wc.close(); procs.append((p, rc))
    res, pending = ("to", None), list(procs)
    while pending and time.monotonic() - t0 < T + 0.3:
        for item in list(pending):
            p, rc = item
            if rc.poll():
                try: st, sol, _ = rc.recv()
                except EOFError: st, sol = "err:EOF", None
                pending.remove(item)
                if st == "sat": res = ("sat", sol); pending = []; break
        time.sleep(0.005)
    for p, rc in procs:
        if p.is_alive(): p.kill()
    return res[0], res[1], time.monotonic() - t0

def solve(fam, inst, solver, T, seed, workers):
    if solver == "ls_nb_w4": st, sol, t = solve_best4(fam, inst, T, seed); return st, sol, t
    return _orig_solve(fam, inst, solver, T, seed, workers)
m.solve = solve

# instance seed offset: shift the seed numbers used for generation so 'confirm' uses 100..119
_orig_rng = m.rng_for
def rng_for(fam, n, p, seed): return _orig_rng(fam, n, p, seed + OFFSET if STAGE[0] == "confirm" else seed)
STAGE = [None]
for i, a in enumerate(sys.argv):
    if a == "--stage" and i + 1 < len(sys.argv): STAGE[0] = sys.argv[i + 1]
m.rng_for = rng_for
if __name__ == "__main__":
    m.main()
