import pandas as pd, numpy as np
R = '/Users/dikshantthapa/Desktop/Claude-projects/Research/nqueens/results_v3f/'
m = pd.read_csv(R + 'main3f.csv'); l = pd.read_csv(R + 'lat3f.csv'); g = pd.read_csv(R + 'long3f.csv')
c = pd.read_csv(R + 'analysis_v3f/cells.csv'); dg = pd.read_csv(R + 'f1_diag.csv')
for x in (m, l, g): x['s'] = ((x.status == 'sat') & (x.verified == 1)).astype(int)
L = pd.concat([l, g])

def w(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n; ce = (p + z * z / 2 / n) / dd; h = z * np.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / dd
    return ce - h, ce + h

def sv(d, **kw):
    q = d
    for k, v in kw.items(): q = q[q[k] == v]
    return int(q.s.sum()), len(q)

def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)

def pp(p): return ('%g' % p)

# Table 4 (main text): bnq v3f
def t_bnq():
    sol = ['ls_nb', 'ls_nb_fixed', 'ls_py', 'cpsat_strong', 'cpsat_alt', 'cpsat_strong_w4', 'cpsat_alt_w4']
    rows = []
    for n, p in [(100, .84), (100, .88), (100, .92), (200, .84), (200, .88), (200, .9), (200, .92)]:
        r = [f'n={n}, p={pp(p)}']
        for s in sol:
            k, nn = sv(m, family='bnq', size=n, param=p, solver=s)
            r.append(f'{k}' if nn else '-')
        rows.append(r)
    return table(['cell', 'ls_nb', 'ls_nb_fixed', 'ls_py', 'cpsat_strong (1)', 'cpsat_alt (1)', 'cpsat_strong_w4 (4)', 'cpsat_alt_w4 (4)'], rows)

# Table 6: 2x2 by family/size
def t_2x2():
    arms = ['ls_nb', 'ls_py_n0', 'ls_nb_n10', 'ls_py']
    rows = []
    for f, n in [('qcp', 30), ('qcp', 40), ('qwh', 30), ('qwh', 40)]:
        rows.append([f'{f} {n}'] + [sv(L, solver=s, family=f, size=n)[0] for s in arms])
    rows.append(['all (80 instances)'] + [sv(L, solver=s)[0] for s in arms])
    return table(['family, N', 'fast, noise 0 (ls_nb)', 'slow, noise 0 (ls_py_n0)', 'fast, noise 0.1 (ls_nb_n10)', 'slow, noise 0.1 (ls_py)'], rows)

# Table 8: 5-SAT noise sweep
def t_noise():
    k5 = m[m.family == 'ksat5']
    cols = [('ls_nbk_n30', '0.30'), ('ls_nbk_n40', '0.40'), ('ls_nbk_n50', '0.50'), ('ls_nbk', '0.567'), ('ls_nbk_n65', '0.65'), ('probsat_nb', 'probSAT'), ('cadical195', 'CaDiCaL')]
    rows = []
    for r in (20, 20.5):
        row = [f'r={pp(r)}']
        for s, _ in cols:
            k, n = sv(k5, param=r, solver=s); lo, hi = w(k, n); row.append(f'{k}/{n} [{lo:.2f}, {hi:.2f}]')
        rows.append(row)
    return table(['cell'] + [f'WalkSAT noise {t}' if t[0] == '0' else t for _, t in cols], rows)

# B1 rows
def b1():
    rows = []
    for st, d in (('main3f', m), ('lat3f', l), ('long3f', g)):
        for s in d.solver.unique():
            q = d[d.solver == s]; ex = q.t_raw - q['T']
            rows.append([st, s, len(q), int(q.killed.sum()), int((ex > 0).sum()), int((ex > 0.05).sum()), f'{max(0, ex.max()):.2f}'])
    return '\n'.join('| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows)

# B4 rows
def b4():
    rows = []
    for s in ('ls_nb', 'ls_nb_fixed'):
        q = m[(m.solver == s) & (m.family == 'bnq') & (m.s == 1)]
        def f(n):
            qq = q[q['size'] == n]; return f'{len(qq)}, {qq.t.min():.3f}, {qq.t.median():.3f}'
        rows.append(f'| v3f {s} | {len(q)} | {q.t.min():.3f} | {q.t.median():.3f} | - | {f(100)} | {f(200)} |')
    return '\n'.join(rows)

# diag table B15
def b15():
    dg['first'] = dg.t_seed + dg.t_init1 + dg.t_steps1; dg['second'] = dg.t_init2 + dg.t_steps2
    dg['sg'] = (dg.sig_init_after - dg.sig_init_before) + (dg.sig_steps_after - dg.sig_steps_before)
    rows = []
    for (cnd, mode), q in dg.groupby(['cond', 'mode'], sort=False):
        rows.append([cnd, mode, len(q), f'{q["first"].median():.4f}', f'{q["first"].min():.4f}', f'{q["first"].max():.4f}', f'{q.second.median():.4f}', f'{q.compile_events.median():g}', f'{q.sg.median():g}'])
    return table(['condition', 'process', 'n', 'first-call sum, median (s)', 'min', 'max', 'second call, median (s)', 'compile events, median', 'new signatures, median'], rows)

# per-cell A tables
def fmt(r): return f'{int(r.solved)}/{int(r.n)} [{r.wilson_lo:.2f}, {r.wilson_hi:.2f}] {r.par10:.1f}'
def a_table(stage_family, sols, stages=('main',)):
    d = c[(c.family == stage_family) & (c.stage.isin(stages))]
    rows = []
    for (n, p), q in d.groupby(['size', 'param']):
        row = [n, pp(p)]
        for s in sols:
            qq = q[q.solver == s]; row.append(fmt(qq.iloc[0]) if len(qq) else '-')
        rows.append(row)
    return table(['size', 'param'] + sols, rows)

def a_lat():
    d = c[c.stage.isin(['lat', 'long'])]; sols = ['ls_nb', 'ls_py_n0', 'ls_nb_n10', 'ls_py']
    rows = []
    for (f, n, p), q in d.groupby(['family', 'size', 'param'], sort=False):
        row = [f, n, pp(p)]
        for s in sols:
            qq = q[q.solver == s]; row.append(fmt(qq.iloc[0]) if len(qq) else '-')
        rows.append(row)
    return table(['family', 'size', 'param'] + sols, rows)
