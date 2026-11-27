"""Builds kaggle_pack/nq_v3f_cpu.ipynb from the scripts beside it (the notebook embeds them with %writefile). Rerun only BEFORE the first non-smoke run."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["nq_v3.py", "nq_v3d.py", "nq_v3e.py", "nq_v3f.py", "diag_f1.py", "analyze_v3d.py", "analyze_v3f.py", "selftest_local.py"]
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
C = [md("# nq_v3f on Kaggle CPU (internet ON for pip, 4 vCPU, session limit planned <= 6 h). Design: DESIGN_NOTE_V3F.md. Order: install, write files, self-tests, smoke, F1 diagnosis (diag_f1.py), main3f, lat3f, long3f (two arms), analyze, zip. "
        "Do not edit anything after the first non-smoke run (the F1 diagnosis counts). Nothing here was run before writing; local checks were py_compile, stubbed solve() path, fake-data analysis."),
     code("!pip -q install ortools python-sat numba numpy\nimport os; os.makedirs('/kaggle/working/v3f',exist_ok=True); os.chdir('/kaggle/working/v3f')")]
for f in FILES: C.append(code(f"%%writefile {f}\n" + open(os.path.join(HERE, f)).read()))
C += [code("!sha256sum nq_*.py diag_f1.py analyze_v3d.py analyze_v3f.py selftest_local.py   # store these hashes externally (git tag / OSF) before the diagnosis"),
      code("!python analyze_v3f.py --selftest | tail -1\n!python analyze_v3d.py --selftest | tail -1\n!python diag_f1.py --selftest\n!python selftest_local.py | tail -12"),
      code("""# smoke: stage with assertions, then a 2-rep diagnosis in a scratch dir (checks the diag mechanics), then ls_nb_fixed with whatever fix the scratch run picked
import csv, collections, subprocess, sys, os, json
r = subprocess.run([sys.executable,'nq_v3f.py','smoke3f.csv','--stage','smoke3f','--jobs','4'],capture_output=True,text=True); print(r.stdout[-1200:], r.stderr[-1200:])
R = list(csv.DictReader(open('smoke3f.csv'))); print(collections.Counter((x['solver'],x['status']) for x in R))
assert not [x for x in R if x['status'].startswith('err') or x['status']=='bad'], 'err/bad rows'
for sv in ('ls_nb','ls_nb_n10','ls_py','ls_py_n0','ls_nbk','ls_nbk_n30','ls_nbk_n65','probsat_nb','cpsat_strong','cpsat_alt','cpsat_alt_w4'): assert any(x['status']=='sat' and x['solver']==sv for x in R), 'no sat for '+sv
assert len({x['seed'] for x in R if x['family']=='bnq'}) == 2
os.makedirs('smoke_diag',exist_ok=True)
r = subprocess.run([sys.executable,'diag_f1.py','--reps','2','--spawns','1','--outdir','smoke_diag'],capture_output=True,text=True)
print(r.stdout[-2500:], r.stderr[-1500:]); assert r.returncode==0 and os.path.exists('smoke_diag/f1_fix.json') and os.path.exists('smoke_diag/f1_types.json')
rows = list(csv.DictReader(open('smoke_diag/f1_diag.csv'))); assert {x['mode'] for x in rows} >= {'parent','fork','spawn'} and len(rows) >= 20, len(rows)
env = dict(os.environ, NQ_F1_FIX='smoke_diag/f1_fix.json'); fx = json.load(open('smoke_diag/f1_fix.json')).get('fix'); print('smoke fix =', fx)
if fx:
    r = subprocess.run([sys.executable,'nq_v3f.py','smoke3f_fix.csv','--stage','smoke3f','--jobs','4','--solvers','ls_nb,ls_nb_fixed'],capture_output=True,text=True,env=env); print(r.stdout[-600:], r.stderr[-600:])
    R2 = list(csv.DictReader(open('smoke3f_fix.csv'))); assert any(x['solver']=='ls_nb_fixed' and x['status']=='sat' for x in R2), 'ls_nb_fixed no sat'
F = list(csv.DictReader(open('smoke3f.csv.first.csv'))); assert {'ls_nb','ls_nbk','probsat_nb'} <= {x['solver'] for x in F}
print('SMOKE OK (scratch files smoke*, smoke_diag/ are not part of the study; delete numba_cache/ if it was created by the smoke fix)')"""),
      code("import shutil; shutil.rmtree('numba_cache', ignore_errors=True)\nassert not os.path.exists('f1_diag.csv'), 'diagnosis already run; do not rerun'\n# F1 diagnosis (about 10 CPU minutes, serial; first non-smoke run of v3f)\n!python diag_f1.py --reps 5 --spawns 3 --outdir . 2>&1 | tail -30\n!cat f1_fix.json | head -30"),
      code("assert os.path.exists('f1_fix.json') and os.path.exists('f1_diag.csv'), 'guard: diagnosis output missing, do not start the main stage'\nprint('guard ok', json.load(open('f1_fix.json')).get('fix'))"),
      code("!python nq_v3f.py main3f.csv --stage main3f --jobs 4 --hours 2.5 2>&1 | tail -5"),
      code("!python nq_v3f.py lat3f.csv --stage lat3f --jobs 4 --hours 0.3 2>&1 | tail -3"),
      code("!python nq_v3f.py long3f.csv --stage long3f --jobs 4 --hours 0.9 --solvers ls_py_n0 2>&1 | tail -3"),
      code("!python nq_v3f.py long3f.csv --stage long3f --jobs 4 --hours 0.9 --solvers ls_py 2>&1 | tail -3"),
      code("!python analyze_v3f.py main3f.csv --lat lat3f.csv --long long3f.csv --diag f1_diag.csv --fix f1_fix.json --types f1_types.json --out analysis_v3f | head -230"),
      code("!zip -r /kaggle/working/v3f_out.zip main3f.csv main3f.csv.env.json main3f.csv.first.csv lat3f.csv lat3f.csv.env.json lat3f.csv.first.csv long3f.csv long3f.csv.env.json long3f.csv.first.csv f1_diag.csv f1_fix.json f1_types.json analysis_v3f smoke3f.csv smoke_diag *.py")]
os.makedirs(os.path.join(HERE, "kaggle_pack"), exist_ok=True)
json.dump({"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}, open(os.path.join(HERE, "kaggle_pack", "nq_v3f_cpu.ipynb"), "w"), indent=1)
print("wrote notebook with", len(C), "cells")
