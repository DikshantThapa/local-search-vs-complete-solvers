import pandas as pd, numpy as np
R='/Users/dikshantthapa/Desktop/Claude-projects/Research/nqueens/results_v3f/'
m=pd.read_csv(R+'main3f.csv'); l=pd.read_csv(R+'lat3f.csv'); g=pd.read_csv(R+'long3f.csv')
print(m[m.killed==1].groupby('solver').size().to_dict())
m['late']=(m.t_raw>m['T'])
print(m[m.late].groupby('solver').size().to_dict())
print('max excess by solver',(m.t_raw-m['T']).groupby(m.solver).max().round(3).to_dict())
print('max excess long',(g.t_raw-g['T']).groupby(g.solver).max().round(3).to_dict(),(l.t_raw-l['T']).groupby(l.solver).max().round(3).to_dict())
print('late>0.05',((m.t_raw-m['T'])>0.05).groupby(m.solver).sum()[lambda x:x>0].to_dict())
# diag
d=pd.read_csv(R+'f1_diag.csv'); d['first']=d.t_seed+d.t_init1+d.t_steps1
d['second']=d.t_init2+d.t_steps2
print(d.groupby(['cond','mode']).agg(n=('first','size'),med=('first','median'),mn=('first','min'),mx=('first','max'),sec=('second','median'),ce=('compile_events','median')).round(4))
print(d.cond.unique())
# cells compare
c=pd.read_csv(R+'analysis_v3f/cells.csv'); print(c.stage.unique(), len(c))
bad=0
for _,r in c.iterrows():
    src={'main':m,'lat':l,'long':g}.get(r.stage)
    if src is None: print('skip',r.stage); break
    q=src[(src.family==r.family)&(src['size']==r['size'])&(src.param==r.param)&(src.solver==r.solver)]
    s=int(((q.status=='sat')&(q.verified==1)).sum())
    if len(q)!=r.n or s!=r.solved:
        bad+=1; print('MISMATCH',dict(r))
print('cells compared',len(c),'mismatches',bad)
# par10
def par10(q): return ((q.t.where(q.status=='sat',10*q['T']))).mean()
mm=0
for _,r in c.iterrows():
    src={'main':m,'lat':l,'long':g}[r.stage]
    q=src[(src.family==r.family)&(src['size']==r['size'])&(src.param==r.param)&(src.solver==r.solver)]
    s=((q.status=='sat')&(q.verified==1))
    p=np.where(s,q.t,10*q['T']).mean()
    if abs(p-r.par10)>0.01: mm+=1; print('par10 diff',r.stage,r.family,r['size'],r.param,r.solver,round(p,2),r.par10)
print('par10 mismatches',mm)
