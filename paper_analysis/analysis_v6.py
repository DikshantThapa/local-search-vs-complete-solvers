import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../nqueens/results_v3d'))  # recomputation of v3d numbers quoted in draft_v6.md
import pandas as pd, numpy as np
from scipy.stats import binom, kendalltau
c=pd.read_csv('confirm3d.csv'); e=pd.read_csv('ext3d.csv')
print(len(c),c.status.value_counts().to_dict(),c.killed.sum(), len(e), e.status.value_counts().to_dict(), e.killed.sum())
c['solved']=((c.status=='sat')&(c.verified==1)).astype(int); e['solved']=((e.status=='sat')&(e.verified==1)).astype(int)
print('sat unverified',((c.status=='sat')&(c.verified!=1)).sum(),((e.status=='sat')&(e.verified!=1)).sum())
print(c.groupby(['family','size']).seed.agg(['min','max']))
print(c.groupby(['family','solver']).size().unstack())
cell=c.groupby(['family','size','param','solver']).agg(n=('solved','size'),s=('solved','sum')).reset_index()
print(cell.n.value_counts())
def w(k,n,z=1.96):
    p=k/n;d=1+z*z/n;ce=(p+z*z/2/n)/d;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/d;return round(ce-h,2),round(ce+h,2)
def pv(fam,size,par):
    d=c[(c.family==fam)&(c['size']==size)&(c.param==par)]
    return d.pivot(index='seed',columns='solver',values='solved')
def mcn(a,b):
    x=int(((a==1)&(b==0)).sum()); y=int(((a==0)&(b==1)).sum()); n=x+y
    return x,y,(binom.sf(x-1,n,.5) if n else 1.0)
print('D1')
for p in [.84,.86,.88,.9]:
    t=pv('bnq',200,p); print(p,t.ls_nb.sum(),t.cpsat_tuned.sum(),t.cpsat_strong.sum(),t.cadical195.sum(),len(t))
print('D2')
for r in [4.2,4.26,4.3]:
    t=pv('ksat',350,r); print(r,{s:int(t[s].sum()) for s in t},len(t),mcn(t.probsat_nb,t.cadical195))
print('D3')
for r in [20,20.5,21]:
    t=pv('ksat5',200,r); print(r,{s:int(t[s].sum()) for s in t},len(t),mcn(t.probsat_nb,t.cadical195),mcn(t.probsat_nb,t.cpsat_tuned))
print('D4')
for h in [.35,.4]:
    t=pv('qwh',40,h); print(h,{s:int(t[s].sum()) for s in t},len(t))
    bc=t[['cpsat_strong','cpsat_tuned','cadical195']].sum().idxmax(); bl=t[['ls_nb','ls_tabu_nb']].sum().idxmax()
    print(bc,bl,mcn(t[bc],t[bl]))
for f in [.38,.42,.46]:
    t=pv('qcp',40,f); print('qcp',f,{s:int(t[s].sum()) for s in t},len(t))

# ---- part 2

import pandas as pd, numpy as np, math, json, itertools
from scipy.stats import kendalltau

c=pd.read_csv('confirm3d.csv'); e=pd.read_csv('ext3d.csv')
for d in (c,e): d['solved']=((d.status=='sat')&(d.verified==1)).astype(int)
def w(k,n,z=1.96):
    p=k/n;d=1+z*z/n;ce=(p+z*z/2/n)/d;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/d;return f"[{max(0,ce-h):.2f},{min(1,ce+h):.2f}]"
print('Wilson 20/20',w(20,20),'0/20',w(0,20))
# D1 bootstrap
def boot(d):
    rg=np.random.default_rng(0);mu=d[rg.integers(0,len(d),(10000,len(d)))].mean(1);return np.percentile(mu,[2.5,97.5])
def pv(fam,size,par):
    d=c[(c.family==fam)&(c['size']==size)&(np.isclose(c.param.astype(float),par))]
    return d.pivot(index='seed',columns='solver',values='solved')
for p in [.84,.86,.88,.9]:
    t=pv('bnq',200,p);print(p,boot((t.ls_nb-t.cpsat_tuned).values),w(int(t.ls_nb.sum()),20),w(int(t.cpsat_tuned.sum()),20))
# Wilson for D2-D4 cells
for fam,size,pars in [('ksat',350,[4.2,4.26,4.3]),('ksat5',200,[20,20.5,21]),('qwh',40,[.35,.4])]:
    for p in pars:
        t=pv(fam,size,p);print(fam,size,p,{s:f"{int(t[s].sum())}/20 {w(int(t[s].sum()),20)}" for s in t})
# Holm
raw={'D2':0.0078125,'D3':9.5367431640625e-07,'D4':1.0}
o=sorted(raw,key=raw.get);run=0
for r,k in enumerate(o): run=max(run,min(1,(3-r)*raw[k]));print(k,raw[k],run)
# D2 and D3: raw p second smallest check
print('D2 ps',sorted([0.00390625,0.015625,0.0078125]))
# D6
fm={}
cells=c.groupby(['family','size','param'])
taus={}
for k,g in cells:
    f=g.groupby('solver').solved.mean()
    if 'cpsat_tuned' not in f or 'cpsat_strong' not in f: continue
    base=[s for s in f.index if s not in('cpsat_strong','cpsat_tuned')]
    x=[f[s] for s in base]+[f['cpsat_strong']];y=[f[s] for s in base]+[f['cpsat_tuned']]
    t=kendalltau(x,y,variant='b')[0]
    taus.setdefault(k[0],[]).append(t)
allt=[]
for fam,l in taus.items():
    ok=[t for t in l if not np.isnan(t)];allt+=ok;print(fam,len(l),len(ok),round(np.mean(ok),3),[round(t,2) for t in ok if t<0.999])
print('all',len(allt),np.mean(allt))
# D5
for fam,g in e.groupby('family'):
    n=g.param.nunique()
    for s,h in g.groupby('solver'):
        k=int(h.solved.sum());print(fam,s,len(h),k,n,w(k,n),h.status.value_counts().to_dict(),'median t solved',h[h.solved==1].t.median())
print(e.groupby(['family','solver']).killed.sum()[lambda x:x>0])
print(e[(e.family=='ext_qg')].pivot(index='param',columns='solver',values='status'))
print(e.seed.unique(),e['size'].unique(),e['T'].unique())

# ---- part 3

import pandas as pd, numpy as np
c=pd.read_csv('confirm3d.csv');t=pd.read_csv('tune_log.csv')
c['solved']=((c.status=='sat')&(c.verified==1)).astype(int)
print(t.groupby(['part','family']).size())
print(t.part.value_counts(), t.setting.unique())
print(t[t.part!='cp'].head(3)); print(t[t.part!='cp'].groupby(['family','size','param']).size())
print(c.groupby('solver').agg(n=('solved','size'),kill=('killed','sum')))
c['ex']=c.t_raw-c['T']
print(c[c.t_raw.notna()].groupby('solver').ex.agg(['max',lambda x:(x>0.05).sum()]))
print(c[c.killed==1].status.value_counts())
b=c[(c.family=='bnq')&(c.solver=='ls_nb')&(c.solved==1)]
print('bnq ls_nb solved',len(b),b.t.min(),b.t.median(),b.groupby('size').t.agg(['size','min','median']))
print(b.groupby(['size','param']).size().unstack())
# other cells
piv=c.groupby(['family','size','param','solver']).solved.sum().unstack()
pd.set_option('display.width',250,'display.max_rows',500)
print(piv)
# ls_nb vs probsat on ksat identical?
k=c[c.family=='ksat'].pivot_table(index=['size','param','seed'],columns='solver',values='solved')
print((k.ls_nb!=k.probsat_nb).sum(), len(k))
tt=c[(c.family=='ksat')&c.solver.isin(['ls_nb','probsat_nb'])&(c.solved==1)].groupby(['solver','size']).t.median();print(tt)
