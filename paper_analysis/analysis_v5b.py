import numpy as np, pandas as pd
from analysis_v5 import *
COMPLETE=['cpsat_naive','cpsat_strong','cpsat_strong_w4','glucose4','cadical195']
def fr(d,fam,n,p,sv,col='solved'):
    g=d[sel(d,fam,n,p,sv)][col];return g.mean() if len(g) else np.nan
def best(d,fam,n,p):
    v={s:fr(d,fam,n,p,s) for s in COMPLETE};v={k:x for k,x in v.items() if not np.isnan(x)};return max(v.values())
def h8(d,lab,fam,ratio=.8,share=.75,allcells=False):
    cs=lab[(lab.family==fam)&lab['size'].isin([30,40])]
    if not allcells:cs=cs[cs.side=='SAT']
    hit=[fr(d,fam,r['size'],r.param,'ls_nb')>=ratio*best(d,fam,r['size'],r.param)-1e-12 for _,r in cs.iterrows()]
    return sum(hit),len(hit),(np.mean(hit)>=share if hit else None)
def boot(z,seed):
    rng=np.random.default_rng(seed);i=rng.integers(0,len(z),(10000,len(z)));return np.percentile(z[i].mean(1),[2.5,97.5])
def paired_lo(d,fam,n,p,a,b,seed):
    m=sel(d,fam,n,p);x=d[m&(d.solver==a)].set_index(KEY).solved.astype(float);y=d[m&(d.solver==b)].set_index(KEY).solved.astype(float).reindex(x.index);return boot((x-y).values,seed)[0]
if __name__=='__main__':
    for s in ['main10','confirm','long60']:
        d=D[s];lab=labels(d)
        print('==',s,'label counts',lab.side.value_counts().to_dict())
        for fam in['qwh','qcp']:
            print(fam,'SAT-side',h8(d,lab,fam),'all cells',h8(d,lab,fam,allcells=True))
            if s!='long60':
                c=lab[(lab.family==fam)&(lab['size']==40)];print('  N=40 labels',c[['param','side']].values.tolist())
        # N=40 pooled
        for fam in ['qwh','qcp']:
            for sv in ['ls_nb','cadical195']:
                g=d[sel(d,fam,40,None,sv)];print(' N40 pooled',fam,sv,int(g.solved.sum()),len(g))
        # failing cells under SAT-side
        for fam in['qwh','qcp']:
            for _,r in lab[(lab.family==fam)&lab['size'].isin([30,40])&(lab.side=='SAT')].iterrows():
                l=fr(d,fam,r['size'],r.param,'ls_nb');b=best(d,fam,r['size'],r.param)
                if l<.8*b:print('  FAIL',fam,r['size'],r.param,round(l,2),round(b,2))
    # threshold sensitivity
    print('\n== SENSITIVITY')
    for s in ['main10','confirm','long60']:
        d=D[s];lab=labels(d)
        for ratio in [.4,.8,1.0]:
            for share in [.375,.75,1.0]:
                print(s,'H8/C3 ratio',ratio,'share',share,{f:h8(d,lab,f,ratio,share)[2] for f in['qwh','qcp']})
    d=D['confirm'];lab=labels(d)
    for thr in [.25,.5,1.0]:
        print('C1 lower bound >=',thr,[ (p,round(paired_lo(d,'bnq',200,p,'ls_nb','cpsat_strong',0),3)) for p in(.84,.86,.88,.90)])
    d=D['main10']
    print('H1 lbs',[ (p,round(paired_lo(d,'bnq',200,p,'ls_nb','cpsat_strong',12345),3)) for p in(.84,.86,.88,.90)])
    # C2
    d=D['confirm']
    k=[(p,fr(d,'bnq',200,p,'ls_nb_w4'),fr(d,'bnq',200,p,'cpsat_strong_w4')) for p in sorted(d[sel(d,'bnq',200)].param.unique())]
    print('C2',k)
    for thr in[.25,.5,1.0]:
        c=[x for x in k if x[1]==1.0];print('C2 thr',thr,len(c),sum(x[2]<thr for x in c))
    # C4 gaps
    for s in ['main10','confirm']:
        d=D[s];lab=labels(d);g=[]
        for _,r in lab[(lab.family=='qcp')&(lab['size']==40)&(lab.side=='SAT')].iterrows():
            g.append((r.param,round(best(d,'qcp',40,r.param)-fr(d,'qcp',40,r.param,'ls_nb'),2)))
        print('C4 gaps',s,g,'max',max(x[1] for x in g))
        allg=[(p,round(best(d,'qcp',40,p)-fr(d,'qcp',40,p,'ls_nb'),2)) for p in sorted(d[sel(d,'qcp',40)].param.unique())];print('  all cells',allg)
    # H6 long60
    d=D['long60']
    print('H6',[(p,fr(d,'bnq',200,p,'ls_nb'),fr(d,'bnq',200,p,'cpsat_strong'),fr(d,'bnq',200,p,'cpsat_strong_w4')) for p in(.84,.88,.92)])
    print('long60 w4 times p=.84',d[sel(d,'bnq',200,.84,'cpsat_strong_w4')&d.solved].t.median(), 'n=100 .88 w4',d[sel(d,'bnq',100,.88,'cpsat_strong_w4')].solved.sum(),d[sel(d,'bnq',100,.88,'cpsat_strong_w4')&d.solved].t.median(),'1w',d[sel(d,'bnq',100,.88,'cpsat_strong')].solved.sum(),d[sel(d,'bnq',100,.88,'cpsat_strong')&d.solved].t.median())
    # C5 first part and ls_nb in confirm
    d=D['confirm'];print('C5 r=4.0 ls',fr(d,'ksat',350,4.0,'ls_nb'),'; cad decided 4.26,4.3',fr(d,'ksat',350,4.26,'cadical195','decided'),fr(d,'ksat',350,4.3,'cadical195','decided'))
    # H4 sensitivity (main10)
    d=D['main10']
    for r in[4.4,4.6]:
        print('H4 ksat250',r,fr(d,'ksat',250,r,'cadical195','decided'),fr(d,'ksat',250,r,'cpsat_strong','decided'))
    # Table3 selected cells recheck
    d=D['confirm']
    for fam,n,p in[('bnq',50,.84),('bnq',100,.88),('bnq',100,.9),('bnq',200,.84),('bnq',200,.92),('bnq',200,.94),('qwh',40,.3),('qwh',40,.35),('qwh',40,.4),('qwh',40,.45),('qwh',40,.6),('qcp',40,.3),('qcp',40,.42),('qcp',40,.46),('qcp',40,.5),('ksat',350,4.1),('ksat',350,4.2),('ksat',350,4.26),('ksat',350,4.3),('ksat',350,4.4)]:
        print(fam,n,p,[round(fr(d,fam,n,p,s),2) for s in['ls_nb','ls_nb_w4','cpsat_strong','cpsat_strong_w4','cadical195']])
