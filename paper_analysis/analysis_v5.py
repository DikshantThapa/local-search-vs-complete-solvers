import numpy as np, pandas as pd
from scipy.stats import binomtest
R='/Users/dikshantthapa/Desktop/Claude-projects/Research/nqueens/'
F={'main10':R+'results_v3/main10.csv','long60':R+'results_v3/long60.csv','confirm':R+'results_v3c/confirm.csv'}
KEY=['family','size','param','seed']
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
D={}
for s,f in F.items():
    d=pd.read_csv(f);d['solved']=(d.status=='sat')&(d.verified==1);d['decided']=d.solved|(d.status=='unsat')
    d['par10']=np.where(d.solved,d.t,10*d['T']);d['stage']=s;D[s]=d
def sel(d,fam=None,n=None,p=None,sv=None):
    m=pd.Series(True,index=d.index)
    if fam:m&=d.family==fam
    if n:m&=d['size']==n
    if p is not None:m&=np.isclose(d.param,p)
    if sv:m&=d.solver==sv
    return m
def labels(d):
    k=d.groupby(KEY).agg(sat=('solved','any'),uns=('status',lambda s:(s=='unsat').any())).reset_index()
    c=k.groupby(['family','size','param']).agg(sat=('sat','mean'),uns=('uns','mean')).reset_index()
    c['side']=np.where(c.sat>=.9,'SAT',np.where(c.uns>=.9,'UNSAT','transition'));return c
if __name__=='__main__':
    for s,d in D.items():
        print('==',s,len(d),d.status.value_counts().to_dict(),'bad/err',int((~d.status.isin(['sat','unsat','to'])).sum()),'unverified sat',int(((d.status=='sat')&(d.verified!=1)).sum()))
        print('killed',int(d.killed.sum()),d.groupby('solver').killed.sum().to_dict())
        o=d[d.t_raw>d['T']];print('overshoot rows (t_raw>T)',len(o),o.groupby('solver').size().to_dict(),'max excess by solver',(o.t_raw-o['T']).groupby(o.solver).max().round(2).to_dict(),'max t_raw',d.t_raw.max())
        print('status of overshoot',o.status.value_counts().to_dict(), 'status in killed',d[d.killed==1].status.value_counts().to_dict())
        print('latest sat by solver (t_raw over T excluded):');
        x=d[(d.t_raw>d['T'])];
        print('t_raw>T+0.02 ls_nb:',int(((d.solver=='ls_nb')&(d.t_raw>d['T']+.02)).sum()))
    print('\n== bnq ls_nb fixed cost')
    for s in ['main10','confirm','long60']:
        d=D[s];g=d[(d.family=='bnq')&d.solved&(d.solver=='ls_nb')]
        print(s,'n',len(g),'min',g.t.min(),'median',g.t.median(),g.groupby('size').t.agg(['count','min','median']).round(4).to_dict('index'))
        for fam in ['qwh','qcp','ksat']:
            h=d[(d.family==fam)&d.solved&(d.solver=='ls_nb')];print('  ',fam,'min',h.t.min(),'median',h.t.median())
        for sv in ['ls_py','ls_nb_w4']:
            h=d[(d.family=='bnq')&d.solved&(d.solver==sv)]
            if len(h):print('  bnq',sv,'n',len(h),'min',h.t.min(),'median',h.t.median())
    # ls_py/ls_nb ratio
    d=D['main10'];x=d[d.solved&d.solver.isin(['ls_py','ls_nb'])].pivot_table(index=KEY,columns='solver',values='t').dropna()
    x['fam']=x.index.get_level_values('family');x['r']=x.ls_py/x.ls_nb
    print('ratio all n',len(x),'median',x.r.median(),x.groupby('fam').r.agg(['count','median']).round(3).to_dict('index'))
    xs=x[~((x.fam=='bnq'))];print('excl bnq',len(xs),xs.r.median())
    x2=x[(x.fam!='bnq')|(x.ls_nb>1.0)];
    print('\n== C5 paired ls_nb vs cadical195, ksat n=350')
    def pair(d,m):
        a=d[m&(d.solver=='ls_nb')].set_index(KEY).solved;b=d[m&(d.solver=='cadical195')].set_index(KEY).solved.reindex(a.index)
        w=int((a&~b).sum());l=int((~a&b).sum());both=int((a&b).sum());nei=int((~a&~b).sum())
        p=binomtest(w,w+l,.5).pvalue if w+l else 1.0;return w,l,both,nei,p
    pool={}
    for s in ['main10','confirm']:
        for r in [4.2,4.26,4.3]:
            m=sel(D[s],'ksat',350,r);w,l,b,n,p=pair(D[s],m);print(s,r,'ls_only',w,'cdcl_only',l,'both',b,'neither',n,'p',round(p,6),'ls',int(D[s][m&(D[s].solver=='ls_nb')].solved.sum()),'cad solved',int(D[s][m&(D[s].solver=='cadical195')].solved.sum()),'cad decided',int(D[s][m&(D[s].solver=='cadical195')].decided.sum()),'cad unsat',int((D[s][m&(D[s].solver=='cadical195')].status=='unsat').sum()))
    def poolp(parts):
        W=L=0;N=0
        for s,rs in parts:
            for r in rs:
                m=sel(D[s],'ksat',350,r);w,l,b,n,p=pair(D[s],m);W+=w;L+=l;N+=w+l+b+n
        return W,L,N,binomtest(W,W+L,.5).pvalue
    print('pooled confirm r in 4.2,4.26,4.3',poolp([('confirm',[4.2,4.26,4.3])]))
    print('pooled main10',poolp([('main10',[4.2,4.26,4.3])]))
    print('pooled both',poolp([('main10',[4.2,4.26,4.3]),('confirm',[4.2,4.26,4.3])]))
    print('pooled both only 4.26 4.3',poolp([('main10',[4.26,4.3]),('confirm',[4.26,4.3])]))
    print('pooled confirm only 4.26 4.3',poolp([('confirm',[4.26,4.3])]))
    print('pooled both 4.1',poolp([('main10',[4.1]),('confirm',[4.1])]))
    print('\n== qwh cells')
    for s in ['main10','confirm','long60']:
        d=D[s];print(s,'solvers',sorted(d.solver.unique()))
        for N,hs in [(30,[.35]),(40,[.30,.35,.40])]:
            for h in hs:
                m=sel(d,'qwh',N,h);row={sv:int(d[m&(d.solver==sv)].solved.sum()) for sv in sorted(d.solver.unique()) if (m&(d.solver==sv)).any()};n=int((m&(d.solver=='ls_nb')).sum());print(' qwh',N,h,'n',n,row)
