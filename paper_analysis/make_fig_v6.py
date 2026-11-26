import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
c=pd.read_csv('../../nqueens/results_v3d/confirm3d.csv'); c['s']=((c.status=='sat')&(c.verified==1)).astype(int)
def w(k,n,z=1.96):
    p=k/n;d=1+z*z/n;ce=(p+z*z/2/n)/d;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/d;return ce-h,ce+h
fig,ax=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
st={'probsat_nb':('probSAT','#1b6ca8','o','-'),'cadical195':('CaDiCaL','#d95f02','s','--'),'cpsat_tuned':('CP-SAT (tuned)','#555555','^',':')}
for a,(fam,n,title) in zip(ax,[('ksat',350,'3-SAT, n = 350'),('ksat5',200,'5-SAT, n = 200')]):
    d=c[(c.family==fam)&(c['size']==n)]
    for s,(lab,col,m,ls) in st.items():
        g=d[d.solver==s].groupby('param').s.sum(); x=g.index.values; k=g.values
        lo=[w(v,20)[0] for v in k]; hi=[w(v,20)[1] for v in k]
        a.errorbar(x,k/20,yerr=[np.clip(k/20-np.array(lo),0,None),np.clip(np.array(hi)-k/20,0,None)],label=lab,color=col,marker=m,ls=ls,capsize=2,lw=1.3,ms=5)
    a.axvline(4.26 if fam=='ksat' else 21.1,color='k',lw=.6,alpha=.4); a.set_title(title); a.set_xlabel('clause-to-variable ratio r'); a.grid(alpha=.25)
ax[0].set_ylabel('fraction solved in 10 s (20 instances)'); ax[0].legend(frameon=False,loc='lower left')
plt.tight_layout(); plt.savefig('figs_v6/fig3_v3d_sat.png',dpi=200)
