import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='../../nqueens/results_v3e/'
d=pd.read_csv(R+'confirm3e.csv'); l=pd.read_csv(R+'long3e.csv')
for x in (d,l): x['s']=((x.status=='sat')&(x.verified==1)).astype(int)
def w(k,n,z=1.96):
    p=k/n;dd=1+z*z/n;ce=(p+z*z/2/n)/dd;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/dd;return ce-h,ce+h
q=d[d.family.isin(['qwh','qcp'])]
cells=[(f,n,p) for f,ps in (('qwh',[.30,.35,.40,.45]),('qcp',[.38,.42,.46,.50])) for n in (30,40) for p in ps]
fig,ax=plt.subplots(figsize=(9,3.6)); xs=np.arange(len(cells))
for sv,lab,col,m,off,n in (('ls_nb','ls_nb (noise 0), 10 s, 20 inst.','#1b6ca8','o',-.2,20),('ls_nb_n10','ls_nb (noise 0.1), 10 s, 20 inst.','#d95f02','s',0,20),('ls_py','ls_py (noise 0.1), 100 s, 5 inst.','#555555','^',.2,5)):
    src=l if sv=='ls_py' else q
    k=np.array([src[(src.family==f)&(src['size']==s)&(src.param==p)&(src.solver==sv)].s.sum() for f,s,p in cells])
    ci=np.array([w(v,n) for v in k]); fr=k/n
    ax.errorbar(xs+off,fr,yerr=[np.clip(fr-ci[:,0],0,None),np.clip(ci[:,1]-fr,0,None)],fmt=m,color=col,label=lab,capsize=2,ms=5,lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"{f} {n}\n{p}" for f,n,p in cells],fontsize=7); ax.set_ylabel('fraction solved'); ax.set_ylim(-.05,1.08); ax.grid(alpha=.25,axis='y')
ax.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3); ax.set_xlabel('family, N, parameter (h for qwh, f for qcp)')
plt.tight_layout(); plt.savefig('figs_v8/fig4_noise_ablation.png',dpi=200)
