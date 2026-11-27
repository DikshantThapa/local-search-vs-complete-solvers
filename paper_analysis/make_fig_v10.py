import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='../../nqueens/results_v3f/'
l=pd.read_csv(R+'lat3f.csv'); g=pd.read_csv(R+'long3f.csv'); d=pd.concat([l,g])
d['s']=((d.status=='sat')&(d.verified==1)).astype(int)
def w(k,n,z=1.96):
    p=k/n;dd=1+z*z/n;ce=(p+z*z/2/n)/dd;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/dd;return ce-h,ce+h
arms=[('ls_nb','compiled, noise 0, 10 s','#1b6ca8'),('ls_py_n0','interpreted, noise 0, 100 s','#6baed6'),('ls_nb_n10','compiled, noise 0.1, 10 s','#d95f02'),('ls_py','interpreted, noise 0.1, 100 s','#fdae6b')]
fig,(a,b)=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1,1.6]})
for i,(s,lab,c) in enumerate(arms):
    k=int(d[d.solver==s].s.sum()); n=int((d.solver==s).sum()); lo,hi=w(k,n)
    a.bar(i,k/n,color=c,yerr=[[max(0,k/n-lo)],[max(0,hi-k/n)]],capsize=3); a.text(i,hi+.02,f'{k}/{n}',ha='center',fontsize=8)
a.set_xticks(range(4)); a.set_xticklabels(['fast\nnoise 0','slow\nnoise 0','fast\nnoise 0.1','slow\nnoise 0.1'],fontsize=8)
a.set_ylim(0,1.12); a.set_ylabel('fraction solved'); a.set_title('all 16 cells x 5 seeds (80 instances)',fontsize=9); a.grid(alpha=.25,axis='y')
groups=[('qcp',30),('qcp',40),('qwh',30),('qwh',40)]
for j,(f,N) in enumerate(groups):
    for i,(s,lab,c) in enumerate(arms):
        q=d[(d.solver==s)&(d.family==f)&(d['size']==N)]; k=int(q.s.sum()); n=len(q); lo,hi=w(k,n)
        b.bar(j+(i-1.5)*.2,k/n,.2,color=c,yerr=[[max(0,k/n-lo)],[max(0,hi-k/n)]],capsize=2,label=lab if j==0 else None)
b.set_xticks(range(4)); b.set_xticklabels([f'{f} N={N}' for f,N in groups],fontsize=8); b.set_ylim(0,1.3); b.set_title('by family and size (20 instances each)',fontsize=9); b.grid(alpha=.25,axis='y')
b.legend(frameon=False,fontsize=7,loc='upper center',ncol=2)
plt.tight_layout(); plt.savefig('figs_v10/fig4_noise_speed_2x2.png',dpi=200)
