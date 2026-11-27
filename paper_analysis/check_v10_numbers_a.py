import pandas as pd, numpy as np
from scipy.stats import binomtest
R='/Users/dikshantthapa/Desktop/Claude-projects/Research/nqueens/results_v3f/'
m=pd.read_csv(R+'main3f.csv'); l=pd.read_csv(R+'lat3f.csv'); g=pd.read_csv(R+'long3f.csv')
for x in (m,l,g): x['s']=((x.status=='sat')&(x.verified==1)).astype(int)
print('rows',len(m),len(l),len(g),m.status.value_counts().to_dict(),'killed',m.killed.sum(),l.killed.sum(),g.killed.sum())
print('sat verified0',((m.status=='sat')&(m.verified!=1)).sum(),((l.status=='sat')&(l.verified!=1)).sum(),((g.status=='sat')&(g.verified!=1)).sum())
print('late', (m.t_raw>m['T']).sum(), ((m.t_raw>m['T'])&(m.status=='sat')).sum(), (g.t_raw>g['T']).sum())
def w(k,n,z=1.96):
    p=k/n;dd=1+z*z/n;ce=(p+z*z/2/n)/dd;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/dd;return round(ce-h,2),round(ce+h,2)
def cell(d,**kw):
    q=d
    for k,v in kw.items(): q=q[q[k]==v]
    return q
print('--bnq')
b=m[m.family=='bnq']
for n in (100,200):
  for p in (.84,.88,.9,.92):
    row=[]
    for s in ['ls_nb','ls_nb_fixed','ls_py','cpsat_strong','cpsat_alt','cpsat_strong_w4','cpsat_alt_w4']:
        q=b[(b['size']==n)&(b.param==p)&(b.solver==s)]
        if len(q): row.append(f"{s} {q.s.sum()}/{len(q)}")
    print(n,p,row)
# times
for s in ['ls_nb','ls_nb_fixed','ls_py']:
  q=b[(b.solver==s)&(b.s==1)&(b['size']==200)]
  print(s,'n200 solved',len(q),'min',q.t.min(),'median',q.t.median())
  for p in (.84,.88,.9,.92):
    qq=q[q.param==p]; print('  ',p,len(qq),round(qq.t.min(),4),round(qq.t.median(),4))
  q4=q[q.param.isin([.84,.88,.9,.92])]
  print('  pooled4 min',q4.t.min(),'median',q4.t.median())
for s in ['ls_nb','ls_nb_fixed','ls_py']:
  for n in (100,200):
    for p in (.84,.88,.92):
      qq=b[(b.solver==s)&(b.s==1)&(b['size']==n)&(b.param==p)]
      if len(qq): print(s,n,p,'med solved',round(qq.t.median(),3),'min',round(qq.t.min(),4))
# paired nb vs fixed
def paired(d,a,bb,keys=['family','size','param','seed']):
    A=d[d.solver==a].set_index(keys).s; B=d[d.solver==bb].set_index(keys).s
    j=pd.concat([A,B],axis=1,keys=['a','b']).dropna()
    oa=((j.a==1)&(j.b==0)).sum(); ob=((j.a==0)&(j.b==1)).sum()
    p=binomtest(min(oa,ob),oa+ob,.5).pvalue if oa+ob else 1.0
    return int(j.a.sum()),int(j.b.sum()),len(j),int(oa),int(ob),round(p,4)
for n in (100,200):
  for p in (.84,.88,.92):
    q=b[(b['size']==n)&(b.param==p)]
    print('alt vs strong',n,p,paired(q,'cpsat_alt','cpsat_strong'),'w4',paired(q,'cpsat_alt_w4','cpsat_strong_w4'))
# first-call
f=pd.read_csv(R+'main3f.csv.first.csv'); fb=f[f.family=='bnq']
print(fb.groupby('solver').first_s.agg(['count','median','max']))
print(f.groupby(['family','solver']).first_s.agg(['count','median','max']).round(4))
print('--ksat5')
k=m[m.family=='ksat5']
for r in (12,14,16,18,20,20.5):
  row=[]
  for s in ['ls_nbk','ls_nbk_n30','ls_nbk_n40','ls_nbk_n50','ls_nbk_n65','probsat_nb','cadical195']:
    q=k[(k.param==r)&(k.solver==s)]
    if len(q): row.append(f"{s} {q.s.sum()}/{len(q)} {w(q.s.sum(),len(q))}")
  print(r,row)
print(k['size'].unique(), 'unsat rows',(m.status=='unsat').groupby(m.solver).sum().to_dict())
print('--lat')
L=pd.concat([l,g]); keys=['family','size','param','seed']
tot=L.groupby('solver').s.agg(['sum','count']); print(tot)
w_=L.pivot_table(index=keys,columns='solver',values='s',aggfunc='first').dropna()
print(len(w_))
def mc(a,b_):
    oa=int(((w_[a]==1)&(w_[b_]==0)).sum()); ob=int(((w_[a]==0)&(w_[b_]==1)).sum())
    p=min(1,2*binomtest(min(oa,ob),oa+ob,.5).pvalue*0+ (binomtest(min(oa,ob),oa+ob,.5).pvalue if oa+ob else 1))
    return oa,ob,round(binomtest(min(oa,ob),oa+ob,.5).pvalue,4) if oa+ob else 1
for a,b_ in [('ls_nb','ls_nb_n10'),('ls_py_n0','ls_py'),('ls_nb','ls_py_n0'),('ls_nb_n10','ls_py'),('ls_nb','ls_py'),('ls_nb_n10','ls_py_n0')]:
    print(a,b_,int(w_[a].sum()),int(w_[b_].sum()),mc(a,b_))
for s in ['ls_nb','ls_nb_n10','ls_py_n0','ls_py']: print(s,int(w_[s].sum()),w(int(w_[s].sum()),80))
# 2x2 by family/size
for (fa,sz),q in w_.groupby(level=['family','size']):
    print(fa,sz,len(q),{s:int(q[s].sum()) for s in w_.columns})
# solved in union sets
print('nb solved & py_n0 not', ((w_.ls_nb==1)&(w_.ls_py_n0==0)).sum())
print('py_n0 solved times median', g[(g.solver=='ls_py_n0')&(g.s==1)].t.median(), g[(g.solver=='ls_py')&(g.s==1)].t.median())
print('lat to times', L.groupby('solver').t.max())
