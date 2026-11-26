from analysis_v5 import *
import os
O='appendix_tables/';os.makedirs(O,exist_ok=True)
names={'main10':'A','confirm':'B','long60':'C'}
cap={'main10':'main10 (T = 10 s, 10 instances per cell, seeds 0-9)','confirm':'confirmation stage (T = 10 s, 20 fresh instances per cell)','long60':'long60 (T = 60 s, 5 instances per cell, seeds 0-4)'}
famname={'bnq':'blocked n-Queens (size n, blocking probability p)','qwh':'quasigroup with holes (N, hole fraction h)','qcp':'quasigroup completion (N, preassigned fraction f)','ksat':'uniform random 3-SAT (n variables, clause ratio r)'}
md=[];k=0
for s,d in [(k,D[k]) for k in ['main10','confirm','long60']]:
    sols=[x for x in ['ls_nb','ls_nb_w4','ls_py','cpsat_naive','cpsat_strong','cpsat_strong_w4','glucose4','cadical195'] if x in set(d.solver)]
    rows=[]
    for (f,n,p,sv),g in d.groupby(['family','size','param','solver']):
        k_=int(g.solved.sum());lo,hi=wil(k_,len(g));lo=max(lo,0.0);hi=min(hi,1.0);rows.append(dict(stage=s,family=f,size=n,param=p,solver=sv,n=len(g),solved=k_,frac=round(k_/len(g),4),wilson_lo=round(lo,4),wilson_hi=round(hi,4),par10=round(g.par10.mean(),3),decided=int(g.decided.sum())))
    c=pd.DataFrame(rows);c.to_csv(f'{O}cells_{s}.csv',index=False)
    for i,fam in enumerate(['bnq','qwh','qcp','ksat'],1):
        k+=1;g=c[c.family==fam]
        md.append(f'Table A{k}. Per-cell results, {fam}, {cap[s]}: solved/n, Wilson 95 % interval, PAR10 in seconds (PAR10 = mean of solve time, or 10 T if unsolved). Family: {famname[fam]}.\n')
        md.append('| size | param | '+' | '.join(sols)+' |');md.append('|---|---|'+'---|'*len(sols))
        for (n,p),h in g.groupby(['size','param']):
            cells=[]
            for sv in sols:
                r=h[h.solver==sv]
                if r.empty:cells.append('-');continue
                r=r.iloc[0];cells.append(f'{r.solved}/{r.n} [{r.wilson_lo:.2f}, {r.wilson_hi:.2f}] {r.par10:.1f}')
            md.append(f'| {n} | {p:g} | '+' | '.join(cells)+' |')
        md.append('')
open(O+'appendix_tables.md','w').write('\n'.join(md));print(k,'tables',len(md))
