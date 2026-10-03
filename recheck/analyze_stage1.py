import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from scipy import stats
rows=[json.loads(l) for l in open('p9_recheck/results/cells_recheck.jsonl')]
df=pd.DataFrame(rows); print(len(df),'cells; seeds',sorted(df.seed.unique()))
band=['awgn@+0dB','awgn@-2dB','awgn@-6dB']
# reproduction vs original
B='/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627/results/'
orig=pd.concat([pd.DataFrame([json.loads(l) for l in open(B+p+'/cells.jsonl')]) for p in ['secondary_20260703-1034','graft_20260705-224137']])
orig=orig.rename(columns={'best_macro_f1':'orig'})[['arm','condition','seed','orig']].drop_duplicates(['arm','condition','seed'])
m=df.merge(orig,on=['arm','condition','seed'],how='left'); m['d_oracle']=(m.oracle-m.orig)*100
print('reproduction: oracle(new) - original best_macro_f1 (pp):\n',m.groupby(['arm','condition']).d_oracle.agg(['mean','std','count']).round(2).to_string())
out=[]
for rule in ['oracle','last5','final']:
    t=df.pivot_table(index=['arm','condition'],values=rule,aggfunc=['mean','std','count']); 
    for (arm,c),r in t.iterrows(): out.append(dict(rule=rule,arm=arm,condition=c,mean=r[('mean',rule)]*100,sd=r[('std',rule)]*100,n=int(r[('count',rule)])))
T=pd.DataFrame(out); T.to_csv('p9_recheck/results/means_by_rule.csv',index=False)
print(T.pivot_table(index=['condition','arm'],columns='rule',values='mean').round(2).to_string())
def v(arm,c,rule): return df[(df.arm==arm)&(df.condition==c)].sort_values('seed').set_index('seed')[rule]*100
C=[]
for rule in ['oracle','last5','final']:
    for name,(a,b) in {'frozen-s4d':('bm3_frozen','s4d'),'frozen-nogate':('bm3_frozen','frozen_nogate'),'graft-s4d':('s4d_plus_gate','s4d')}.items():
        for c in band:
            x,y=v(a,c,rule),v(b,c,rule); s=x.index.intersection(y.index); d=(x[s]-y[s]).values
            if len(d)<3: continue
            h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            C.append(dict(rule=rule,contrast=name,condition=c,n=len(d),mean_diff_pp=d.mean(),ci_lo=d.mean()-h,ci_hi=d.mean()+h,n_pos=int((d>0).sum()),wilcoxon_p=wilcoxon(d,alternative='greater').pvalue))
    # R
    Rs=[]
    for c in band:
        g,s_,f=v('s4d_plus_gate',c,rule),v('s4d',c,rule),v('bm3_frozen',c,rule)
        if min(len(g),len(s_),len(f))>=1: Rs.append((g.mean()-s_.mean())/(f.mean()-s_.mean()))
    if len(Rs)==3: C.append(dict(rule=rule,contrast='R_recovery_ratio',condition='band_mean',n=min(len(g),len(s_),len(f)),mean_diff_pp=np.mean(Rs)))
Cd=pd.DataFrame(C); Cd.to_csv('p9_recheck/results/contrasts_by_rule.csv',index=False); print(Cd.round(3).to_string())
