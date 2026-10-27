import pandas as pd, numpy as np
m=pd.read_parquet('data/cec_metar.parquet')
m=m.sort_values('valid')
m['ceil_f']=m.ceil.fillna(99999); m['vis_f']=m.vsby.fillna(10)
# minimums (ILS/LPV 12): DA 310 MSL = 250 HAT, vis 3/4 SM (1/2 with FD/AP). METAR ceilings are in 100ft steps.
def tier(c,v):
    if c<300 or v<0.75: return 2   # below published mins -> approach very unlikely to succeed
    if c<=500 or v<=1.5: return 1  # marginal: at/near mins, approach-dependent
    return 0
m['tier']=[tier(c,v) for c,v in zip(m.ceil_f,m.vis_f)]
m['date']=m.local.dt.date; m['hr']=m.local.dt.hour+m.local.dt.minute/60
def window_stats(lo,hi,label):
    w=m[(m.hr>=lo)&(m.hr<hi)]
    g=w.groupby('date').agg(n=('tier','size'),worst=('tier','max'),minceil=('ceil_f','min'),minvis=('vis_f','min'))
    g=g[g.n>=1]
    g.index=pd.to_datetime(g.index); g['dow']=g.index.dayofweek; g['month']=g.index.month; g['year']=g.index.year
    g.to_parquet(f'data/outcome_{label}.parquet')
    return g
g=window_stats(18.0,19.75,'eve1842')   # arrival 18:42 window: obs 18:00-19:45 local
print("Days with obs:",len(g), g.index.min().date(), g.index.max().date())
print("\nAll days — fraction of evenings (18:00-19:45) by tier, by month:")
t=g.groupby('month').worst.value_counts(normalize=True).unstack().fillna(0).round(3); t.columns=['ok','marginal','below']; print(t)
print("\nSundays only:")
s=g[g.dow==6]; t=s.groupby('month').worst.value_counts(normalize=True).unstack().fillna(0).round(3); t.columns=['ok','marginal','below']; t['n']=s.groupby('month').size(); print(t)
print("\nSunday below-mins by year (all months):"); print(s.groupby('year').apply(lambda d: pd.Series(dict(n=len(d),below=(d.worst==2).mean().round(3),marg_or_below=(d.worst>=1).mean().round(3)))))
# persistence: if below at 18:00-19:45, how often is it still below at 20-21, 21-22, 22-23 and next morning 07-08?
def tier_in(lo,hi):
    w=m[(m.hr>=lo)&(m.hr<hi)]; return w.groupby('date').tier.max()
bad=g[g.worst==2].index.date
for lo,hi,lab in [(20,21,'20-21'),(21,22,'21-22'),(22,23.5,'22-23:30'),]:
    t2=tier_in(lo,hi); t2.index=pd.to_datetime(t2.index)
    sub=t2.reindex(pd.to_datetime(bad)).dropna()
    print(f"Given below-mins at 18:00-19:45, still below at {lab}: {(sub==2).mean():.2f} (n={len(sub)})")
# next-morning 07-09 for the Monday-morning CEC->OAK 08:00 (not needed) and OAK->CEC alt; skip
# Oct-May season Sundays only
oc=s[s.month.isin([10,11,12,1,2,3,4,5])]
print("\nOct–May Sundays: n=%d below=%.3f marginal+=%.3f"%(len(oc),(oc.worst==2).mean(),(oc.worst>=1).mean()))
ju=s[s.month.isin([6,7,8,9])]
print("Jun–Sep Sundays: n=%d below=%.3f marginal+=%.3f"%(len(ju),(ju.worst==2).mean(),(ju.worst>=1).mean()))
