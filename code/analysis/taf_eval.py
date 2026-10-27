import pandas as pd, numpy as np, sys
sys.path.insert(0,'code')
from tafdecode import decode, forecast_at
taf=pd.read_parquet('data/cec_taf_raw.parquet').dropna(subset=['issued'])
# validity start timestamp
def vstart(r):
    base=r.issued.replace(day=1,hour=0,minute=0)
    try: t=base.replace(day=r.vfrom_d)+pd.Timedelta(hours=r.vfrom_h)
    except ValueError: t=(base+pd.offsets.MonthBegin(1)).replace(day=r.vfrom_d)+pd.Timedelta(hours=r.vfrom_h)
    if t<r.issued-pd.Timedelta(days=15): t=t+pd.offsets.MonthBegin(1)
    if t>r.issued+pd.Timedelta(days=15): t=t-pd.offsets.MonthBegin(1)
    return t
taf['vstart']=taf.apply(vstart,axis=1)
taf['vend']=taf.vstart+pd.Timedelta(hours=24)  # KCEC TAFs are 24h (vto fields)
taf['vend']=[vs.replace(hour=0)+pd.Timedelta(hours=24*((vt_d-vs.day)%31)+vt_h) if False else vs+pd.Timedelta(hours=(( (vt_d-vs.day)%31)*24 + vt_h - vs.hour)) for vs,vt_d,vt_h in zip(taf.vstart,taf.vto_d,taf.vto_h)]
taf['periods']=[decode(t,v) for t,v in zip(taf.text,taf.vstart)]
out=pd.read_parquet('data/outcome_eve1842.parquet')
tz='America/Los_Angeles'
def tier(c,v):
    c=99999 if np.isnan(c) else c; v=10 if np.isnan(v) else v
    if c<300 or v<0.75: return 2
    if c<=500 or v<=1.5: return 1
    return 0
rows=[]
issued=taf.issued.values
for d,r in out.iterrows():
    T=pd.Timestamp(d.date()).tz_localize(tz)+pd.Timedelta(hours=18,minutes=42); Tu=T.tz_convert('UTC')
    for lead in [2,8,14,20,26]:
        cutoff=Tu-pd.Timedelta(hours=lead)
        idx=np.searchsorted(issued,np.datetime64(cutoff.tz_convert(None)))-1
        if idx<0: continue
        t=taf.iloc[idx]
        if not (t.vstart<=Tu<t.vend): 
            rows.append(dict(date=d,lead=lead,covered=False)); continue
        f=forecast_at(t.periods,Tu,t.vend)
        if f is None: continue
        rows.append(dict(date=d,lead=lead,covered=True,age_h=(Tu-t.issued).total_seconds()/3600,
            base_tier=tier(f['ceil'],f['vis']),cons_tier=tier(f['tempo_ceil'],f['tempo_vis']),
            fc_ceil=f['ceil'],fc_vis=f['vis'],tempo=f['tempo'],prob=f['prob'],obs_tier=r.worst,dow=r.dow,month=r.month))
ev=pd.DataFrame(rows); ev.to_parquet('data/taf_eval.parquet')
print("coverage by lead:"); print(ev.groupby('lead').covered.mean().round(2))
e=ev[ev.covered]
def metrics(g,col):
    obs_bad=g.obs_tier==2; fc_bad=g[col]==2; fc_warn=g[col]>=1
    return pd.Series(dict(n=len(g),base_rate=obs_bad.mean(),
        hit_red=(fc_bad&obs_bad).sum()/max(obs_bad.sum(),1),          # recall of below-mins using RED only
        hit_yellow_or_red=(fc_warn&obs_bad).sum()/max(obs_bad.sum(),1),
        far_red=(fc_bad&~obs_bad).sum()/max(fc_bad.sum(),1),         # false alarm ratio of red
        frac_red=fc_bad.mean(),frac_warn=fc_warn.mean(),
        miss_given_green=(obs_bad&~fc_warn).sum()/max((~fc_warn).sum(),1)))
for col in ['base_tier','cons_tier']:
    print(f"\n== {col} (all days, 2019-2026) =="); print(e.groupby('lead').apply(metrics,col).round(3))
print("\n== cons_tier, Sundays only =="); print(e[e.dow==6].groupby('lead').apply(metrics,'cons_tier').round(3))
print("\n== cons_tier, Jun-Oct all days =="); print(e[e.month.isin([6,7,8,9,10])].groupby('lead').apply(metrics,'cons_tier').round(3))
