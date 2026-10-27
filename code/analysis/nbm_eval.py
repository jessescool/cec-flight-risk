import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
pd.set_option('display.width',250)
tz='America/Los_Angeles'
nb=pd.read_parquet('data/nbm_kcec.parquet'); out=pd.read_parquet('data/outcome_eve1842.parquet')
out['T_utc']=[pd.Timestamp(d.date()).tz_localize(tz).tz_convert('UTC')+pd.Timedelta(hours=18,minutes=42) for d in out.index]
rows=[]
for d,r in out.iterrows():
    for lead_d in [1,2,3]:   # 12Z cycles ~14h, ~38h, ~62h before a Monday-01Z arrival
        init=(r.T_utc-pd.Timedelta(days=lead_d)).normalize()+pd.Timedelta(hours=12)
        g=nb[nb.init==init]
        if g.empty: continue
        g=g.assign(dt=(g.valid-r.T_utc).abs()); near=g[g.dt<=pd.Timedelta(hours=2)]
        if near.empty: continue
        rows.append(dict(date=d,lead_h=(r.T_utc-init).total_seconds()/3600,cig_min=near.cig_ft.min(),vis_min=near.vis_mi.min(),ifc_max=near.IFC.max(),ifv_max=near.IFV.max(),worst=r.worst,month=r.month,dow=r.dow))
E=pd.DataFrame(rows); E.to_parquet('data/nbm_eval.parquet')
print("NBM bulletins used:",nb.init.nunique(),"days evaluated:",E.date.nunique(), E.date.min().date(), E.date.max().date())
E=E.dropna(subset=["ifc_max","cig_min"])
E["lead_bin"]=pd.cut(E.lead_h,[0,24,48,72],labels=["~14h (Sun 12Z)","~38h (Sat 12Z)","~62h (Fri 12Z)"])
for lh,g in E.groupby("lead_bin",observed=True):
    y2=(g.worst==2).astype(int); y1=(g.worst>=1).astype(int)
    # deterministic rule: NBM CIG<=500 or VIS<=1.5 at nearest hours -> flag
    flag=(g.cig_min<=500)|(g.vis_min<=1.5); flag2=(g.cig_min<300)|(g.vis_min<0.75)
    print(f"\n-- NBM 12Z, lead {lh} (n={len(g)}, below={y2.sum()}, marginal+={y1.sum()})")
    print(f"   AUC(IFC prob -> below)={roc_auc_score(y2,g.ifc_max):.3f}  AUC(IFC -> marginal+)={roc_auc_score(y1,g.ifc_max):.3f}  AUC(det. ceiling, inverted)={roc_auc_score(y2,-g.cig_min.clip(upper=5000)):.3f}")
    print(f"   rule 'NBM says marginal or worse': flagged {flag.mean()*100:.0f}% | P(below|flag)={y2[flag].mean():.2f} P(below|no flag)={y2[~flag].mean():.3f} recall={ (flag&(y2==1)).sum()/max(y2.sum(),1):.2f}")
    print(f"   rule 'NBM says below mins':        flagged {flag2.mean()*100:.0f}% | P(below|flag)={y2[flag2].mean() if flag2.sum() else float('nan'):.2f} recall={(flag2&(y2==1)).sum()/max(y2.sum(),1):.2f}")
    for thr in [20,30,40,50]:
        f=g.ifc_max>=thr; print(f"   IFC>={thr}%: flagged {f.mean()*100:4.0f}%  P(below|flag)={y2[f].mean() if f.sum() else float('nan'):.2f}  P(below|green)={y2[~f].mean():.3f}  recall={(f&(y2==1)).sum()/max(y2.sum(),1):.2f}")
