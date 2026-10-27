import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
rng=np.random.default_rng(0)
h=pd.read_csv('downloads/hrrr_cec.csv',header=None,names=['date','run','fh','var','val'])
h=h[~h['var'].isin(['err','missing'])]; h['val']=h.val.astype(float); h['date']=pd.to_datetime(h.date)
w=h.pivot_table(index=['date','run','fh'],columns='var',values='val').reset_index()
w['vis_mi']=w.vis/1609.34; w['ceil_ft']=np.where(w.ceil>=9000,20000,(w.ceil-18)*3.281).clip(0,20000); w['spread']=np.nan
def agg(g):
    return pd.Series(dict(vis_min=g.vis_mi.min(),ceil_min=g.ceil_ft.min(),lcc_max=g.lcc.max(),nh=len(g)))
F={}
for run in [0,6]:
    a=w[w.run==run].groupby('date').apply(agg); a=a[a.nh>=2]; F[run]=a.add_prefix(f'h{run:02d}_')
o=pd.read_parquet('data/outcome_eve1842.parquet'); o.index=pd.to_datetime(o.index)
X=o[['worst','month']].join(F[0],how='inner').join(F[6],how='inner')
X['month_sin']=np.sin(2*np.pi*X.month/12); X['month_cos']=np.cos(2*np.pi*X.month/12)
X['y1']=(X.worst>=1).astype(int); X['y2']=(X.worst==2).astype(int)
print(f'HRRR days with labels: {len(X)}  {X.index.min().date()}..{X.index.max().date()}  y1={X.y1.sum()} y2={X.y2.sum()}')
aud=pd.read_parquet('data/audit_lead3.parquet'); aud.index=pd.to_datetime(aud.index)
def feats(prefixes): return [c for c in X.columns if any(c.startswith(p) for p in prefixes)]+['month_sin','month_cos']
def roll(cols, start, end, y='y1', extra=None):
    qs=pd.date_range(start,end,freq='QS'); out=[]
    for q0 in qs:
        q1=q0+pd.offsets.QuarterBegin(startingMonth=1)
        tr=X[X.index<q0]; te=X[(X.index>=q0)&(X.index<q1)]
        if extra is not None:
            tr=tr.join(extra,how='inner'); te=te.join(extra,how='inner')
        if len(te)==0 or tr[y].sum()<5: continue
        cc=cols+(list(extra.columns) if extra is not None else [])
        m=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000)).fit(tr[cc].fillna(tr[cc].median()),tr[y])
        p=m.predict_proba(te[cc].fillna(tr[cc].median()))[:,1]
        out.append(pd.DataFrame(dict(p=p,y=te[y].values,clim=tr[y].mean()),index=te.index))
    return pd.concat(out)
def score(r,label):
    y,p=r.y.values,r.p.values; bs=brier_score_loss(y,p); bsc=brier_score_loss(y,r.clim.values)
    aucs=[];bss=[]
    for _ in range(400):
        i=rng.integers(0,len(y),len(y))
        if y[i].sum()==0 or y[i].sum()==len(i): continue
        aucs.append(roc_auc_score(y[i],p[i])); bss.append(1-brier_score_loss(y[i],p[i])/brier_score_loss(y[i],r.clim.values[i]))
    print(f'{label:46s} n={len(y):4d} events={int(y.sum()):3d}  AUC={roc_auc_score(y,p):.3f} [{np.percentile(aucs,2.5):.3f},{np.percentile(aucs,97.5):.3f}]  BSS={1-bs/bsc:+.3f} [{np.percentile(bss,2.5):+.3f},{np.percentile(bss,97.5):+.3f}]')
    return r
print('\n=== Target: marginal-or-worse at arrival (y1). Rolling quarterly CV 2025Q1-2026Q4, same window as the ensemble audit ===')
ens=aud[['p1']].rename(columns={'p1':'p'}).join(X[['y1']]).rename(columns={'y1':'y'}).dropna(); ens['clim']=X.loc[X.index<'2025-01-01','y1'].mean()
common=ens.index
score(ens,'Ensemble (ECMWF/GFS/GEM/ICON, Fri 00Z)')
r0=roll(feats(['h00_']),'2025-01-01','2026-10-01'); score(r0.loc[r0.index.intersection(common)],'HRRR 00Z Sat run (Fri 17:00 PDT), f46-48')
r6=roll(feats(['h06_']),'2025-01-01','2026-10-01'); score(r6.loc[r6.index.intersection(common)],'HRRR 06Z Sat run (Sat 01:00), f42-45')
lp=np.log(aud.p1.clip(1e-4,1-1e-4)/(1-aud.p1.clip(1e-4,1-1e-4))).rename('ens_logit').to_frame()
rs=roll(feats(['h00_']),'2025-01-01','2026-10-01',extra=lp); score(rs,'Ensemble + HRRR 00Z stacked')
rs6=roll(feats(['h06_']),'2025-01-01','2026-10-01',extra=lp); score(rs6,'Ensemble + HRRR 06Z stacked')
print('\n=== y1, HRRR alone over the long archive: rolling CV 2022Q1-2026Q4 (train from Dec 2020) ===')
score(roll(feats(['h00_']),'2022-01-01','2026-10-01'),'HRRR 00Z, 2022-2026')
score(roll(feats(['h06_']),'2022-01-01','2026-10-01'),'HRRR 06Z, 2022-2026')
print('\n=== Target: below minimums (y2) ===')
e2=aud[['p2c']].rename(columns={'p2c':'p'}).join(X[['y2']]).rename(columns={'y2':'y'}).dropna(); e2['clim']=X.loc[X.index<'2025-01-01','y2'].mean()
score(e2,'Ensemble nested P(below)')
score(roll(feats(['h00_']),'2025-01-01','2026-10-01',y='y2').loc[lambda d:d.index.isin(common)],'HRRR 00Z direct P(below)')
score(roll(feats(['h06_']),'2025-01-01','2026-10-01',y='y2').loc[lambda d:d.index.isin(common)],'HRRR 06Z direct P(below)')
score(roll(feats(['h00_']),'2022-01-01','2026-10-01',y='y2'),'HRRR 00Z P(below), 2022-2026')
print('\n=== Raw HRRR field as a rule (no fitting), all labelled days ===')
for run in [0,6]:
    for thr,(cf,vf,yy) in {'below (<300ft or <0.75mi)':(300,0.75,'y2'),'marginal+ (<=500ft or <=1.5mi)':(500,1.5,'y1')}.items():
        pred=((X[f'h{run:02d}_ceil_min']<cf)|(X[f'h{run:02d}_vis_min']<vf)).astype(int); y=X[yy]
        tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
        print(f'HRRR {run:02d}Z says {thr:32s}: flagged {pred.sum():4d} of {len(X)}, recall {tp/(tp+fn):.2f}, precision {tp/max(tp+fp,1):.2f}, base rate {y.mean():.3f}')
X.to_parquet('data/hrrr_features.parquet')
