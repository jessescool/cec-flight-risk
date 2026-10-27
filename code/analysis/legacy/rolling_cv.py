import pandas as pd, numpy as np, sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/jcool/Desktop/cec-flight-risk/code'); from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
pd.set_option('display.width',250)
MODELS=['ecmwf_ifs025','gfs_seamless','gem_seamless','icon_seamless']
out=pd.read_parquet('data/outcome_eve1842.parquet')
out['T_utc']=[pd.Timestamp(d.date()).tz_localize(TZ).tz_convert('UTC')+pd.Timedelta(hours=18,minutes=42) for d in out.index]
clim1=out[out.index<'2024-01-01'].groupby('month').worst.apply(lambda s:(s>=1).mean()); clim2=out[out.index<'2024-01-01'].groupby('month').worst.apply(lambda s:(s==2).mean())
def design(lead):
    Fs=[]
    for m in MODELS:
        pr=pd.read_parquet(f'data/prevruns_{m}.parquet').set_index('time'); cols={f'{v}_previous_day{lead}':v for v in VARS}
        if not all(c in pr.columns for c in cols): continue
        pr=pr[list(cols)].rename(columns=cols); rows={d:window_features(pr,r.T_utc,m[:4]+'_') for d,r in out.iterrows()}
        F=pd.DataFrame({d:v for d,v in rows.items() if v}).T
        if F.notna().mean().mean()<0.5: continue
        Fs.append(F)
    X=add_season(pd.concat(Fs,axis=1)); X=X.loc[:,X.notna().mean()>0.5]; F=X.join(out[['worst','month','dow']],how='inner'); return F
allres=[]
for lead in [2,3,4,5]:
    F=design(lead); X=F.drop(columns=['worst','month','dow']); X=X.fillna(X.median())
    y1=(F.worst>=1).astype(int); y2=(F.worst==2).astype(int)
    P=pd.DataFrame(index=F.index,columns=['p1','p2'],dtype=float)
    for q in pd.date_range('2025-01-01','2026-10-01',freq='QS'):
        tr=F.index<q; te=(F.index>=q)&(F.index<q+pd.offsets.QuarterBegin(1))
        if te.sum()==0: continue
        for yy,col in [(y1,'p1'),(y2,'p2')]:
            P.loc[te,col]=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000)).fit(X[tr],yy[tr]).predict_proba(X[te])[:,1]
    T=F.join(P).dropna(subset=['p1']); T['y1']=y1; T['y2']=y2; T['c1']=T.month.map(clim1); T['c2']=T.month.map(clim2)
    T.to_parquet(f'data/rollingcv_lead{lead}.parquet')
    print(f"\n===== lead {lead} d  rolling-origin CV {T.index.min().date()}..{T.index.max().date()}  n={len(T)} marginal+={T.y1.sum()} below={T.y2.sum()}")
    print(f"AUC marginal+: model {roc_auc_score(T.y1,T.p1):.3f} vs climatology {roc_auc_score(T.y1,T.c1):.3f} | AUC below: model {roc_auc_score(T.y2,T.p2):.3f} vs clim {roc_auc_score(T.y2,T.c2):.3f} | BSS below {1-brier_score_loss(T.y2,T.p2)/brier_score_loss(T.y2,T.c2):.3f}")
    S=T[T.dow==6]; print(f"Sundays: n={len(S)} marginal+={S.y1.sum()} below={S.y2.sum()}")
    print("thr(p1)  all: flagged%  recall_below  P(below|green)  P(marg|green) || Sundays: flagged  caught_below/total  stranded_green")
    for thr in [0.10,0.15,0.20,0.25,0.30,0.40]:
        red=T.p1>=thr; g=~red; sr=S.p1>=thr
        print(f"{thr:.2f}         {red.mean()*100:5.1f}     {(red&(T.y2==1)).sum()/T.y2.sum():.2f}         {T.y2[g].mean():.3f}          {T.y1[g].mean():.3f}     ||   {sr.sum():3d}/{len(S)}     {(sr&(S.y2==1)).sum()}/{S.y2.sum()}            {((~sr)&(S.y2==1)).sum()}")
    # seasonal breakdown for the chosen thresholds
    for lab,ms in [('Oct-May',[10,11,12,1,2,3,4,5]),('Jun-Sep',[6,7,8,9])]:
        Z=T[T.month.isin(ms)]; red=Z.p1>=0.30; yel=(Z.p1>=0.15)&~red
        print(f"  {lab}: n={len(Z)} base below={Z.y2.mean():.3f} | red {red.mean()*100:.0f}% (P below|red {Z.y2[red].mean() if red.sum() else float('nan'):.2f}) yellow {yel.mean()*100:.0f}% (P below|yel {Z.y2[yel].mean() if yel.sum() else float('nan'):.2f}) green {(~red&~yel).mean()*100:.0f}% (P below|green {Z.y2[~red&~yel].mean():.3f})")
