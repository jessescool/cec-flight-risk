"""Model audit: (1) rolling-origin reproduction of AUC/BSS with bootstrap CIs, (2) nested probabilities
P(below)=P(marg+)*P(below|marg+), (3) Wilson intervals for green/yellow/red failure rates, (4) Sunday-only intervals."""
import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/jcool/Desktop/cec-flight-risk/code'); from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import beta
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
    X=add_season(pd.concat(Fs,axis=1)); X=X.loc[:,X.notna().mean()>0.5]; return X.join(out[['worst','month','dow']],how='inner')
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (max(0,c-h),min(1,c+h))
def jeffreys(k,n): return beta.ppf([0.025,0.975],k+0.5,n-k+0.5)
rng=np.random.default_rng(0)
for lead in [3,4]:
    F=design(lead); X=F.drop(columns=['worst','month','dow']); X=X.fillna(X.median())
    y1=(F.worst>=1).astype(int); y2=(F.worst==2).astype(int)
    P=pd.DataFrame(index=F.index,columns=['p1','p2c'],dtype=float)
    for q in pd.date_range('2025-01-01','2026-10-01',freq='QS'):
        tr=F.index<q; te=(F.index>=q)&(F.index<q+pd.offsets.QuarterBegin(1))
        if te.sum()==0: continue
        m1=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000)).fit(X[tr],y1[tr])
        sub=tr&(y1==1)   # P(below | marginal+)
        m2=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000)).fit(X[sub],y2[sub])
        P.loc[te,'p1']=m1.predict_proba(X[te])[:,1]; P.loc[te,'p2c']=P.loc[te,'p1']*m2.predict_proba(X[te])[:,1]
    T=F.join(P).dropna(subset=['p1']); T['y1']=y1; T['y2']=y2; T['c1']=T.month.map(clim1); T['c2']=T.month.map(clim2)
    assert (T.p2c<=T.p1+1e-9).all()
    T.to_parquet(f'data/audit_lead{lead}.parquet')
    print(f"\n===== lead {lead} d, rolling-origin CV {T.index.min().date()}..{T.index.max().date()}, n={len(T)}, marginal+={T.y1.sum()}, below={T.y2.sum()}")
    # bootstrap CIs
    aucs=[];bss=[]
    for _ in range(1000):
        idx=rng.integers(0,len(T),len(T)); t=T.iloc[idx]
        if t.y2.sum()<3: continue
        aucs.append(roc_auc_score(t.y2,t.p2c)); bss.append(1-brier_score_loss(t.y2,t.p2c)/brier_score_loss(t.y2,t.c2))
    print(f"AUC(below) nested model = {roc_auc_score(T.y2,T.p2c):.3f}  95% CI [{np.percentile(aucs,2.5):.3f},{np.percentile(aucs,97.5):.3f}] ; climatology AUC {roc_auc_score(T.y2,T.c2):.3f}")
    print(f"AUC(marginal+) = {roc_auc_score(T.y1,T.p1):.3f} ; BSS(below) = {1-brier_score_loss(T.y2,T.p2c)/brier_score_loss(T.y2,T.c2):.3f}  95% CI [{np.percentile(bss,2.5):.3f},{np.percentile(bss,97.5):.3f}]")
    # calibration of nested p2c
    T['bin']=pd.cut(T.p2c,[0,0.02,0.05,0.1,0.2,0.4,1]); print(T.groupby('bin').agg(n=('y2','size'),pred=('p2c','mean'),obs=('y2','mean')).round(3))
    for lab,sel in [('all',T.index==T.index),('Oct-May',T.month.isin([10,11,12,1,2,3,4,5])),('Jun-Sep',T.month.isin([6,7,8,9])),('Sundays',T.dow==6),('Sundays Oct-May',(T.dow==6)&T.month.isin([10,11,12,1,2,3,4,5]))]:
        Z=T[sel]; g=Z.p1<0.15; yl=(Z.p1>=0.15)&(Z.p1<0.30); r=Z.p1>=0.30
        def line(name,mask):
            n=mask.sum(); k=int(Z.y2[mask].sum()); lo,hi=jeffreys(k,n) if n else (np.nan,np.nan)
            return f"   {name:7s} n={n:4d} ({n/len(Z)*100:4.0f}%)  below={k:3d}  rate={k/n if n else float('nan'):.3f}  95% Jeffreys [{lo:.3f},{hi:.3f}]"
        print(f" {lab} (n={len(Z)}, base below={Z.y2.mean():.3f})"); print(line('green',g)); print(line('yellow',yl)); print(line('red',r))
