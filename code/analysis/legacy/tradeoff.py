import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.argv=['x']; exec(open('code/nwp_model.py').read().split('# climatology baseline')[0])  # reuse feats_for/out
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve
pd.set_option('display.width',250)
def build(model,lead,target='y1'):
    F=feats_for(model,lead).join(out[['worst','month','dow']],how='inner').dropna(subset=['spread_min'])
    F['y']=(F.worst>=1).astype(int) if target=='y1' else (F.worst==2).astype(int); F['ybelow']=(F.worst==2).astype(int)
    X=F.drop(columns=['worst','month','dow','y','ybelow']); X=X.loc[:,X.notna().mean()>0.5]; X=X.fillna(X.median())
    return F,X
for model,lead in [('ecmwf_ifs025',3),('ecmwf_ifs025',2),('ecmwf_ifs025',4),('gfs_seamless',3)]:
    F,X=build(model,lead)
    split=pd.Timestamp('2025-10-01'); tr=F.index<split; te=~tr
    lr=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000)).fit(X[tr],F.y[tr])
    p=pd.Series(lr.predict_proba(X[te])[:,1],index=F.index[te])
    T=F[te].copy(); T['p']=p
    print(f"\n===== {model} lead {lead}d — holdout {T.index.min().date()}..{T.index.max().date()}, n={len(T)}, marginal+={T.y.sum()}, below={T.ybelow.sum()}")
    # calibration
    bins=[0,0.1,0.2,0.3,0.5,0.7,1.0]; T['bin']=pd.cut(T.p,bins)
    print(T.groupby('bin').agg(n=('y','size'),pred=('p','mean'),obs_marg=('y','mean'),obs_below=('ybelow','mean')).round(3))
    # threshold tradeoff
    print("thr  flagged%  recall_marg  recall_below  P(below|green)  P(marg|green)")
    for thr in [0.1,0.15,0.2,0.25,0.3,0.4,0.5]:
        red=T.p>=thr; g=~red
        print(f"{thr:.2f}  {red.mean()*100:5.1f}   {(red&(T.y==1)).sum()/T.y.sum():.2f}         {(red&(T.ybelow==1)).sum()/max(T.ybelow.sum(),1):.2f}          {T.ybelow[g].mean():.3f}          {T.y[g].mean():.3f}")
    # Sundays only in holdout
    S=T[T.dow==6]; print(f"Sundays in holdout: n={len(S)}, marginal+={S.y.sum()}, below={S.ybelow.sum()}; mean p on below-days={S.p[S.ybelow==1].mean():.2f} vs ok-days={S.p[S.worst==0].mean():.2f}")
    if model=='ecmwf_ifs025' and lead==3:
        coef=pd.Series(lr[-1].coef_[0],index=X.columns).sort_values(); print("coefficients (std.):"); print(coef.round(2).to_string())
