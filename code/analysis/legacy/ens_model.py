import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
ARGS=sys.argv[1:]; sys.argv=['x']; exec(open('code/nwp_model.py').read().split('# climatology baseline')[0])
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score
pd.set_option('display.width',250)
models=ARGS if ARGS else ['ecmwf_ifs025','gfs_seamless','gem_seamless','icon_seamless']
clim=out[out.index<'2024-01-01'].groupby('month').worst.apply(lambda s:(s==2).mean())
clim1=out[out.index<'2024-01-01'].groupby('month').worst.apply(lambda s:(s>=1).mean())
res=[]
for lead in range(1,8):
    Fs=[]
    for m in models:
        F=feats_for(m,lead)
        if F is None: continue
        F=F.drop(columns=['month_sin','month_cos']).add_prefix(m[:4]+'_'); Fs.append(F)
    if not Fs: continue
    X=pd.concat(Fs,axis=1); X=X.loc[:,X.notna().mean()>0.5]
    X['month_sin']=np.sin(2*np.pi*X.index.month/12); X['month_cos']=np.cos(2*np.pi*X.index.month/12)
    F=X.join(out[['worst','month','dow']],how='inner'); F=F[F.filter(like='spread_min').notna().any(axis=1)]
    X=F.drop(columns=['worst','month','dow']); X=X.fillna(X.median())
    split=pd.Timestamp('2025-10-01'); tr=F.index<split; te=~tr
    for yname,yv,cl in [('below',(F.worst==2).astype(int),clim),('marginal+',(F.worst>=1).astype(int),clim1)]:
        lr=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000)).fit(X[tr],yv[tr])
        p=lr.predict_proba(X[te])[:,1]; pc=F.month[te].map(cl).values; yt=yv[te].values
        res.append(dict(lead=lead,target=yname,n_models=len(Fs),n_test=te.sum(),events=int(yt.sum()),auc=roc_auc_score(yt,p),auc_clim=roc_auc_score(yt,pc),brier=brier_score_loss(yt,p),brier_clim=brier_score_loss(yt,pc)))
R=pd.DataFrame(res); R['BSS']=1-R.brier/R.brier_clim; print(R.round(3).to_string()); R.to_csv('data/ens_model_results.csv',index=False)
