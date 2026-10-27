import pandas as pd, numpy as np, sys, glob, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
pd.set_option('display.width',250)
tz='America/Los_Angeles'
out=pd.read_parquet('data/outcome_eve1842.parquet')
out['T_utc']=[pd.Timestamp(d.date()).tz_localize(tz).tz_convert('UTC')+pd.Timedelta(hours=18,minutes=42) for d in out.index]
models=sys.argv[1:] or ['gfs_seamless']
VARS=["temperature_2m","dew_point_2m","relative_humidity_2m","cloud_cover","wind_speed_10m","wind_direction_10m","weather_code","surface_pressure"]
def feats_for(model,lead):
    pr=pd.read_parquet(f'data/prevruns_{model}.parquet').set_index('time')
    cols={v:f'{v}_previous_day{lead}' for v in VARS if f'{v}_previous_day{lead}' in pr.columns}
    if 'temperature_2m' not in cols: return None
    rows=[]
    for d,r in out.iterrows():
        T=r.T_utc.floor('h')
        win=pr.loc[T-pd.Timedelta(hours=2):T+pd.Timedelta(hours=1)]   # 16:00-19:00 local-ish
        day=pr.loc[T-pd.Timedelta(hours=14):T+pd.Timedelta(hours=6)]   # preceding afternoon through late evening
        if len(win)<3 or win[cols['temperature_2m']].isna().all(): continue
        f=dict(date=d)
        t=win[cols['temperature_2m']]; td=win[cols['dew_point_2m']]
        f['spread_min']=(t-td).min(); f['spread_mean']=(t-td).mean()
        f['rh_max']=win[cols['relative_humidity_2m']].max() if 'relative_humidity_2m' in cols else np.nan
        f['cc_mean']=win[cols['cloud_cover']].mean() if 'cloud_cover' in cols else np.nan
        f['cc_day_mean']=day[cols['cloud_cover']].mean() if 'cloud_cover' in cols else np.nan
        f['ws']=win[cols['wind_speed_10m']].mean() if 'wind_speed_10m' in cols else np.nan
        wd=win[cols['wind_direction_10m']].mean() if 'wind_direction_10m' in cols else np.nan
        f['wd_sin']=np.sin(np.deg2rad(wd)); f['wd_cos']=np.cos(np.deg2rad(wd))
        f['wx_fog']=float((win[cols['weather_code']]==45).any()) if 'weather_code' in cols else np.nan
        f['wx_max']=win[cols['weather_code']].max() if 'weather_code' in cols else np.nan
        f['psfc']=win[cols['surface_pressure']].mean() if 'surface_pressure' in cols else np.nan
        f['tmax_day']=day[cols['temperature_2m']].max(); f['t_eve']=t.mean()
        f['spread_day_min']=(day[cols['temperature_2m']]-day[cols['dew_point_2m']]).min()
        rows.append(f)
    F=pd.DataFrame(rows).set_index('date')
    F['month_sin']=np.sin(2*np.pi*F.index.month/12); F['month_cos']=np.cos(2*np.pi*F.index.month/12)
    return F
# climatology baseline: monthly base rate from 2019-2023 (strictly earlier than the eval period)
clim=out[out.index<'2024-01-01'].groupby('month').worst.apply(lambda s:(s==2).mean())
clim1=out[out.index<'2024-01-01'].groupby('month').worst.apply(lambda s:(s>=1).mean())
results=[]
for model in models:
    for lead in range(1,8):
        F=feats_for(model,lead)
        if F is None or len(F)<200: continue
        F=F.join(out[['worst','month','dow']],how='inner').dropna(subset=['spread_min'])
        F['y2']=(F.worst==2).astype(int); F['y1']=(F.worst>=1).astype(int)
        X=F.drop(columns=["worst","month","dow","y1","y2"]); X=X.loc[:,X.notna().mean()>0.5]; X=X.fillna(X.median())
        split=pd.Timestamp('2025-10-01')   # train < split, test >= split  (time-based holdout)
        tr=F.index<split; te=~tr
        if te.sum()<100 or F.y2[te].sum()<5: continue
        for yname in ['y2','y1']:
            y=F[yname]
            lr=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000)).fit(X[tr],y[tr])
            gb=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=150,l2_regularization=1.0).fit(X[tr],y[tr])
            p_lr=lr.predict_proba(X[te])[:,1]; p_gb=gb.predict_proba(X[te])[:,1]
            p_clim=F.month[te].map(clim if yname=='y2' else clim1).values
            yt=y[te].values
            results.append(dict(model=model,lead=lead,target=yname,n_test=te.sum(),events=int(yt.sum()),
                brier_clim=brier_score_loss(yt,p_clim),brier_lr=brier_score_loss(yt,p_lr),brier_gb=brier_score_loss(yt,p_gb),
                auc_clim=roc_auc_score(yt,p_clim),auc_lr=roc_auc_score(yt,p_lr),auc_gb=roc_auc_score(yt,p_gb)))
R=pd.DataFrame(results); R['bss_lr']=1-R.brier_lr/R.brier_clim; R['bss_gb']=1-R.brier_gb/R.brier_clim
print(R.round(3).to_string())
R.to_csv('data/nwp_model_results.csv',index=False)
