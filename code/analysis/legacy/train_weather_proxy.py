"""Train lead-specific logistic models on archived model forecasts (Open-Meteo previous-runs) vs KCEC METAR outcomes.
Writes models/lead{N}.joblib. Run from the project root: python3 code/train.py"""
import pandas as pd, numpy as np, joblib, os, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
MODELS = ['ecmwf_ifs025','gfs_seamless','gem_seamless','icon_seamless']
out = pd.read_parquet('data/outcome_eve1842.parquet')
out['T_utc'] = [pd.Timestamp(d.date()).tz_localize(TZ).tz_convert('UTC')+pd.Timedelta(hours=18,minutes=42) for d in out.index]
for lead in range(1, 8):
    Fs = []
    for m in MODELS:
        pr = pd.read_parquet(f'data/prevruns_{m}.parquet').set_index('time')
        cols = {f'{v}_previous_day{lead}': v for v in VARS}
        if not all(c in pr.columns for c in cols): continue
        pr = pr[list(cols)].rename(columns=cols)
        rows = {d: window_features(pr, r.T_utc, m[:4]+'_') for d, r in out.iterrows()}
        F = pd.DataFrame({d: v for d, v in rows.items() if v}).T
        if F.notna().mean().mean() < 0.5: continue
        Fs.append(F)
    X = add_season(pd.concat(Fs, axis=1)); X = X.loc[:, X.notna().mean() > 0.5]
    F = X.join(out['worst'], how='inner'); X = F.drop(columns='worst'); med = X.median(); X = X.fillna(med)
    y1, y2 = (F.worst>=1).astype(int), (F.worst==2).astype(int)
    fits = {'marginal': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X, y1),
            # nested: P(below) = P(marginal+) * P(below | marginal+), so the two probabilities can never cross
            'below_given_marginal': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[y1==1], y2[y1==1])}
    joblib.dump(dict(fits=fits, columns=list(X.columns), median=med, n=len(X), models=MODELS), f'models/lead{lead}.joblib')
    print(f'lead {lead}: trained on {len(X)} days ({F.index.min().date()}..{F.index.max().date()}), {len(X.columns)} features')
