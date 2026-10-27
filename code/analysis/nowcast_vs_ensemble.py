"""Does the afternoon METAR nowcast add to the lead-1 ensemble forecast, or replace it? Rolling quarterly CV 2025Q1-2026Q4
on every day with both inputs, target = below minimums at arrival. Three predictors at each horizon h = 1..5 h:
persistence table alone, ensemble lead-1 nested P(below) alone, logistic on [logit p_ens, state dummies]."""
import os, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
os.chdir(ROOT)
N = pd.read_parquet('data/nowcast_days.parquet'); N = N[N.index >= '2024-02-01']
FL = pd.read_csv('data/flight_outcomes.csv', parse_dates=['date']).set_index('date').sched_arr
T_utc = {d: arrival_ts(d, FL.get(d, default_arrival(d))).tz_convert('UTC') for d in N.index}
X = archived_features(1, T_utc); X = X.loc[X.index.intersection(N.index)]; X = X.fillna(X.median()); N = N.loc[X.index]
y1 = (N.worst >= 1).astype(int); y2 = (N.worst == 2).astype(int)
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
quarters = pd.period_range('2025Q1', '2026Q4', freq='Q')
print(f'days {len(N)}, below minimums {int(y2.sum())}')
print(f"{'h':>2} {'n':>4} {'ev':>3}   persistence AUC/BSS     ensemble AUC/BSS     combined AUC/BSS")
for h in range(1, 6):
    s = N[f's{h}']; ok = s.notna(); rows = []
    for q in quarters:
        t0 = q.start_time; tr = (N.index < t0) & ok; te = (N.index >= t0) & (N.index <= q.end_time) & ok
        if te.sum() == 0 or y2[tr].sum() < 10: continue
        f1 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y1[tr])
        f2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr & (y1 == 1)], y2[tr & (y1 == 1)])
        pe = lambda idx: f1.predict_proba(X[idx])[:, 1] * f2.predict_proba(X[idx])[:, 1]
        rate = {k: jeffreys(int(y2[tr & (s == k)].sum()), int((tr & (s == k)).sum()))[0] for k in ('0', 'lowcld', '1', '2')}
        pp = s[te].map(rate).values; p_ens = pe(te)
        Ztr = np.column_stack([logit(pe(tr)), pd.get_dummies(s[tr]).reindex(columns=['lowcld', '1', '2'], fill_value=0).values])
        Zte = np.column_stack([logit(p_ens), pd.get_dummies(s[te]).reindex(columns=['lowcld', '1', '2'], fill_value=0).values])
        pc = LogisticRegression(C=1.0, max_iter=3000).fit(Ztr, y2[tr]).predict_proba(Zte)[:, 1]
        rows.append(pd.DataFrame(dict(y=y2[te].values, pp=pp, pe=p_ens, pc=pc, base=y2[tr].mean()), index=N.index[te]))
    O = pd.concat(rows); b = brier_score_loss(O.y, np.full(len(O), O.base.mean()))
    f = lambda p: f'{roc_auc_score(O.y, p):.3f} {1 - brier_score_loss(O.y, p) / b:+.3f}'
    print(f'{h:>2} {len(O):>4} {int(O.y.sum()):>3}   {f(O.pp):>18}   {f(O.pe):>16}   {f(O.pc):>16}')
