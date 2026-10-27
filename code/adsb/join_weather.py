"""Build the canonical flight-outcome table data/flight_outcomes.csv: every reconstructed OAK->CEC evening
(Sunday + weekday tables from evening_outcomes.py) joined to the KCEC METAR tier at its scheduled arrival.
Columns: date, sched_arr, outcome, conf, ac, evidence, wx_worst, wx_min_ceil, wx_min_vis, reached, fail, unknown."""
import os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); from common import *
os.chdir(ROOT)

S = pd.read_csv('data/sunday_outcomes.csv', parse_dates=['date'])
S['sched_arr'] = [default_arrival(d) for d in S.date]
# manual review 2026-06-14: the only 328 at OAK (N395MS) flew OAK->Merced that evening instead of CEC; fog at CEC (OVC002, 3/4 SM)
S.loc[S.date == '2026-06-14', ['outcome', 'conf', 'evidence']] = ['no_328_near_CEC', 'medium', 'manual review: N395MS left OAK 20:20 for Merced; no 328 inbound to CEC; fog at arrival time']
W = pd.read_csv('data/weekday_outcomes.csv', parse_dates=['date'])
R = pd.concat([S, W]).sort_values('date').drop_duplicates('date').reset_index(drop=True)

ml = load_metars()
wx = []
for _, r in R.iterrows():
    T = arrival_ts(r.date, r.sched_arr); w = ml.loc[T + ARRIVAL_WINDOW[0]: T + ARRIVAL_WINDOW[1]]
    wx.append(dict(wx_worst=w.tier.max() if len(w) else np.nan, wx_min_ceil=w.ceil.fillna(99999).min() if len(w) else np.nan, wx_min_vis=w.vsby.fillna(10).min() if len(w) else np.nan))
R = pd.concat([R, pd.DataFrame(wx)], axis=1)
R['reached'] = R.outcome.str.startswith('landed') | R.outcome.str.startswith('overnight')
R['fail'] = (R.outcome.str.startswith('missed') | R.outcome.str.startswith('no_328')) & (R.conf != 'low')
R['unknown'] = ~(R.reached | R.fail)
R = R[['date', 'sched_arr', 'outcome', 'conf', 'ac', 'evidence', 'wx_worst', 'wx_min_ceil', 'wx_min_vis', 'reached', 'fail', 'unknown']]
R.to_csv('data/flight_outcomes.csv', index=False)
K = R[~R.unknown & (R.sched_arr >= '17:00')]
print(f'{len(R)} flights, {len(K)} evening with a determinate outcome, {int(K.fail.sum())} did not land')
for t in (0, 1, 2):
    z = K[K.wx_worst == t]; m, lo, hi, _, _ = jeffreys(int(z.fail.sum()), len(z))
    print(f'  {TIER_LABEL[t]:22s} {int(z.fail.sum()):2d}/{len(z):3d}  {m:.3f} [{lo:.3f}, {hi:.3f}]')
