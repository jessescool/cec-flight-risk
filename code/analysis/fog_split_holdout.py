"""Held-out check of the confidence-conditioned fog rate. The split and its cut (0.3) were identified on the 2025Q1-2026Q4
out-of-sample table, so that table cannot also confirm them. Here the rates are estimated from out-of-sample rows before
2026-07-01 only and scored on 2026Q3-Q4 flights, for cuts 0.2 / 0.3 / 0.4, against the pooled rate on the same rows.
Writes data/fog_split_holdout.txt."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import ROOT, jeffreys
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
os.chdir(ROOT)
FL = pd.read_csv('data/flight_outcomes.csv', parse_dates=['date']).set_index('date')
SPLIT = pd.Timestamp('2026-07-01')
out = open('data/fog_split_holdout.txt', 'w')
def P(s): print(s); out.write(s + '\n')
P(f'held-out period: flights from {SPLIT.date()}; rates from out-of-sample rows before it')
P(f"{'lead':>4} {'cut':>4} {'n':>3} {'ev':>3}   pooled  BSS/logloss      conditioned BSS/logloss   conf k/n  surprise k/n")
for lead in range(1, 8):
    O = pd.read_parquet(f'data/outcome_oos_lead{lead}.parquet').join(FL[['wx_worst']])
    tr = O[O.index < SPLIT]; te = O[O.index >= SPLIT]
    b = tr[tr.wx_worst == 2]; r_pool = jeffreys(int(b.y.sum()), len(b))[0]
    base_te = te.p_pool - te.p2 * r_pool                         # mixture without its fog term (monotone max never binds for these rates)
    bb = brier_score_loss(te.y, np.full(len(te), tr.y.mean()))
    for cut in (0.2, 0.3, 0.4):
        hi, lo = b[b.p2 >= cut], b[b.p2 < cut]
        r_hi, r_lo = jeffreys(int(hi.y.sum()), len(hi))[0], jeffreys(int(lo.y.sum()), len(lo))[0]
        p_cond = base_te + te.p2 * np.where(te.p2 >= cut, r_hi, r_lo)
        f = lambda p: f'{1 - brier_score_loss(te.y, p) / bb:+.3f} {log_loss(te.y, p):.4f}'
        P(f'{lead:>4} {cut:>4} {len(te):>3} {int(te.y.sum()):>3}   {f(te.p_pool):>16}      {f(p_cond):>16}       {int(hi.y.sum())}/{len(hi):<4}  {int(lo.y.sum())}/{len(lo)}')
out.close()
