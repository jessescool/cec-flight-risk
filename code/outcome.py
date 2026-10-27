"""Crew stage of the outcome model: P(fail | weather tier) and the non-weather floor, shared by training and the live check.

Estimation windows differ by design: the clear-weather floor uses the trailing 12 months because it drifts (2025 vs 2026),
while the low-cloud and fog rates use all flights because they rest on 1 and 9 events. The mixture is forced monotone in
weather: low cloud can never be scored safer than clear (max(rate, floor)), fog never safer than low cloud.
The previous-evening split of the floor is only usable when that evening is over, i.e. at lead 1."""
import numpy as np, pandas as pd
from common import jeffreys

FLOOR_WINDOW = pd.Timedelta(days=365)

def crew_rates(fl):
    """Rates from a flight table (index = date; columns fail, wx_worst, prev_fail, sched_arr). Each entry: (mean, lo, hi, k, n)."""
    def J(z): return jeffreys(int(z.fail.sum()), len(z))
    r = {t: J(fl[fl.wx_worst == t]) for t in (0, 1, 2)}
    recent = fl[(fl.wx_worst == 0) & (fl.index > fl.index.max() - FLOOR_WINDOW)]
    r['floor'] = J(recent)
    ok = recent[recent.prev_fail == 0]; r['floor_prev_ok'] = J(ok) if len(ok) else r['floor']
    pf = fl[(fl.wx_worst == 0) & (fl.prev_fail == 1)]; r['floor_prev_fail'] = J(pf) if len(pf) else r['floor']
    b = fl[fl.wx_worst == 2]; sun = b.index.dayofweek == 6; late = b.sched_arr >= '19:00'      # sensitivities only
    r['fog_sunday'], r['fog_weekday'], r['fog_late'], r['fog_early'] = J(b[sun]), J(b[~sun]), J(b[late]), J(b[~late])
    return r

def floor_for(r, prev_fail):
    """Floor given the previous evening's outcome: 1 did not land, 0 landed, anything else unknown."""
    if prev_fail == 1: return r['floor_prev_fail'][0]
    if prev_fail == 0: return r['floor_prev_ok'][0]
    return r['floor'][0]

def mixture(p_marg, p_below, floor, r, fog_key=2):
    """P(no landing) for P(tier>=1) = p_marg, P(tier 2) = p_below and a clear-weather floor; monotone in weather."""
    r1 = max(r[1][0], floor); r2 = max(r[fog_key][0], r1)
    return (1 - p_marg) * floor + (p_marg - p_below) * r1 + p_below * r2

def mixture_rows(p1, p2, r, prev_fail, fog_key=2):
    return np.array([mixture(a, b, floor_for(r, p), r, fog_key) for a, b, p in zip(p1, p2, prev_fail)])
