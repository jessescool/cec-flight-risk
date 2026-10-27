"""Crew stage of the outcome model: P(fail | weather tier) and the non-weather floor, shared by training and the live check.

Estimation windows differ by design: the clear-weather floor uses the trailing 12 months because it drifts (2025 vs 2026),
the low-cloud rate uses all flights (1 event). The fog rate is conditioned on the forecast's own confidence: fog the
forecast saw coming (P(below) >= FOG_CONF_CUT) is the persistent kind and crews fail about half the time; fog it did not
see is brief and crews land. The two rates are estimated from flights with an out-of-sample P(below) for the lead in
question (column p2_oos), falling back to the pooled rate when fewer than MIN_FOG_ROWS such flights exist.
The mixture is forced monotone in weather. The previous-evening split of the floor is only usable at lead 1."""
import numpy as np, pandas as pd
from common import jeffreys

FLOOR_WINDOW = pd.Timedelta(days=365)   # the clear-weather rate quadrupled 2025 -> 2026; one year keeps ~200 flights
FOG_CONF_CUT = 0.3            # P(below) at or above this = "forecast saw it coming". Chosen on the 2025-2026 out-of-sample table; 0.2 and 0.4 give the
                              # same sign of gain, and a held-out check on 2026Q3-Q4 is in data/fog_split_holdout.txt
MIN_FOG_ROWS = 8              # below this many fog flights with an out-of-sample P(below) the halves would hold ~4 flights each

def crew_rates(fl):
    """Rates from a flight table (index = date; columns fail, wx_worst, prev_fail, sched_arr, optional p2_oos). Entries: (mean, lo, hi, k, n)."""
    def J(z): return jeffreys(int(z.fail.sum()), len(z))
    r = {t: J(fl[fl.wx_worst == t]) for t in (0, 1, 2)}
    recent = fl[(fl.wx_worst == 0) & (fl.index > fl.index.max() - FLOOR_WINDOW)]
    r['floor'] = J(recent)
    ok = recent[recent.prev_fail == 0]; r['floor_prev_ok'] = J(ok) if len(ok) else r['floor']        # both halves from the same trailing window
    pf = recent[recent.prev_fail == 1]; r['floor_prev_fail'] = J(pf) if len(pf) else r['floor']
    b = fl[fl.wx_worst == 2]
    bo = b.dropna(subset=['p2_oos']) if 'p2_oos' in b else b.iloc[0:0]
    conf, surp = (J(bo[bo.p2_oos >= FOG_CONF_CUT]), J(bo[bo.p2_oos < FOG_CONF_CUT])) if len(bo) >= MIN_FOG_ROWS else (r[2], r[2])
    # the split is justified only by the mechanism (forecastable fog is persistent fog); if the data at this lead do not show it, stay pooled
    r['fog_conf'], r['fog_surprise'] = (conf, surp) if conf[0] >= surp[0] else (r[2], r[2])
    sun = b.index.dayofweek == 6; late = b.sched_arr >= '19:00'      # sensitivities only
    r['fog_sunday'], r['fog_weekday'], r['fog_late'], r['fog_early'] = J(b[sun]), J(b[~sun]), J(b[late]), J(b[~late])
    return r

def floor_for(r, prev_fail):
    """Floor given the previous evening's outcome: 1 did not land, 0 landed, anything else unknown."""
    if prev_fail == 1: return r['floor_prev_fail'][0]
    if prev_fail == 0: return r['floor_prev_ok'][0]
    return r['floor'][0]

def fog_rate(r, p_below, fog_key=None):
    if fog_key: return r[fog_key][0]
    return r['fog_conf'][0] if p_below >= FOG_CONF_CUT else r['fog_surprise'][0]

def mixture(p_marg, p_below, floor, r, fog_key=None):
    """P(no landing) for P(tier>=1) = p_marg, P(tier 2) = p_below and a clear-weather floor; monotone in weather."""
    r1 = max(r[1][0], floor); r2 = max(fog_rate(r, p_below, fog_key), r1)
    return (1 - p_marg) * floor + (p_marg - p_below) * r1 + p_below * r2

def mixture_rows(p1, p2, r, prev_fail, fog_key=None):
    return np.array([mixture(a, b, floor_for(r, p), r, fog_key) for a, b, p in zip(p1, p2, prev_fail)])
