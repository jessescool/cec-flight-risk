"""Train and validate the outcome model: P(the evening OAK->CEC flight does not land at CEC) at forecast leads 1-7 days.

  weather stage   P(tier t at arrival | archived forecast)   two nested logistic fits on every day since Feb 2024
  crew stage      P(fail | tier t)                            code/outcome.py; the fog rate is conditioned on the forecast's
                                                              own out-of-sample P(below) (persistent vs surprise fog)
  P(no landing) = sum_t P(tier t) * P(fail | tier t)

Validation: rolling-origin quarterly cross-validation 2025Q1-2026Q4 of what weekly_check.py ships at each lead: weather
fits, feature medians and crew rates all re-estimated from data before the quarter; the previous-evening floor split is
used at lead 1 only (it is not knowable earlier). The weather stage is also scored against the observed tier on every
day, and the red/green ledger (P(low cloud) >= 0.30 / < 0.15) against weather-coincident failures.
Writes data/outcome_model_results.txt and models/outcome_lead{N}.joblib."""
import os, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, MODELS, load_metars, worst_tier, arrival_ts, default_arrival, archived_features
from outcome import crew_rates, mixture_rows, FOG_CONF_CUT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
os.chdir(ROOT)
rng = np.random.default_rng(0)
QUARTERS = pd.period_range('2025Q1', '2026Q4', freq='Q')
RED, GREEN = 0.30, 0.15                       # P(low cloud) thresholds used only for the ledger below
BINS = [0, 0.04, 0.06, 0.10, 0.20, 1.0]       # reliability bins: around the floor, above it, and the two tails
C_WX = 0.1                                    # logistic regularisation; chosen in the Oct-19 weather-proxy CV (code/analysis/legacy/rolling_cv.py), not re-tuned here
MIN_TRAIN_BELOW_DAYS = 10                     # a quarter is scored only when the weather stage has seen >= 10 below-minimums days ...
MIN_TRAIN_FOG_FLIGHTS, MIN_TRAIN_FAILURES = 5, 3   # ... and the crew stage >= 5 fog flights and >= 3 failures, so early folds do not score the prior alone
MIN_PLATT_EVENTS = 5                          # Platt sensitivity needs a few events to fit a slope

# ---- flights with a determinate reconstructed outcome, evening arrivals only
FL = pd.read_csv('data/flight_outcomes.csv', parse_dates=['date'])
FL = FL[(FL.sched_arr >= '17:00') & ~FL.unknown].set_index('date').sort_index()
FL['fail'] = FL.fail.astype(int)
FL['wx_fail'] = ((FL.fail == 1) & (FL.wx_worst >= 1)).astype(int)
FL['prev_fail'] = np.where(FL.index.to_series().diff().dt.days == 1, FL.fail.shift(1), np.nan)

# ---- weather-stage target for every calendar day: worst tier around that day's scheduled arrival
ml = load_metars()
days = pd.date_range('2024-02-01', ml.index[-1].date() - pd.Timedelta(days=1))
D = pd.DataFrame(index=days)
D['arr'] = [FL.sched_arr.get(d, default_arrival(d)) for d in days]
D['T_local'] = [arrival_ts(d, a) for d, a in zip(days, D.arr)]
D['worst'] = [worst_tier(ml, t) for t in D.T_local]
D = D.dropna(subset=['worst'])

def wx_fit(X, y1, y2):
    f1 = make_pipeline(StandardScaler(), LogisticRegression(C=C_WX, max_iter=3000)).fit(X, y1)
    f2 = make_pipeline(StandardScaler(), LogisticRegression(C=C_WX, max_iter=3000)).fit(X[y1 == 1], y2[y1 == 1])
    return f1, f2
def wx_pred(f1, f2, X):
    p1 = f1.predict_proba(X)[:, 1]; return p1, p1 * f2.predict_proba(X)[:, 1]

def boot_auc(y, p, B=1000):
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y): vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])
def score(lab, y, p, base):
    y = np.asarray(y); p = np.asarray(p); auc = roc_auc_score(y, p); lo, hi = boot_auc(y, p)
    bss = 1 - brier_score_loss(y, p) / brier_score_loss(y, np.full_like(p, base))
    return dict(label=lab, n=len(y), events=int(y.sum()), auc=auc, auc_lo=lo, auc_hi=hi, bss=bss, logloss=log_loss(y, p), logloss_clim=log_loss(y, np.full_like(p, base)))
def fmt(s): return f"  {s['label']:50s} n={s['n']:4d} events={s['events']:3d}  AUC={s['auc']:.3f} [{s['auc_lo']:.3f},{s['auc_hi']:.3f}]  BSS={s['bss']:+.3f}  logloss {s['logloss']:.4f} (clim {s['logloss_clim']:.4f})"

out = open('data/outcome_model_results.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')

rates = crew_rates(FL)
P(f'evening flights with a determinate outcome: {len(FL)} ({FL.index.min().date()}..{FL.index.max().date()}), did not land {FL.fail.sum()}, of which with marginal-or-worse weather {FL.wx_fail.sum()}')
P(f'weather-stage days: {len(D)}, marginal-or-worse {int((D.worst >= 1).sum())}, below minimums {int((D.worst == 2).sum())}')
for yr in (2025, 2026):
    z = FL[(FL.index.year == yr) & (FL.wx_worst == 0)]; P(f'  clear-weather failures {yr}: {int(z.fail.sum())}/{len(z)} = {z.fail.mean():.3f}')
P('crew-stage rates (Jeffreys mean [95% interval], k/n):')
for key, lab in [(0, 'P(fail | clear), all-time'), ('floor', 'P(fail | clear), trailing 12 mo  <- floor'), ('floor_prev_ok', '  ... previous evening landed (lead 1 only)'), ('floor_prev_fail', '  ... previous evening did not land (lead 1 only)'),
                 (1, 'P(fail | low cloud, landable); floored at the floor'), (2, 'P(fail | below minimums), pooled; shipped split by lead below'), ('fog_sunday', '  sensitivity: Sunday'), ('fog_weekday', '  sensitivity: weekday'), ('fog_late', '  sensitivity: arrival >= 19:00'), ('fog_early', '  sensitivity: arrival < 19:00')]:
    m, lo, hi, k, n = rates[key]; P(f'  {lab:52s} {m:.3f} [{lo:.3f}, {hi:.3f}]  {k}/{n}')

T_utc = {d: t.tz_convert('UTC') for d, t in D.T_local.items()}
for lead in range(1, 8):
    Xraw = archived_features(lead, T_utc); Xraw = Xraw.loc[Xraw.index.intersection(D.index)]
    used = [m for m in MODELS if any(c.startswith(m[:4] + '_') for c in Xraw.columns)]
    y1 = (D.loc[Xraw.index].worst >= 1).astype(int); y2 = (D.loc[Xraw.index].worst == 2).astype(int)
    fl = FL.loc[FL.index.intersection(Xraw.index)]
    rows, wx_rows = [], []; fl['p2_oos'] = np.nan
    for q in QUARTERS:
        t0 = q.start_time; tr = Xraw.index < t0; te_d = (Xraw.index >= t0) & (Xraw.index <= q.end_time)
        te = (fl.index >= t0) & (fl.index <= q.end_time); trf = fl.index < t0
        if te_d.sum() == 0 or y2[tr].sum() < MIN_TRAIN_BELOW_DAYS or (fl.wx_worst.values[trf] == 2).sum() < MIN_TRAIN_FOG_FLIGHTS or fl.fail.values[trf].sum() < MIN_TRAIN_FAILURES: continue
        med = Xraw[tr].median(); Xtr = Xraw[tr].fillna(med)
        f1, f2 = wx_fit(Xtr, y1[tr], y2[tr]); r = crew_rates(fl[trf])          # fl.p2_oos holds earlier quarters' out-of-sample P(below)
        d1, d2 = wx_pred(f1, f2, Xraw[te_d].fillna(med)); wx_rows.append(pd.DataFrame(dict(y1=y1[te_d].values, y2=y2[te_d].values, p1=d1, p2=d2, b1=y1[tr].mean(), b2=y2[tr].mean()), index=Xraw.index[te_d]))
        if te.sum() == 0: continue
        p1, p2 = wx_pred(f1, f2, Xraw.loc[fl.index[te]].fillna(med)); fl.loc[fl.index[te], 'p2_oos'] = p2
        pv = fl.prev_fail.values[te] if lead == 1 else np.full(te.sum(), np.nan)
        rows.append(pd.DataFrame(dict(y=fl.fail.values[te], ywx=fl.wx_fail.values[te], p1=p1, p2=p2, base=fl.fail.values[trf].mean(),
                                      p_live=mixture_rows(p1, p2, r, pv), p_pool=mixture_rows(p1, p2, r, pv, 2),
                                      p_sun=np.where(fl.index[te].dayofweek == 6, mixture_rows(p1, p2, r, pv, 'fog_sunday'), mixture_rows(p1, p2, r, pv, 'fog_weekday')),
                                      p_time=np.where(fl.sched_arr.values[te] >= '19:00', mixture_rows(p1, p2, r, pv, 'fog_late'), mixture_rows(p1, p2, r, pv, 'fog_early'))),
                                 index=fl.index[te]))
    O = pd.concat(rows); base = O.base.mean(); Wd = pd.concat(wx_rows); rates = crew_rates(fl)      # final rates: fog split uses every flight's out-of-sample P(below) at this lead
    P(f"\n=== lead {lead} day(s): models {used}, weather-stage days {len(Xraw)}, flights {len(fl)}, out-of-sample days {len(Wd)}, flights {len(O)} ({O.index.min().date()}..{O.index.max().date()})")
    w1 = score('weather stage: P(low cloud) vs observed tier >= 1, all days', Wd.y1, Wd.p1, Wd.b1.mean()); P(fmt(w1))
    w2 = score('weather stage: P(below) vs observed tier 2, all days', Wd.y2, Wd.p2, Wd.b2.mean()); P(fmt(w2))
    wx_auc = roc_auc_score(O.ywx, O.p2); wx_lo, wx_hi = boot_auc(O.ywx.values, O.p2.values)
    P(f"  weather stage: P(below) vs weather-coincident flight failure   events={int(O.ywx.sum()):3d}  AUC={wx_auc:.3f} [{wx_lo:.3f},{wx_hi:.3f}]")
    red = O.p1 >= RED; green = O.p1 < GREEN
    P(f"  ledger: red (P(low cloud) >= {RED}) on {int(red.sum())} flights caught {int(O.ywx[red].sum())}/{int(O.ywx.sum())} weather failures; green (< {GREEN}) on {int(green.sum())} flights had {int(O.ywx[green].sum())}")
    live = score('P(no landing), shipped configuration', O.y, O.p_live, base); P(fmt(live))
    P(fmt(score('  sensitivity: pooled fog rate (unconditioned)', O.y, O.p_pool, base)))
    P(f"  fog rate at this lead: forecast saw it (P(below) >= {FOG_CONF_CUT}) {rates['fog_conf'][3]}/{rates['fog_conf'][4]} = {rates['fog_conf'][0]:.2f} [{rates['fog_conf'][1]:.2f}, {rates['fog_conf'][2]:.2f}];  surprise fog {rates['fog_surprise'][3]}/{rates['fog_surprise'][4]} = {rates['fog_surprise'][0]:.2f} [{rates['fog_surprise'][1]:.2f}, {rates['fog_surprise'][2]:.2f}]")
    P(fmt(score('  sensitivity: Sunday/weekday fog rate', O.y, O.p_sun, base)))
    P(fmt(score('  sensitivity: arrival-time fog rate', O.y, O.p_time, base)))
    O['p_platt'] = np.nan; lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    for q in QUARTERS:
        tr = O.index < q.start_time; te = (O.index >= q.start_time) & (O.index <= q.end_time)
        if te.sum() and O.y[tr].sum() >= MIN_PLATT_EVENTS: O.loc[te, 'p_platt'] = LogisticRegression(C=1.0).fit(lg(O.p_live[tr]).values.reshape(-1, 1), O.y[tr]).predict_proba(lg(O.p_live[te]).values.reshape(-1, 1))[:, 1]
    Pl = O.dropna(subset=['p_platt']); P(fmt(score('  sensitivity: Platt-rescaled (same rows)', Pl.y, Pl.p_platt, Pl.base.mean()))); P(fmt(score('  shipped on those rows', Pl.y, Pl.p_live, Pl.base.mean())))
    g = O.groupby(pd.cut(O.p_live, BINS), observed=True).agg(n=('y', 'size'), observed=('y', 'mean'), predicted=('p_live', 'mean'), failures=('y', 'sum'))
    P('  reliability:'); P(g.round(3).to_string())
    O.to_parquet(f'data/outcome_oos_lead{lead}.parquet')
    reliability = [dict(lo=float(i.left), hi=float(i.right), n=int(r.n), failures=int(r.failures), observed=float(r.observed), predicted=float(r.predicted)) for i, r in g.iterrows()]
    med = Xraw.median(); f1, f2 = wx_fit(Xraw.fillna(med), y1, y2)
    joblib.dump(dict(wx=dict(marginal=f1, below_given_marginal=f2), rates=rates, columns=list(Xraw.columns), median=med, models=MODELS, models_used=used,
                     n_days=len(Xraw), n_flights=len(fl), n_fail=int(fl.fail.sum()), trained_through=str(Xraw.index.max().date()),
                     metrics=dict(wx_auc=wx_auc, wx_auc_lo=wx_lo, wx_auc_hi=wx_hi, tier1_auc=w1['auc'], tier2_auc=w2['auc'], tier2_bss=w2['bss'], **{k: v for k, v in live.items() if k != 'label'}),
                     reliability=reliability), f'models/outcome_lead{lead}.joblib')
out.close()
