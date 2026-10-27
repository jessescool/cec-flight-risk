"""Same-day persistence nowcast: P(tier at arrival | latest METAR h hours before arrival), from the KCEC METAR archive.
Arrival = the scheduled time for reconstructed flight days, else the era default. Writes data/nowcast_table.csv
(lead_h, state, n, n_marg, n_below, p_marg, p_below) which weekly_check.py reads when run inside 1.5 h of arrival; from 2 h out the lead-1 forecast scores better (code/analysis/nowcast_vs_ensemble.py)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, NOWCAST_STATES, load_metars, nowcast_state, arrival_ts, default_arrival, worst_tier, jeffreys
os.chdir(ROOT)
MAX_METAR_AGE = pd.Timedelta(minutes=75)   # routine METARs are hourly; older than this means the hour's report is missing

def build_days():
    ml = load_metars(); ml['state'] = [nowcast_state(c, v) for c, v in zip(ml.ceil, ml.vsby)]
    FL = pd.read_csv('data/flight_outcomes.csv', parse_dates=['date']).set_index('date').sched_arr
    rows = []
    for d in pd.date_range(ml.index[0].date() + pd.Timedelta(days=1), ml.index[-1].date() - pd.Timedelta(days=1)):
        T = arrival_ts(d, FL.get(d, default_arrival(d))); w = worst_tier(ml, T)
        if pd.isna(w): continue
        r = {'date': d, 'worst': w}
        for h in range(1, 6):
            s = ml.loc[:T - pd.Timedelta(hours=h)].tail(1)
            if len(s) and (T - pd.Timedelta(hours=h)) - s.index[0] < MAX_METAR_AGE: r[f's{h}'] = s.state.iloc[0]
        rows.append(r)
    return pd.DataFrame(rows).set_index('date')

def table(D):
    out = []
    for h in range(1, 6):
        for s in NOWCAST_STATES:
            z = D[D[f's{h}'] == s]
            if len(z) == 0: continue
            pm, pm_lo, pm_hi, km, n = jeffreys(int((z.worst >= 1).sum()), len(z)); pb, pb_lo, pb_hi, kb, _ = jeffreys(int((z.worst == 2).sum()), len(z))
            out.append(dict(lead_h=h, state=s, n=n, n_marg=km, n_below=kb, p_marg=pm, p_marg_lo=pm_lo, p_marg_hi=pm_hi, p_below=pb, p_below_lo=pb_lo, p_below_hi=pb_hi))
    return pd.DataFrame(out)

if __name__ == '__main__':
    D = build_days(); D.to_parquet('data/nowcast_days.parquet')
    t = table(D); t.to_csv('data/nowcast_table.csv', index=False)
    print(f'days {len(D)}  {D.index[0].date()}..{D.index[-1].date()}  below minimums at arrival {int((D.worst == 2).sum())}')
    print(t.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
