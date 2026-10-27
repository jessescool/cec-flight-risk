"""Go/no-go check for the OAK->CEC flight on a given day, from the forecasts available now.

    python3 code/weekly_check.py [--date YYYY-MM-DD] [--yesterday-cancelled]

Default target is the next Sunday. The arrival time comes from the airline's published schedule for that day; the model
for the matching lead (1-7 days) gives P(low cloud) and P(below minimums) at arrival, and the crew-stage rates
(code/outcome.py) turn those into P(you do not land at CEC that evening). Within 1.5 h of arrival a persistence nowcast
from the latest METAR is shown alongside; beyond that the forecast scores better (data/nowcast_vs_ensemble.txt).
Targets in the past or more than 7.5 days out are refused. Every run appends one row to data/predictions.csv."""
import argparse, os, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, TZ, LAT, LON, VARS, FEED, NOWCAST_STATE_DESC, window_features, add_season, parse_metar, nowcast_state, arrival_ts, default_arrival
from outcome import mixture
os.chdir(ROOT)
LABEL_W = 52                                             # label column width in the report
NOWCAST_HOURS = (0.5, 1.5)                               # window in which persistence beats the forecast (data/nowcast_vs_ensemble.txt)
UPSTREAM_LATE_MIN = 90                                   # an inbound leg this late counts as a disruption

def feed(endpoint):
    return requests.get(FEED + endpoint, params={'_': int(pd.Timestamp.now().timestamp())}, headers={'Cache-Control': 'no-cache'}, timeout=30).json()

def scheduled_arrival(day):
    """(arrival HH:MM, flight ident, warning) for the latest OAK->CEC arrival on `day` from the published schedule."""
    try:
        r = next(x for x in feed('flight-schedules')['routes'] if x['slug'] == 'cec-to-oak')['meta']
    except Exception as e:
        return default_arrival(day), '?', f'WARNING schedule fetch failed ({e}); arrival time is the era default'
    for suf in ['', '_upcoming', '_upcoming_2', '_upcoming_3']:
        sd, ed, blk = r.get('schedule_start_date' + suf), r.get('schedule_end_date' + suf), r.get(f'{day.day_name().lower()}_flights' + suf)
        if sd and ed and sd <= day.strftime('%Y-%m-%d') <= ed and isinstance(blk, dict):
            inbound = sorted((v['arrival_time'], 'AN' + v['flight_number']) for v in blk.values() if v['flight_origin'] == 'OAK')
            return (inbound[-1][0], inbound[-1][1], '') if inbound else (None, None, 'no OAK->CEC flight scheduled')
    return default_arrival(day), '?', 'WARNING day not covered by the published schedule; arrival time is the era default'

def fetch_forecasts(Tu, models):
    """Per-model hourly forecasts and the feature row at arrival time Tu (UTC); returns (features, frames, warnings)."""
    feats, H, warn = {}, {}, []
    for m in models:
        try:
            j = requests.get('https://api.open-meteo.com/v1/forecast', params=dict(latitude=LAT, longitude=LON, hourly=','.join(VARS + ['cloud_cover_low']), models=m, forecast_days=10, timezone='UTC'), timeout=60).json()
            h = pd.DataFrame(j['hourly']); h['time'] = pd.to_datetime(h['time'], utc=True); h = h.set_index('time')
            f = window_features(h, Tu, m[:4] + '_')
            if f: feats.update(f); H[m] = h
            else: warn.append(f'{m}: no data at arrival time')
        except Exception as e: warn.append(f'{m}: {e}')
    return feats, H, warn

def print_forecast(H, T):
    Tu = T.tz_convert('UTC').floor('h')
    for m, h in H.items():
        sp = h.loc[Tu - pd.Timedelta(hours=2): Tu + pd.Timedelta(hours=1)]
        print(f"  {m:14s} T-Td min {(sp.temperature_2m - sp.dew_point_2m).min():.1f}C  RH max {sp.relative_humidity_2m.max():.0f}%  cloud {sp.cloud_cover.mean():.0f}%  wind {sp.wind_speed_10m.mean():.0f} km/h from {sp.wind_direction_10m.mean():.0f}")
    print(f'Forecast for {T:%a %d %b} (range across models):')
    for lo, hi, label in [(5, 9, 'morning 05-09'), (11, 15, 'midday 11-15'), (T.hour - 1, T.hour + 1, f'arrival {T.hour - 1:02d}-{T.hour + 1:02d}')]:
        sp, rh, lc, fog = [], [], [], 0
        for h in H.values():
            w = h.tz_convert(TZ); w = w[(w.index.date == T.date()) & (w.index.hour >= lo) & (w.index.hour <= hi)]
            if w.empty: continue
            sp.append((w.temperature_2m - w.dew_point_2m).min()); rh.append(w.relative_humidity_2m.max()); lc.append(w.cloud_cover_low.mean()); fog += int((w.weather_code == 45).any())
        if sp: print(f"  {label:22s} T-Td {min(sp):.1f}-{max(sp):.1f}C, RH {min(rh):.0f}-{max(rh):.0f}%, low cloud {min(lc):.0f}-{max(lc):.0f}%, fog in {fog}/{len(sp)} models")

def previous_evening(T, log, forced_cancelled=False):
    """1 if the previous evening's OAK->CEC flight did not land at CEC (cancelled or diverted), 0 if it landed, None if unknown."""
    if forced_cancelled: return 1
    if log is None or log.empty: return None
    L = log[(log.origin == 'OAK') & (log.destination == 'CEC')]
    L = L[pd.to_datetime(L.scheduled_out, utc=True).dt.tz_convert(TZ).dt.date == (T - pd.Timedelta(days=1)).date()]
    if L.empty: return None
    st = L.status.astype(str)
    if L.cancelled.astype(str).eq('True').any() or st.str.contains('ancel|ivert').any(): return 1
    if st.str.contains('Arrived|Landed').any(): return 0
    return None

def same_day_feed(T, flight):
    """Lines describing today's legs into OAK and this flight, plus a flag: 'cancelled', 'disrupted' or ''."""
    try:
        legs = feed('flight-status')['flights']
        ts = lambda f: pd.to_datetime(f['scheduled_out'], utc=True)
        today = [f for f in legs if ts(f).tz_convert(TZ).date() == T.date()]
        mine = [f for f in today if f['origin'] == 'OAK' and f['destination'] == 'CEC' and (f['ident'] == flight or flight == '?')] or [f for f in today if f['origin'] == 'OAK' and f['destination'] == 'CEC']
        cutoff = ts(mine[0]) if mine else T.tz_convert('UTC')
        up = [f for f in today if f['destination'] == 'OAK' and ts(f) < cutoff]
        def delay(f):
            e = f.get('estimated_out') or f.get('actual_out') or f['scheduled_out']
            return (pd.to_datetime(e, utc=True) - ts(f)).total_seconds() / 60
        lines = [f"  {f['ident']:7s} {f['origin']}->{f['destination']} sched {ts(f).tz_convert(TZ):%H:%M}  {str(f.get('status') or '?'):10s} delay {delay(f):+.0f} min{'  CANCELLED' if f.get('cancelled') else ''}"
                 for f in sorted(up + mine, key=lambda f: f['scheduled_out'])]
        flag = 'cancelled' if (mine and mine[0].get('cancelled')) else 'disrupted' if any(f.get('cancelled') or delay(f) > UPSTREAM_LATE_MIN for f in up + mine) else ''
        return lines, flag
    except Exception as e: return [f'status feed failed: {e}'], ''

def fetch_nbm(now, Tu):
    """One line of National Blend ceiling/visibility guidance for the arrival hour, or '' (text bulletins reach 72 h)."""
    try:
        for back in range(4):
            cyc = (now.tz_convert('UTC') - pd.Timedelta(hours=back * 6)).floor('6h')
            r = requests.get(f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{cyc:%Y%m%d}/{cyc:%H}/text/blend_nbstx.t{cyc:%H}z", timeout=120, stream=True); r.encoding = 'utf-8'
            if r.status_code != 200: continue
            blk, cap = [], False
            for line in r.iter_lines(decode_unicode=True):
                if line.startswith(' KCEC '): cap = True
                if cap:
                    if not line.strip(): break
                    blk.append(line)
            r.close()
            if not blk: continue
            rows = {l[1:4]: [l[5:][i:i + 3].strip() for i in range(0, len(l[5:]), 3)] for l in blk[1:]}
            fhr = [int(x) for x in rows['FHR'] if x]; tgt = int(round((Tu - cyc).total_seconds() / 3600))
            if tgt > max(fhr): continue
            i = int(np.argmin([abs(f - tgt) for f in fhr])); g = lambda k: (rows.get(k) or [''])[i] if i < len(rows.get(k) or []) else ''
            cig = 'unlimited' if g('CIG').startswith('-') else f"{g('CIG')}00 ft"
            return f"NBM {cyc:%d/%HZ} FHR{fhr[i]}: CIG {cig}, VIS {int(g('VIS') or 0) / 10:.1f} mi, P(IFR ceiling) {g('IFC')}%, P(IFR vis) {g('IFV')}%"
    except Exception as e: print(f'NBM fetch failed: {e}')
    return ''

def fetch_taf_metar(want_taf):
    taf = met = ''
    try:
        if want_taf:
            row = next((x for x in requests.get('https://mesonet.agron.iastate.edu/api/1/nws/taf_overview.json', timeout=30).json()['data'] if x['station'] == 'KCEC'), None)
            if row and row.get('raw'): taf = f"TAF issued {row.get('utc_issue')}: {row['raw'][:200]}"
    except Exception as e: print(f'TAF fetch failed: {e}')
    try:
        met = requests.get('https://aviationweather.gov/api/data/metar?ids=KCEC&format=raw', timeout=30).text.strip()
        if not met.startswith(('METAR', 'SPECI', 'KCEC')): met = ''
    except Exception as e: print(f'METAR fetch failed: {e}')
    return taf, met

def nowcast(met, hrs, table):
    """Persistence row for the current airport state and hours-to-arrival, or None outside NOWCAST_HOURS."""
    if table is None or not (NOWCAST_HOURS[0] <= hrs <= NOWCAST_HOURS[1]): return None
    ceil, vis = parse_metar(met)
    if vis is None: return None
    state = nowcast_state(ceil, vis); h = int(np.clip(round(hrs), 1, 5))
    z = table[(table.lead_h == h) & (table.state == state)]
    return None if z.empty else dict(z.iloc[0], ceil=ceil, vis=vis, h=h, desc=NOWCAST_STATE_DESC[state])

def log_prediction(row, path='data/predictions.csv'):
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path); new = pd.concat([old, new], ignore_index=True)[list(dict.fromkeys(list(old.columns) + list(row)))]
    new.to_csv(path, index=False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--date'); ap.add_argument('--yesterday-cancelled', action='store_true'); a = ap.parse_args()
    now = pd.Timestamp.now(tz=TZ)
    day = pd.Timestamp(a.date).tz_localize(TZ) if a.date else (now + pd.Timedelta(days=((6 - now.dayofweek) % 7) or 7)).normalize()
    arr, flight, note = scheduled_arrival(day)
    if arr is None: print(f'{day:%a %Y-%m-%d}: {note}'); return
    if note: print(note)
    T = arrival_ts(day, arr); Tu = T.tz_convert('UTC')
    lead_days = (Tu - now.tz_convert('UTC')).total_seconds() / 86400; lead = int(np.clip(np.ceil(lead_days), 1, 7)); hrs = lead_days * 24
    if lead_days < 0: print(f'{T:%a %Y-%m-%d %H:%M} has already passed'); return
    if lead_days > 7.5: print(f'{T:%a %Y-%m-%d} is {lead_days:.1f} days out; the models stop at 7 days. Re-run on or after {T - pd.Timedelta(days=7):%a %Y-%m-%d}.'); return
    if arr < '17:00': print(f'note: {arr} arrival; the crew-stage rates come from evening flights and overstate daytime fog risk')
    print(f"OAK->CEC {flight}  arr CEC {T:%a %Y-%m-%d %H:%M %Z}   issued {now:%a %Y-%m-%d %H:%M}   lead {lead_days:.1f} d -> lead-{lead} model")

    mdl = joblib.load(f'models/outcome_lead{lead}.joblib'); rates = mdl['rates']; metrics = mdl['metrics']
    feats, H, warn = fetch_forecasts(Tu, mdl['models_used'] if 'models_used' in mdl else mdl['models'])
    for w in warn: print('  WARNING', w)
    if not feats: print('no forecast data; stopping'); return
    print_forecast(H, T)
    X = add_season(pd.DataFrame([feats], index=[T.tz_localize(None)])).reindex(columns=mdl['columns']).fillna(mdl['median'])
    p_marg = float(mdl['wx']['marginal'].predict_proba(X)[0, 1]); p_below = p_marg * float(mdl['wx']['below_given_marginal'].predict_proba(X)[0, 1])
    ens_note = f"{len(H)}-model ensemble" + (f" (WARNING {len(mdl['models_used']) - len(H)} model(s) missing, features imputed with training medians)" if len(H) < len(mdl['models_used']) else '')

    # floor: previous-evening split only when that evening is over (lead 1); same-day feed can override
    log = pd.read_csv('data/flight_log.csv') if os.path.exists('data/flight_log.csv') else None
    prev = previous_evening(T, log, a.yesterday_cancelled) if lead_days < 1 else None
    floor = rates[{1: 'floor_prev_fail', 0: 'floor_prev_ok', None: 'floor'}[prev]]
    floor_note = {1: "previous evening's flight did not land", 0: "previous evening's flight landed", None: 'previous evening not used' if lead_days >= 1 else 'previous evening unknown'}[prev]
    upstream, flag = same_day_feed(T, flight) if lead_days < 1 else ([], '')
    if flag == 'disrupted' and floor[0] < rates['floor_prev_fail'][0]: floor = rates['floor_prev_fail']; floor_note = f'an inbound leg today is cancelled or >{UPSTREAM_LATE_MIN} min late; after-a-cancellation rate'
    p_noland = 1.0 if flag == 'cancelled' else mixture(p_marg, p_below, floor[0], rates)
    r1 = max(rates[1][0], floor[0]); r2 = max(rates[2][0], r1)

    nbm = fetch_nbm(now, Tu); taf, met = fetch_taf_metar(lead_days < 1.1)
    table = pd.read_csv('data/nowcast_table.csv', dtype={'state': str}) if os.path.exists('data/nowcast_table.csv') else None
    nc = nowcast(met, hrs, table); q_noland = np.nan
    ci = lambda r: f'{r[3]}/{r[4]}, 95% {r[1]:.3f}-{r[2]:.3f}'
    W = LABEL_W

    print()
    print(f"{'WEATHER AT THE AIRPORT AT ARRIVAL (forecast)':<{W}}{'p':>7}   basis")
    print(f"{'  P(low cloud: ceiling <= 500 ft or vis <= 1.5 mi)':<{W}}{p_marg:>7.3f}   logistic on {ens_note} features, {mdl['n_days']} days; out-of-sample AUC vs observed {metrics['tier1_auc']:.2f}")
    print(f"{'  P(below minimums: ceiling < 300 ft or vis < 3/4 mi)':<{W}}{p_below:>7.3f}   subset of the above; AUC vs observed {metrics['tier2_auc']:.2f}, vs weather-caused failures {metrics['wx_auc']:.2f} [{metrics['wx_auc_lo']:.2f}-{metrics['wx_auc_hi']:.2f}]")
    print()
    print(f"{'WHAT THE CREW DOES, GIVEN THAT WEATHER (past flights)':<{W}}{'P(sky)':>7}{'P(fail|sky)':>12}{'product':>9}   basis (failures/flights, 95% interval)")
    print(f"{'  clear':<{W}}{1 - p_marg:>7.3f}{floor[0]:>12.3f}{(1 - p_marg) * floor[0]:>9.3f}   {ci(floor)}; {floor_note}")
    print(f"{'  low cloud, landable':<{W}}{p_marg - p_below:>7.3f}{r1:>12.3f}{(p_marg - p_below) * r1:>9.3f}   {ci(rates[1])}{'; floored at the clear rate' if r1 > rates[1][0] else ''}")
    print(f"{'  below minimums':<{W}}{p_below:>7.3f}{r2:>12.3f}{p_below * r2:>9.3f}   {ci(rates[2])}; Sunday-only {rates['fog_sunday'][0]:.2f} ({rates['fog_sunday'][3]}/{rates['fog_sunday'][4]}) is a sensitivity, not shipped")
    print()
    print('YOUR ODDS')
    if flag == 'cancelled': print(f"{'  P(you do not land at CEC that evening)':<{W}}{'':>19}{1.0:>9.3f}   THIS FLIGHT IS SHOWN CANCELLED in the airline feed")
    else: print(f"{'  P(you do not land at CEC that evening)':<{W}}{'':>19}{p_noland:>9.3f}   sum of the product column; {mdl['n_fail']} failures / {mdl['n_flights']} flights; out-of-sample AUC {metrics['auc']:.2f}, Brier skill {metrics['bss']:+.2f}")
    rb = next((r for r in mdl.get('reliability', []) if r['lo'] < p_noland <= r['hi']), None)
    if rb and flag != 'cancelled': print(f"{'  calibration check':<{W}}{'':>28}   past flights scored {rb['lo']:.2f}-{rb['hi']:.2f} out of sample: {rb['failures']}/{rb['n']} did not land ({rb['observed']:.2f} observed vs {rb['predicted']:.2f} predicted; small n)")
    if nc:
        q_noland = mixture(float(nc['p_marg']), float(nc['p_below']), floor[0], rates)
        ceil_txt = 'none' if nc['ceil'] is None else f"{nc['ceil']} ft"
        print()
        print(f"{'NOWCAST FROM THE AIRPORT NOW (' + f'{hrs:.1f} h before arrival)':<{W}}{'p':>7}   basis (2019-2026 METAR archive, days in this state {nc['h']} h out)")
        print(f"{'  airport now: ' + nc['desc']:<{W}}{'':>7}   ceiling {ceil_txt}, vis {nc['vis']:g} mi")
        print(f"{'  P(low cloud at arrival)':<{W}}{nc['p_marg']:>7.3f}   {int(nc['n_marg'])}/{int(nc['n'])}, 95% {nc['p_marg_lo']:.3f}-{nc['p_marg_hi']:.3f}")
        print(f"{'  P(below minimums at arrival)':<{W}}{nc['p_below']:>7.3f}   {int(nc['n_below'])}/{int(nc['n'])}, 95% {nc['p_below_lo']:.3f}-{nc['p_below_hi']:.3f}")
        print(f"{'  P(you do not land at CEC), nowcast':<{W}}{q_noland:>7.3f}   same crew rates; the forecast-based figure above is {p_noland:.3f}")
    print()
    if upstream: print("Today's legs into OAK before this flight, and this flight:"); print('\n'.join(upstream)); print()
    for s in (nbm, taf):
        if s: print(s)
    print('Latest METAR:', met or 'unavailable')
    log_prediction(dict(run_time=now.isoformat(), target=T.isoformat(), flight=flight, lead_days=round(lead_days, 2), lead_model=lead, models_returned=len(H), p_marginal=round(p_marg, 3), p_below=round(p_below, 3),
                        floor=round(floor[0], 3), prev_evening=prev, feed_flag=flag, p_noland=round(p_noland, 3), nowcast_state=nc['state'] if nc else '', nowcast_p_noland=round(q_noland, 3) if nc else np.nan, nbm=nbm, taf=taf))

if __name__ == '__main__': main()
