import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from common import (ROOT, TZ, tier, tier_series, parse_metar, arrival_ts, default_arrival, jeffreys, window_features, worst_tier, nowcast_state)

@pytest.mark.parametrize('ceil,vis,expected', [
    (299, 10, 2), (300, 10, 1), (500, 10, 1), (501, 10, 0),
    (None, 0.74, 2), (None, 0.75, 1), (None, 1.5, 1), (None, 1.6, 0),
    (None, None, 0), (np.nan, np.nan, 0), (200, 10, 2), (5000, 0.5, 2),
])
def test_tier_boundaries(ceil, vis, expected):
    assert tier(ceil, vis) == expected

def test_tier_series_matches_scalar():
    c = pd.Series([299, 300, 501, np.nan]); v = pd.Series([10, 10, 10, 0.5])
    assert list(tier_series(c, v)) == [tier(a, b) for a, b in zip(c, v)]

@pytest.mark.parametrize('met,expected', [
    ('SPECI KCEC 270038Z AUTO 16005KT 1 3/4SM BR BKN003 OVC006 12/11 A3016', (300, 1.75)),
    ('KCEC 270247Z AUTO 00000KT M1/4SM FG VV001 10/09 A3017', (100, 0.0)),
    ('METAR KCEC 262056Z AUTO 19009KT 10SM CLR 14/12 A3018', (None, 10.0)),
    ('METAR KCEC 262356Z AUTO 17003KT 1/2SM FG VV002 12/11 A3016', (200, 0.5)),
    ('SPECI KCEC 262329Z AUTO 18005KT 3SM BR SCT003 13/12 A3017', (None, 3.0)),
    ('METAR KCEC 270256Z AUTO 00000KT 7SM SCT016 09/09 A3017 RMK AO2', (None, 7.0)),
    ('METAR KCEC 261756Z AUTO 17007KT 10SM FEW012 FEW036 BKN070 13/12', (7000, 10.0)),
    ('METAR fetch failed: timeout', (None, None)),
])
def test_parse_metar(met, expected):
    assert parse_metar(met) == expected

def test_arrival_ts_handles_dst():
    assert arrival_ts('2026-11-01', '18:42').utcoffset() == pd.Timedelta(hours=-8)    # first day after fall-back
    assert arrival_ts('2026-10-31', '18:42').utcoffset() == pd.Timedelta(hours=-7)
    assert arrival_ts('2026-11-01', '18:42').strftime('%H:%M') == '18:42'

def test_default_arrival_eras():
    assert default_arrival('2026-10-26') == '18:42'       # Monday, current timetable
    assert default_arrival('2026-06-28') == '22:24'       # Sunday, summer 2026
    assert default_arrival('2025-03-02') == '19:06'       # Sunday, 2024-2026 timetable
    assert default_arrival('2025-03-03') == '18:06'       # Monday, 2024-2026 timetable

def test_jeffreys():
    m, lo, hi, k, n = jeffreys(9, 40)
    assert (k, n) == (9, 40)
    assert abs(m - 9.5 / 41) < 1e-9 and lo < m < hi and 0 < lo and hi < 1
    m0, lo0, hi0, _, _ = jeffreys(0, 10)
    assert lo0 < 0.01 and hi0 < 0.3

def _hourly(n=24, spread=2.0):
    idx = pd.date_range('2026-10-26 00:00', periods=n, freq='h', tz='UTC')
    df = pd.DataFrame(index=idx)
    df['temperature_2m'] = 12.0; df['dew_point_2m'] = 12.0 - spread; df['relative_humidity_2m'] = 90.0
    df['cloud_cover'] = 50.0; df['wind_speed_10m'] = 5.0; df['wind_direction_10m'] = 180.0
    df['weather_code'] = 3; df['surface_pressure'] = 1015.0
    return df

def test_window_features_values():
    df = _hourly(); T = pd.Timestamp('2026-10-26 18:00', tz='UTC')
    df.loc[T, 'weather_code'] = 45
    f = window_features(df, T, 'x_')
    assert f['x_spread_min'] == pytest.approx(2.0) and f['x_wx_fog'] == 1.0 and f['x_wd_sin'] == pytest.approx(0, abs=1e-9)

def test_window_features_requires_coverage():
    df = _hourly(n=3)
    assert window_features(df, pd.Timestamp('2026-10-27 18:00', tz='UTC')) is None

def test_worst_tier_uses_arrival_window():
    idx = pd.to_datetime(['2026-10-26 17:30', '2026-10-26 18:30', '2026-10-26 20:00']).tz_localize(TZ)
    ml = pd.DataFrame({'tier': [0, 2, 0]}, index=idx)
    assert worst_tier(ml, arrival_ts('2026-10-26', '18:42')) == 2
    assert worst_tier(ml, arrival_ts('2026-10-26', '20:30')) == 0
    assert np.isnan(worst_tier(ml, arrival_ts('2026-10-27', '18:42')))

@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'data/nowcast_table.csv')), reason='nowcast table not built')
def test_nowcast_table_complete():
    t = pd.read_csv(os.path.join(ROOT, 'data/nowcast_table.csv'), dtype={'state': str})
    for h in range(1, 6):
        for s in ('0', 'lowcld', '1', '2'):
            z = t[(t.lead_h == h) & (t.state == s)]
            assert len(z) == 1, (h, s)
            z = z.iloc[0]; assert z.n_below <= z.n_marg <= z.n and 0 < z.p_below <= z.p_marg < 1
    assert t[(t.lead_h == 1) & (t.state == '2')].p_below.iloc[0] > t[(t.lead_h == 1) & (t.state == '0')].p_below.iloc[0]

@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'models/outcome_lead1.joblib')), reason='model not trained')
def test_model_bundle_consistent():
    import joblib
    for lead in range(1, 8):
        m = joblib.load(os.path.join(ROOT, f'models/outcome_lead{lead}.joblib'))
        assert set(m) >= {'wx', 'rates', 'columns', 'median', 'models', 'metrics', 'n_flights', 'n_fail'}
        for k in (0, 1, 2, 'floor', 'floor_prev_ok', 'floor_prev_fail', 'fog_conf', 'fog_surprise'):
            mean, lo, hi, kk, n = m['rates'][k]; assert 0 <= lo <= mean <= hi <= 1 and kk <= n
        assert m['rates'][2][0] > m['rates'][0][0]            # fog must raise the failure rate
        assert m['rates']['fog_conf'][0] >= m['rates']['fog_surprise'][0]
        assert 0.5 < m['metrics']['wx_auc'] <= 1
        assert sum(r['n'] for r in m['reliability']) == m['metrics']['n'] and all(r['failures'] <= r['n'] for r in m['reliability'])

# ---- crew stage and live-check helpers
from outcome import crew_rates, mixture, mixture_rows, floor_for
import weekly_check as wc

def _flights():
    idx = pd.date_range('2025-01-01', periods=400, freq='D')
    rng = np.random.default_rng(1)
    wx = rng.choice([0, 1, 2], size=400, p=[0.8, 0.12, 0.08])
    p2 = np.where(wx == 2, rng.random(400) * 0.8, rng.random(400) * 0.2)
    fail = np.where(wx == 2, rng.random(400) < np.where(p2 >= 0.3, 0.5, 0.1), rng.random(400) < 0.03).astype(int)   # forecastable fog fails more
    fl = pd.DataFrame({'wx_worst': wx, 'fail': fail, 'sched_arr': '18:42', 'p2_oos': p2}, index=idx)
    fl['prev_fail'] = fl.fail.shift(1)
    return fl

def test_mixture_is_monotone_in_weather():
    r = crew_rates(_flights()); floor = r['floor'][0]
    r[1] = (0.001, 0.0, 0.01, 0, 50)                      # force a low-cloud rate below the floor, as in the real data
    p_clear = mixture(0.0, 0.0, floor, r); p_cloud = mixture(0.3, 0.0, floor, r); p_fog = mixture(0.3, 0.3, floor, r)
    assert p_clear <= p_cloud <= p_fog and abs(p_clear - floor) < 1e-12

def test_mixture_rows_matches_scalar_and_prev_flag():
    r = crew_rates(_flights())
    rows = mixture_rows(np.array([0.2, 0.2]), np.array([0.1, 0.1]), r, [1, 0])
    assert rows[0] == mixture(0.2, 0.1, floor_for(r, 1), r) and rows[1] == mixture(0.2, 0.1, floor_for(r, 0), r)
    assert floor_for(r, np.nan) == r['floor'][0]

def test_fog_rate_conditioned_on_confidence():
    from outcome import fog_rate, FOG_CONF_CUT
    r = crew_rates(_flights())
    assert r['fog_conf'] != r['fog_surprise'] and fog_rate(r, FOG_CONF_CUT) == r['fog_conf'][0] and fog_rate(r, FOG_CONF_CUT - 0.01) == r['fog_surprise'][0]
    assert fog_rate(r, 0.9, 2) == r[2][0]
    fl = _flights().drop(columns='p2_oos'); r0 = crew_rates(fl); assert r0['fog_conf'] == r0['fog_surprise'] == r0[2]   # no out-of-sample column -> pooled
    fl = _flights(); fl.loc[fl.wx_worst == 2, 'p2_oos'] = np.nan; fl.loc[fl[fl.wx_worst == 2].index[:3], 'p2_oos'] = 0.5
    r3 = crew_rates(fl); assert r3['fog_conf'] == r3[2]                                                           # too few rows -> pooled
    fl = _flights(); fl.loc[fl.wx_worst == 2, 'fail'] = (fl.loc[fl.wx_worst == 2, 'p2_oos'] < 0.3).astype(int)       # inverted effect -> pooled
    r4 = crew_rates(fl); assert r4['fog_conf'] == r4['fog_surprise'] == r4[2]

def test_crew_rates_with_no_previous_failures_falls_back():
    fl = _flights(); fl['prev_fail'] = 0.0
    r = crew_rates(fl); assert r['floor_prev_fail'] == r['floor']

def _log(status, cancelled='False', sched='2026-10-27T00:31:00Z'):
    return pd.DataFrame([dict(ident='AN92', origin='OAK', destination='CEC', scheduled_out=sched, status=status, cancelled=cancelled, actual_on='2026-10-27T02:37:05Z')])

def test_previous_evening_flags():
    T = arrival_ts('2026-10-27', '18:42')
    assert wc.previous_evening(T, _log('Arrived')) == 0
    assert wc.previous_evening(T, _log('Diverted')) == 1                        # a diversion is a failure in training, so here too
    assert wc.previous_evening(T, _log('Scheduled', cancelled='True')) == 1
    assert wc.previous_evening(T, _log('En Route / Delayed')) is None
    assert wc.previous_evening(T, _log('Arrived', sched='2026-10-26T00:31:00Z')) is None   # wrong day
    assert wc.previous_evening(T, None) is None and wc.previous_evening(T, None, forced_cancelled=True) == 1

def test_nowcast_lookup_and_window():
    table = pd.DataFrame([dict(lead_h=1, state='1', n=80, n_marg=71, n_below=29, p_marg=0.88, p_marg_lo=0.8, p_marg_hi=0.94, p_below=0.36, p_below_lo=0.26, p_below_hi=0.47)])
    met = 'SPECI KCEC 270038Z AUTO 16005KT 1 3/4SM BR BKN003 OVC006 12/11 A3016'
    r = wc.nowcast(met, 1.2, table); assert r and r['state'] == '1' and r['ceil'] == 300
    assert wc.nowcast(met, 3.0, table) is None and wc.nowcast(met, 1.5, table) is None   # outside the window where persistence wins
    assert wc.nowcast('METAR fetch failed', 1.0, table) is None and wc.nowcast(met, 1.0, None) is None

@pytest.mark.parametrize('ceil,vis,state', [(None, 10, '0'), (800, 10, 'lowcld'), (None, 3, 'lowcld'), (500, 10, '1'), (200, 10, '2'), (5000, 0.5, '2')])
def test_nowcast_state(ceil, vis, state):
    assert nowcast_state(ceil, vis) == state

def test_window_features_minimum_rows():
    df = _hourly(n=24); T = pd.Timestamp('2026-10-26 22:00', tz='UTC')     # window 20:00-23:00 -> rows 20,21,22,23 = 4
    assert window_features(df.iloc[:22], T) is None                        # rows 20,21 only -> 2 < 3
    assert window_features(df.iloc[:23], T) is not None                    # rows 20,21,22 -> 3

def test_log_prediction_keeps_old_columns(tmp_path):
    p = tmp_path / 'pred.csv'
    pd.DataFrame([dict(run_time='a', old_col=1, p_noland=0.1)]).to_csv(p, index=False)
    wc.log_prediction(dict(run_time='b', p_noland=0.2, new_col=3), str(p))
    d = pd.read_csv(p); assert list(d.columns) == ['run_time', 'old_col', 'p_noland', 'new_col'] and len(d) == 2


def test_floor_halves_share_one_window():
    fl = _flights(); r = crew_rates(fl)
    recent = fl[(fl.wx_worst == 0) & (fl.index > fl.index.max() - pd.Timedelta(days=365))]
    assert r['floor_prev_ok'][4] + r['floor_prev_fail'][4] == int(recent.prev_fail.isin([0, 1]).sum()) and r['floor'][4] == len(recent)

def test_crew_rates_ignore_rows_without_oos_prediction():
    fl = _flights(); r_all = crew_rates(fl)
    fl2 = fl.copy(); fl2.loc[fl2.index[-60:], 'p2_oos'] = np.nan          # a quarter not yet scored carries NaN
    r = crew_rates(fl2); assert r['fog_conf'][4] + r['fog_surprise'][4] == int(fl2[fl2.wx_worst == 2].p2_oos.notna().sum()) < r_all['fog_conf'][4] + r_all['fog_surprise'][4]

SCHED = {'routes': [{'slug': 'cec-to-oak', 'meta': {'schedule_start_date': '2026-10-01', 'schedule_end_date': '2027-05-29',
         'tuesday_flights': {'a': {'flight_origin': 'OAK', 'arrival_time': '18:42', 'flight_number': '3008'}, 'b': {'flight_origin': 'CEC', 'arrival_time': '08:56', 'flight_number': '3001'}},
         'thursday_flights': {'a': {'flight_origin': 'CEC', 'arrival_time': '08:56', 'flight_number': '3001'}}}}]}
STATUS = {'flights': [dict(ident='AN3001', origin='CEC', destination='OAK', scheduled_out='2026-10-27T15:00:00Z', status='Arrived', cancelled=False),
                      dict(ident='AN3008', origin='OAK', destination='CEC', scheduled_out='2026-10-28T00:36:00Z', status='Scheduled', cancelled=False, estimated_out='2026-10-28T00:36:00Z')]}

def test_scheduled_arrival_from_feed(monkeypatch):
    monkeypatch.setattr(wc, 'feed', lambda ep: SCHED)
    assert wc.scheduled_arrival(pd.Timestamp('2026-10-27', tz=TZ)) == ('18:42', 'AN3008', '')
    assert wc.scheduled_arrival(pd.Timestamp('2026-10-29', tz=TZ))[0] is None                       # Thursday: no inbound flight
    arr, flight, note = wc.scheduled_arrival(pd.Timestamp('2027-07-06', tz=TZ)); assert arr == default_arrival('2027-07-06') and note.startswith('WARNING')
    monkeypatch.setattr(wc, 'feed', lambda ep: (_ for _ in ()).throw(RuntimeError('down')))
    arr, flight, note = wc.scheduled_arrival(pd.Timestamp('2026-10-27', tz=TZ)); assert arr == '18:42' and flight == '?' and 'WARNING' in note

def test_same_day_feed_flags(monkeypatch):
    T = arrival_ts('2026-10-27', '18:42')
    monkeypatch.setattr(wc, 'feed', lambda ep: STATUS)
    lines, flag = wc.same_day_feed(T, 'AN3008'); assert flag == '' and len(lines) == 2 and 'AN3001' in lines[0]
    c = {'flights': [dict(STATUS['flights'][1], cancelled=True)]}; monkeypatch.setattr(wc, 'feed', lambda ep: c)
    assert wc.same_day_feed(T, 'AN3008')[1] == 'cancelled'
    monkeypatch.setattr(wc, 'feed', lambda ep: (_ for _ in ()).throw(RuntimeError('down')))
    assert wc.same_day_feed(T, 'AN3008')[1] == 'unknown'

def test_previous_evening_ignores_daytime_flights():
    T = arrival_ts('2026-10-27', '18:42')
    assert wc.previous_evening(T, _log('Diverted', sched='2026-10-26T18:00:00Z')) is None        # 11:00 local departure the day before
