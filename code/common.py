"""Shared definitions for the OAK->CEC flight-risk model: weather tiers, schedule, forecast features, interval helpers."""
import os, re
from fractions import Fraction
import numpy as np, pandas as pd
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TZ = 'America/Los_Angeles'
LAT, LON = 41.78, -124.237                      # KCEC
MODELS = ['ecmwf_ifs025', 'gfs_seamless', 'gem_seamless', 'icon_seamless']
VARS = ['temperature_2m', 'dew_point_2m', 'relative_humidity_2m', 'cloud_cover',
        'wind_speed_10m', 'wind_direction_10m', 'weather_code', 'surface_pressure']
ARRIVAL_WINDOW = (pd.Timedelta(minutes=-45), pd.Timedelta(minutes=60))   # METARs scored around the scheduled arrival
FEED = 'https://advancedairlines.com/wp-json/advanced-air/v1/'

# Weather tiers against the KCEC ILS/LPV 12 minimums (DA 250 ft HAT, 3/4 SM).
#   2 below minimums: ceiling < 300 ft or visibility < 3/4 mi
#   1 marginal:       ceiling <= 500 ft or visibility <= 1.5 mi
#   0 otherwise
TIER_LABEL = {0: 'clear', 1: 'low cloud, landable', 2: 'below minimums'}

def tier(ceil_ft, vis_mi):
    c = 99999 if pd.isna(ceil_ft) else ceil_ft
    v = 10 if pd.isna(vis_mi) else vis_mi
    if c < 300 or v < 0.75: return 2
    if c <= 500 or v <= 1.5: return 1
    return 0

def tier_series(ceil_ft, vis_mi):
    c = pd.Series(ceil_ft).fillna(99999).to_numpy(); v = pd.Series(vis_mi).fillna(10).to_numpy()
    return np.where((c < 300) | (v < 0.75), 2, np.where((c <= 500) | (v <= 1.5), 1, 0))

def load_metars():
    """KCEC METAR archive with a local-time index and a tier column."""
    m = pd.read_parquet(os.path.join(ROOT, 'data/cec_metar.parquet'))
    m['tier'] = tier_series(m.ceil, m.vsby)
    return m.set_index('local').sort_index()

def worst_tier(ml, T):
    """Worst tier in the arrival window around local timestamp T; NaN if no report."""
    w = ml.loc[T + ARRIVAL_WINDOW[0]: T + ARRIVAL_WINDOW[1]]
    return w.tier.max() if len(w) else np.nan

def parse_metar(met):
    """(ceiling_ft, visibility_mi) from a raw METAR/SPECI; missing -> (None, None)."""
    mv = re.search(r'\s(M?)((?:\d+ )?\d+(?:/\d+)?)SM', met)
    vis = None
    if mv: vis = 0.0 if mv.group(1) else float(sum(Fraction(x) for x in mv.group(2).split()))
    cl = [int(x) * 100 for x in re.findall(r'(?:BKN|OVC|VV)(\d{3})', met)]
    return (min(cl) if cl else None), vis

def arrival_ts(date, hhmm):
    """Local tz-aware arrival timestamp; built from the wall-clock string so DST transitions are handled."""
    return pd.Timestamp(f'{pd.Timestamp(date).date()} {hhmm}').tz_localize(TZ)

def default_arrival(d):
    """Scheduled OAK->CEC arrival (HH:MM) by era, for days without a reconstructed flight record.
    Eras from the airline's schedule feed and Wayback captures (Mar/Jun/Oct 2025, Feb/Jun 2026); a timetable change
    needs a new line here and a rebuild of data/flight_outcomes.csv and the models."""
    d = pd.Timestamp(d)
    if d.dayofweek == 6:
        if d < pd.Timestamp('2024-10-01'): return '14:06'
        if d < pd.Timestamp('2026-06-28'): return '19:06'
        if d < pd.Timestamp('2026-10-01'): return '22:24'
        return '18:42'
    return '18:06' if d < pd.Timestamp('2026-10-01') else '18:42'

NOWCAST_STATES = ('0', 'lowcld', '1', '2')
NOWCAST_STATE_DESC = {'0': 'clear or high cloud', 'lowcld': 'low cloud 600-1000 ft or vis <= 3 mi', '1': 'ceiling <= 500 ft or vis <= 1.5 mi', '2': 'below minimums'}

def nowcast_state(ceil_ft, vis_mi):
    """Airport state for the persistence table: the tier when marginal or worse, else 'lowcld' or '0'."""
    t = tier(ceil_ft, vis_mi)
    if t: return str(t)
    c = 99999 if pd.isna(ceil_ft) else ceil_ft; v = 10 if pd.isna(vis_mi) else vis_mi
    return 'lowcld' if (c <= 1000 or v <= 3) else '0'

def jeffreys(k, n):
    """(mean, lo, hi, k, n): posterior mean and 95% interval for a binomial rate under the Jeffreys prior."""
    mean = (k + 0.5) / (n + 1)
    lo, hi = beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5) if n else (np.nan, np.nan)
    return float(mean), float(lo), float(hi), int(k), int(n)

def window_features(df, T, prefix=''):
    """df: hourly frame indexed by UTC time with columns VARS. T: target UTC Timestamp (arrival).
    Window = T-2h..T+1h (approach and arrival); day = preceding afternoon through late evening."""
    T = T.floor('h')
    win = df.loc[T - pd.Timedelta(hours=2): T + pd.Timedelta(hours=1)]
    day = df.loc[T - pd.Timedelta(hours=14): T + pd.Timedelta(hours=6)]
    if len(win) < 3 or win['temperature_2m'].isna().all(): return None
    t, td = win['temperature_2m'], win['dew_point_2m']
    wd = win['wind_direction_10m'].mean()
    f = {
        'spread_min': (t - td).min(), 'spread_mean': (t - td).mean(),
        'rh_max': win['relative_humidity_2m'].max(),
        'cc_mean': win['cloud_cover'].mean(), 'cc_day_mean': day['cloud_cover'].mean(),
        'ws': win['wind_speed_10m'].mean(),
        'wd_sin': np.sin(np.deg2rad(wd)), 'wd_cos': np.cos(np.deg2rad(wd)),
        'wx_fog': float((win['weather_code'] == 45).any()), 'wx_max': win['weather_code'].max(),
        'psfc': win['surface_pressure'].mean(),
        'tmax_day': day['temperature_2m'].max(), 't_eve': t.mean(),
        'spread_day_min': (day['temperature_2m'] - day['dew_point_2m']).min(),
    }
    return {prefix + k: v for k, v in f.items()}

def add_season(X):
    X = X.copy()
    X['month_sin'] = np.sin(2 * np.pi * X.index.month / 12); X['month_cos'] = np.cos(2 * np.pi * X.index.month / 12)
    return X

def archived_features(lead, T_utc_by_day):
    """Feature matrix at a given lead from the Open-Meteo previous-runs archive, one row per day in T_utc_by_day."""
    Fs = []
    for mm in MODELS:
        pr = pd.read_parquet(os.path.join(ROOT, f'data/prevruns_{mm}.parquet')).set_index('time')
        cmap = {f'{v}_previous_day{lead}': v for v in VARS}
        if not all(c in pr.columns for c in cmap): continue
        pr = pr[list(cmap)].rename(columns=cmap)
        rows = {d: window_features(pr, t, mm[:4] + '_') for d, t in T_utc_by_day.items()}
        F = pd.DataFrame({d: v for d, v in rows.items() if v}).T
        if F.notna().mean().mean() < 0.5: continue
        Fs.append(F)
    X = add_season(pd.concat(Fs, axis=1)); return X.loc[:, X.notna().mean() > 0.5]
