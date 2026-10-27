"""Poll Advanced Air's public flight-status feed (FlightAware-backed) and log CEC flights, attaching the KCEC METAR nearest
the scheduled arrival once the flight is in the past. Run a few times a day via cron/launchd. Updates data/flight_log.csv."""
import requests, pandas as pd, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); LOG = 'data/flight_log.csv'
j = requests.get('https://advancedairlines.com/wp-json/advanced-air/v1/flight-status', params={'_': int(pd.Timestamp.now().timestamp())}, headers={'Cache-Control': 'no-cache'}, timeout=30).json()
new = pd.DataFrame([f for f in j['flights'] if 'CEC' in (f['origin'], f['destination'])]); new['seen'] = j['generated']
old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=['ident', 'scheduled_out', 'metar'])
new['metar'] = new.set_index(['ident', 'scheduled_out']).index.map(old.set_index(['ident', 'scheduled_out']).metar.to_dict())   # keep a METAR already attached
df = pd.concat([old, new]).drop_duplicates(['ident', 'scheduled_out'], keep='last').reset_index(drop=True)
past = df[(df.destination=='CEC') & (pd.to_datetime(df.scheduled_out, utc=True) < pd.Timestamp.now('UTC')) & df.metar.isna()]
for i, r in past.iterrows():
    try:
        t = pd.to_datetime(r.actual_on if isinstance(r.actual_on, str) else r.scheduled_out, utc=True)
        u = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=CEC&data=metar&year1={t.year}&month1={t.month}&day1={t.day}&year2={t.year}&month2={t.month}&day2={t.day}&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4"
        m = pd.read_csv(u); m['valid'] = pd.to_datetime(m.valid, utc=True)
        if len(m): df.loc[i, 'metar'] = m.iloc[(m.valid - t).abs().argsort().iloc[0]].metar
    except Exception as e: print('metar attach failed:', e)
df.to_csv(LOG, index=False); print(f'{len(new)} CEC flights in feed now; log has {len(df)} rows')
