"""Reconstruct Advanced Air approaches to CEC from adsb.lol traces despite no low-altitude coverage at CEC.
A 'CEC visit' = trace point within 150 km of CEC, descending/inbound, followed by a coverage gap, then reappearance.
Gap >= 45 min with reappearance outbound from CEC => landed and later departed (or overnight).
Gap < 30 min and reappearance near CEC => missed approach / hold then diversion; reappearance near MFR/ACV => diverted."""
import json, glob, gzip, math, pandas as pd, numpy as np
AP={'CEC':(41.780,-124.237),'OAK':(37.721,-122.221),'HHR':(33.923,-118.335),'MFR':(42.374,-122.873),'ACV':(40.978,-124.109),'RDD':(40.509,-122.293),'STS':(38.509,-122.813),'SMF':(38.695,-121.591),'SFO':(37.619,-122.375)}
REG={'a38cb0':'N328GT J328','a3a5cf':'N334MS J328','a49680':'N395MS J328','a3e542':'N350ES B350','a4955d':'N395AA B350','a49570':'N395AV B350','a49671':'N395MB B350','ac1b8b':'N88AQ PC12'}
def hav(a,b,c,d):
    p=math.pi/180; return 2*6371*math.asin(math.sqrt(math.sin((c-a)*p/2)**2+math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2))
frames={}
for fn in sorted(glob.glob('data/raw/adsb_traces/day_*/traces/*/trace_full_*.json')):
    hexid=fn.split('trace_full_')[1][:6]; raw=open(fn,'rb').read(); j=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
    t0=j['timestamp']; pts=[(t0+p[0],p[1],p[2],p[3],p[4]) for p in j['trace']]
    df=pd.DataFrame(pts,columns=['t','lat','lon','alt','gs']); frames.setdefault(hexid,[]).append(df)
out=[]
for hexid,fs in frames.items():
    df=pd.concat(fs).drop_duplicates('t').sort_values('t'); df['t']=pd.to_datetime(df.t,unit='s',utc=True).dt.tz_convert('America/Los_Angeles')
    df['alt_n']=pd.to_numeric(df.alt.replace('ground',0),errors='coerce'); df['dCEC']=[hav(a,b,*AP['CEC']) for a,b in zip(df.lat,df.lon)]
    df['gap_after']=df.t.shift(-1)-df.t
    # candidate: last point before a gap >= 15 min while within 200 km of CEC and below 30000 ft
    cand=df[(df.gap_after>=pd.Timedelta(minutes=15))&(df.dCEC<200)]
    for i,r in cand.iterrows():
        nxt=df.loc[i+1] if i+1 in df.index else None
        j=df.index.get_loc(i); nxt=df.iloc[j+1] if j+1<len(df) else None
        if nxt is None: continue
        near=min(AP.items(),key=lambda kv:hav(nxt.lat,nxt.lon,*kv[1]))
        out.append(dict(ac=REG.get(hexid,hexid),last_seen=r.t.strftime('%Y-%m-%d %H:%M'),dCEC_km=round(r.dCEC),alt=r.alt,gap_min=round(r.gap_after.total_seconds()/60),reappear=nxt.t.strftime('%H:%M'),reappear_dCEC=round(nxt.dCEC),reappear_alt=nxt.alt,nearest_ap=f"{near[0]} {hav(nxt.lat,nxt.lon,*near[1]):.0f}km"))
R=pd.DataFrame(out).sort_values(['last_seen','ac']); pd.set_option('display.width',220); print(R.to_string(index=False)); R.to_csv('data/adsb_cec_visits.csv',index=False)
