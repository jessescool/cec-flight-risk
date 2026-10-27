"""Gap-based reconstruction of Advanced Air movements at CEC from adsb.lol traces.
Coverage: none below ~8-20k ft near CEC, so a landing is a GAP. We classify each coverage gap of a fleet aircraft by
what the trace looked like going in and coming out. Each event carries evidence + confidence."""
import json, glob, gzip, math, os, sys, pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
AP={'CEC':(41.780,-124.237),'OAK':(37.721,-122.221),'HHR':(33.923,-118.335),'MFR':(42.374,-122.873),'ACV':(40.978,-124.109),'RDD':(40.509,-122.293),'STS':(38.509,-122.813),'SMF':(38.695,-121.591),'SFO':(37.619,-122.375),'OTH':(43.417,-124.246),'EUG':(44.125,-123.212),'SJC':(37.363,-121.929),'SCK':(37.894,-121.238)}
REG={'a38cb0':'N328GT','a3a5cf':'N334MS','a49680':'N395MS','a3e542':'N350ES','a4955d':'N395AA','a49570':'N395AV','a49671':'N395MB','ac1b8b':'N88AQ'}
J328={'N328GT','N334MS','N395MS'}
TZ='America/Los_Angeles'
def hav(a,b,c,d):
    p=math.pi/180; return 2*6371*math.asin(math.sqrt(math.sin((c-a)*p/2)**2+math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2))
def nearest(lat,lon):
    k,v=min(AP.items(),key=lambda kv:hav(lat,lon,*kv[1])); return k,hav(lat,lon,*v)
def load_days(days):
    fr={}
    for day in days:
        for fn in glob.glob(f'data/raw/adsb_traces/day_{day}/traces/*/trace_full_*.json'):
            h=fn.split('trace_full_')[1][:6]; raw=open(fn,'rb').read(); j=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
            t0=j['timestamp']; df=pd.DataFrame([(t0+p[0],p[1],p[2],p[3],p[4]) for p in j['trace']],columns=['t','lat','lon','alt','gs'])
            fr.setdefault(h,[]).append(df)
    out={}
    for h,fs in fr.items():
        df=pd.concat(fs).drop_duplicates('t').sort_values('t').reset_index(drop=True)
        df['t']=pd.to_datetime(df.t,unit='s',utc=True).dt.tz_convert(TZ); df['alt_n']=pd.to_numeric(df.alt.replace('ground',0),errors='coerce')
        df['dCEC']=[hav(a,b,*AP['CEC']) for a,b in zip(df.lat,df.lon)]
        out[h]=df
    return out
def gap_events(df,ac):
    """Return list of gap events (>=12 min) with context."""
    ev=[]
    t=df.t.values; n=len(df)
    for i in range(n-1):
        gap=(df.t.iloc[i+1]-df.t.iloc[i]).total_seconds()/60
        if gap<12: continue
        a=df.iloc[i]; b=df.iloc[i+1]
        # trend before: altitude change over previous 3 min
        prev=df[(df.t<a.t)&(df.t>=a.t-pd.Timedelta(minutes=4))]; post=df[(df.t>b.t)&(df.t<=b.t+pd.Timedelta(minutes=4))]
        desc=(a.alt_n-prev.alt_n.iloc[0]) if len(prev) and pd.notna(a.alt_n) and pd.notna(prev.alt_n.iloc[0]) else np.nan
        climb=(post.alt_n.iloc[-1]-b.alt_n) if len(post) and pd.notna(b.alt_n) and pd.notna(post.alt_n.iloc[-1]) else np.nan
        apA,dA=nearest(a.lat,a.lon); apB,dB=nearest(b.lat,b.lon)
        ev.append(dict(ac=ac,t_in=a.t,t_out=b.t,gap_min=round(gap),in_dCEC=round(a.dCEC),in_alt=a.alt_n,in_trend=desc,in_ap=apA,in_apd=round(dA),out_dCEC=round(b.dCEC),out_alt=b.alt_n,out_trend=climb,out_ap=apB,out_apd=round(dB),in_ground=(a.alt=='ground'),out_ground=(b.alt=='ground')))
    return ev
def classify(e):
    """Classify a gap event. Returns (label, confidence, note)."""
    inb = e['in_dCEC']<160 and (pd.isna(e['in_alt']) or e['in_alt']<30000)
    outb= e['out_dCEC']<160
    if e['in_ground'] or e['out_ground']:
        ap=e['in_ap'] if e['in_ground'] else e['out_ap']; return (f'on_ground_{ap}','high','ground point visible')
    if inb and outb:
        if e['gap_min']>=35 and (pd.isna(e['out_trend']) or e['out_trend']>-200) and e['out_alt']>=3000:
            if e['gap_min']>=240: return ('CEC_landed_overnight_or_long_stop','high',f"inbound {e['in_dCEC']}km/{e['in_alt']:.0f}ft, out {e['gap_min']}min later climbing {e['out_dCEC']}km/{e['out_alt']:.0f}ft")
            return ('CEC_landed_then_departed','high' if e['gap_min']>=45 else 'medium',f"inbound {e['in_dCEC']}km/{e['in_alt']:.0f}ft, reappeared {e['gap_min']}min later at {e['out_dCEC']}km/{e['out_alt']:.0f}ft climbing")
        if e['gap_min']<35: return ('CEC_missed_approach_or_hold','medium',f"short gap {e['gap_min']}min; in {e['in_dCEC']}km/{e['in_alt']:.0f}ft, out {e['out_dCEC']}km/{e['out_alt']:.0f}ft")
        return ('CEC_unclear','low','')
    if inb and not outb:
        if e['gap_min']<90: return (f"CEC_abandoned_to_{e['out_ap']}",'medium',f"in {e['in_dCEC']}km/{e['in_alt']:.0f}ft; {e['gap_min']}min later near {e['out_ap']} ({e['out_apd']}km, {e['out_alt']:.0f}ft)")
        return (f"CEC_stop_then_to_{e['out_ap']}",'low',f"in {e['in_dCEC']}km; out {e['gap_min']}min later near {e['out_ap']} {e['out_apd']}km")
    if e['in_apd']<40 and e['in_alt']<8000: return (f"stop_{e['in_ap']}",'high' if e['gap_min']>=25 else 'medium',f"descending near {e['in_ap']}, gap {e['gap_min']}min")
    return ('other_gap','low','')
if __name__=='__main__':
    days=sorted(d.split('day_')[1] for d in glob.glob('data/raw/adsb_traces/day_*'))
    rows=[]
    for day in days:
        fr=load_days([day])
        for h,df in fr.items():
            for e in gap_events(df,REG.get(h,h)):
                lab,conf,note=classify(e); e.update(file_day=day,label=lab,conf=conf,note=note); rows.append(e)
    E=pd.DataFrame(rows); E.to_csv('data/adsb_events.csv',index=False)
    pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
    c=E[E.label.str.startswith('CEC')]
    print(c[['file_day','ac','t_in','gap_min','label','conf','note']].to_string(index=False))
    print('\n',E.label.value_counts())
