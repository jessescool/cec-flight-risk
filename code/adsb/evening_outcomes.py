"""Classify the evening OAK->CEC arrival for a given local date from adsb.lol traces (uses the local day's UTC file and the
next UTC day's file, merged per aircraft so cross-midnight gaps are seen). Rules (priority order):
 1 landed_ground      : a 328 has 'ground' points at CEC in the window                                   (high)
 2 landed             : CEC_landed_* gap event, inbound in window                                          (high)
 3 missed->X          : CEC_missed/abandoned event in window, then first ground/stop at X                   (medium-high)
 4 landed_overnight   : 328 left OAK northbound in window; next seen next morning climbing out near CEC     (medium-high)
 5 overnight_at_CEC   : 328's first point next morning is climbing out near CEC, arrival unobserved        (medium)
 6 no_328_near_CEC    : no 328 within 150 km of CEC from window start to next-day 13:00                    (medium if a 328 seen elsewhere, low if none)
"""
import pandas as pd, numpy as np, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from adsb_events import load_days, gap_events, classify, REG, J328, AP, hav, TZ
THIN=pd.read_csv('data/adsb_thin_days.csv',dtype=str).set_index('day').why.to_dict() if os.path.exists('data/adsb_thin_days.csv') else {}
def classify_evening(date, w_start_h=16.0):
    d=pd.Timestamp(date); files=[d.strftime('%Y.%m.%d'),(d+pd.Timedelta(days=1)).strftime('%Y.%m.%d')]
    files=[f for f in files if os.path.isdir(f'data/raw/adsb_traces/day_{f}/traces')]
    if len(files)<2: return None   # need the day itself and the next morning; a missing day file would masquerade as 'no service'
    s=d.tz_localize(TZ); w0=s+pd.Timedelta(hours=w_start_h); w1=s+pd.Timedelta(hours=27); m0=s+pd.Timedelta(hours=29); m1=s+pd.Timedelta(hours=37)
    fr={REG.get(h,h):df for h,df in load_days(files).items() if REG.get(h,h) in J328}
    ev=[]
    for ac,df in fr.items():
        for e in gap_events(df,ac):
            lab,conf,note=classify(e); e.update(label=lab,conf=conf,note=note); ev.append(e)
    E=pd.DataFrame(ev) if ev else pd.DataFrame(columns=['ac','t_in','t_out','label','conf','note'])
    W=E[(E.t_in>=w0)&(E.t_in<w1)] if len(E) else E
    ground=[(ac,df[(df.t>=w0)&(df.t<w1)&(df.alt=='ground')&(df.dCEC<5)]) for ac,df in fr.items()]
    ground=[(ac,g) for ac,g in ground if len(g)]
    if ground:
        ac,g=ground[0]; return dict(outcome='landed',conf='high',ac=ac,arrival_est=g.t.min().strftime('%H:%M'),evidence=f'ground-level ADS-B points at CEC from {g.t.min():%H:%M} (receiver coverage that day)')
    landed=W[W.label.str.startswith('CEC_landed')] if len(W) else W; missed=W[W.label.str.contains('missed|abandoned')] if len(W) else W
    if len(landed):
        f=landed.sort_values('t_in').iloc[0]; return dict(outcome='landed'+('_after_missed_approach' if len(missed) and missed.t_in.min()<f.t_in else ''),conf='high',ac=f.ac,arrival_est=(f.t_in+pd.Timedelta(minutes=15)).strftime('%H:%M'),evidence=f"{f.label}: {f.note}")
    if len(missed):
        m=missed.sort_values('t_in').iloc[0]; nxt=E[(E.ac==m.ac)&(E.t_in>m.t_out)].sort_values('t_in'); g=nxt[nxt.label.str.startswith('on_ground')|nxt.label.str.startswith('stop_')]
        dest=g.label.iloc[0].replace('on_ground_','').replace('stop_','') if len(g) else 'unknown'
        return dict(outcome=f'missed->{dest}',conf='medium-high' if dest!='unknown' else 'medium',ac=m.ac,arrival_est='',evidence=f"{m.label} at {m.t_in:%H:%M}; {m.note}; then first ground/stop at {dest}")
    for ac,df in fr.items():
        eve=df[(df.t>=w0)&(df.t<w1)]; mon=df[(df.t>=m0)&(df.t<m1)]
        oak=eve[[hav(a,b,*AP['OAK'])<40 for a,b in zip(eve.lat,eve.lon)]] if len(eve) else eve
        left_oak=bool(len(oak) and (eve[eve.t>oak.t.max()].lat.max()>39.0))
        fm=mon.iloc[0] if len(mon) else None; mon_cec=fm is not None and fm.dCEC<150 and (fm.alt_n or 0)>2000
        if left_oak and mon_cec: return dict(outcome='landed_overnight',conf='medium-high',ac=ac,arrival_est='',evidence=f"left OAK northbound after {oak.t.max():%H:%M}; next seen {fm.t:%a %H:%M} climbing {fm.dCEC:.0f}km from CEC at {fm.alt}ft; nothing at MFR/ACV/OAK between")
    for ac,df in fr.items():
        mon=df[(df.t>=m0)&(df.t<m1)]; fm=mon.iloc[0] if len(mon) else None
        if fm is not None and fm.dCEC<150 and (fm.alt_n or 0)>2000: return dict(outcome='overnight_at_CEC_arrival_unobserved',conf='medium',ac=ac,arrival_est='',evidence=f"first point next morning {fm.t:%H:%M} climbing {fm.dCEC:.0f}km from CEC at {fm.alt}ft")
    near=[ac for ac,df in fr.items() if ((df.t>=w0)&(df.t<m1)&(df.dCEC<150)).any()]
    if near: return dict(outcome='unclear',conf='low',ac=','.join(near),arrival_est='',evidence='328 near CEC but no classifiable event')
    thin=[f for f in files if f in THIN]
    if thin: return dict(outcome='no_328_near_CEC',conf='low',ac=','.join(fr) or 'none',arrival_est='',evidence=f"day file(s) {thin} look incomplete ({'; '.join(THIN[f] for f in thin)}); absence of the 328 is not evidence")
    return dict(outcome='no_328_near_CEC',conf='medium' if fr else 'low',ac=','.join(fr) or 'none',arrival_est='',evidence=f"328s seen in files: {list(fr) or 'none'}; none within 150 km of CEC from window start to next-day 13:00")

def sched_arr_weekday(d):
    """Scheduled OAK->CEC arrival (local decimal hour) for non-Sunday days, from live feed + Wayback captures (Mar 2025, Jun 2025, Oct 2025, Feb 2026, Jun 2026)."""
    dow=d.dayofweek
    if d<pd.Timestamp('2026-06-28'):                      # Oct 2024 - Jun 27 2026 timetable
        if dow in (0,1,2): return 18.1                    # Mon-Wed 17:00 -> 18:06
        if dow==3: return 19.1                            # Thu 18:00 -> 19:06
        if dow==4: return 18.1 if d>=pd.Timestamp('2025-12-05') else 14.1   # Fri 13:00->14:06, 17:00->18:06 from Dec 5 2025
        if dow==5: return 12.1 if d>=pd.Timestamp('2025-10-01') else 14.1   # Sat 13:00->14:06, then 11:00->12:06
    elif d<pd.Timestamp('2026-10-01'):                    # summer 2026
        return {0:18.1,1:18.1,2:18.1,3:19.1,4:17.1,5:16.0}[dow]   # ADS-B shows two Thu arrivals (~18:45, ~22:15) and two Sat (~16:00, ~19:30); window opens 3 h before the first
    else:                                                 # from Oct 1 2026: Mon-Wed/Fri/Sat 17:36 -> 18:42
        return None if dow==3 else 18.7
    return None

if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'sundays'
    rows=[]
    if which=='sundays':
        for sun in pd.date_range('2024-03-17','2026-10-18',freq='W-SUN'):
            r=classify_evening(sun, 12.5 if sun<pd.Timestamp('2024-10-01') else 16.0)
            if r: r['date']=sun.date(); rows.append(r)
        out='data/sunday_outcomes.csv'
    else:   # every non-Sunday day with a scheduled OAK->CEC flight, Oct 2024 onward
        for d in pd.date_range('2024-10-01','2026-10-17'):
            if d.dayofweek==6: continue
            arr=sched_arr_weekday(d)
            if arr is None: continue
            r=classify_evening(d, max(arr-3.0,8.0))
            if r: r['date']=d.date(); r['sched_arr']=f"{int(arr):02d}:{int(round((arr%1)*60)):02d}"; rows.append(r)
        out='data/weekday_outcomes.csv'
    R=pd.DataFrame(rows); R=R[[c for c in ['date','sched_arr','outcome','conf','ac','arrival_est','evidence'] if c in R.columns]]; R.to_csv(out,index=False)
    pd.set_option('display.width',260); pd.set_option('display.max_colwidth',110); print(R.to_string(index=False)); print('\n',R.outcome.value_counts())
