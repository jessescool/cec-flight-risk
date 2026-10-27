"""Can non-weather cancellations (no Dornier flew to CEC) be anticipated from fleet activity visible in ADS-B?
Builds a daily table of the three Dornier 328s (points, first/last seen, airports touched) and tests same-day and prior-week
availability against the reconstructed evening outcomes."""
import glob, gzip, json, os, re, sys, pandas as pd, numpy as np
sys.path.insert(0,'code/adsb'); from adsb_events import REG, J328, nearest, AP, TZ
H328={h for h,r in REG.items() if r in J328}
rows=[]
for d in (sorted(glob.glob('data/raw/adsb_traces/day_*')) if not os.path.exists('data/fleet_daily.csv') else []):
    day=d.split('day_')[1]
    for h in H328:
        fs=glob.glob(f'{d}/traces/*/trace_full_{h}.json')
        if not fs: rows.append(dict(day=day,ac=REG[h],n=0)); continue
        raw=open(fs[0],'rb').read(); j=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
        tr=j['trace']; t=[j['timestamp']+p[0] for p in tr]
        aps=set()
        for p in tr[::20]:
            alt=p[3]; k,dist=nearest(p[1],p[2])
            if dist<25 and (alt=='ground' or (isinstance(alt,(int,float)) and alt<6000)): aps.add(k)
        rows.append(dict(day=day,ac=REG[h],n=len(tr),first=pd.Timestamp(min(t),unit='s',tz='UTC').tz_convert(TZ),last=pd.Timestamp(max(t),unit='s',tz='UTC').tz_convert(TZ),aps=','.join(sorted(aps))))
if rows:
    F=pd.DataFrame(rows); F['date']=pd.to_datetime(F.day.str.replace('.','-')); F.to_csv('data/fleet_daily.csv',index=False)
    print(F.groupby('ac').apply(lambda g: f"active {int((g.n>0).sum())}/{len(g)} days, {g.date.min().date()}..{g.date.max().date()}").to_string())

# ---- join with evening outcomes
F=pd.read_csv('data/fleet_daily.csv',parse_dates=['date']); F['first']=pd.to_datetime(F['first'],utc=True,format='ISO8601'); F['last']=pd.to_datetime(F['last'],utc=True,format='ISO8601')
FO=pd.read_csv('data/flight_outcomes.csv',parse_dates=['date']); S=FO[FO.date.dt.dayofweek==6]; W=FO[FO.date.dt.dayofweek!=6]
c=['date','sched_arr','outcome','conf','fail','unknown','wx_worst']
FL=pd.concat([S[c],W[c]]).sort_values('date'); FL=FL[(FL.sched_arr>='17:00')&~FL.unknown].drop_duplicates('date').set_index('date')
FL['nonwx_fail']=(FL.fail&(FL.wx_worst==0)).astype(int); FL['fail']=FL.fail.astype(int)
act=F[F.n>0]
# trace files are UTC days: file day d holds local d-1 17:00 .. d 17:00. "known by 15:00 local on d" = points in file d with local time on d before 15:00
def known_today(d):
    g=act[(act.date==d)]; k=0
    for _,r in g.iterrows():
        if r['first'].tz_convert(TZ).date()==d.date() and r['first'].tz_convert(TZ).hour<15: k+=1
        elif r['last'].tz_convert(TZ).date()==d.date(): k+=1   # still airborne into the local day
    return k
feat=[]
for d in FL.index:
    prev=act[(act.date<d)&(act.date>=d-pd.Timedelta(days=7))]
    prev1=act[(act.date<d)&(act.date>=d-pd.Timedelta(days=1))]
    feat.append(dict(date=d,n_prev7=prev.ac.nunique(),n_prev1=prev1.ac.nunique(),n_today15=known_today(d),
                     cec_prev7=int(prev.aps.fillna('').str.contains('CEC').sum()),
                     days_cov=F[(F.date<d)&(F.date>=d-pd.Timedelta(days=7))].date.nunique()))
X=FL.join(pd.DataFrame(feat).set_index('date'))
X=X[X.days_cov>=5]   # need archive coverage for the look-back
from scipy.stats import beta, fisher_exact
def ci(k,n): lo,hi=beta.ppf([0.025,0.975],k+0.5,n-k+0.5); return f'{k}/{n} = {k/n:.3f} [{lo:.3f},{hi:.3f}]'
print(f"\nflights with 7-day fleet look-back: {len(X)}, non-weather failures {X.nonwx_fail.sum()}, all failures {X.fail.sum()}")
for col in ['n_prev7','n_prev1','n_today15']:
    print(f"\n-- by {col}:")
    for v,g in X.groupby(col): print(f"   {col}={v}: non-weather fail {ci(int(g.nonwx_fail.sum()),len(g))};  any fail {ci(int(g.fail.sum()),len(g))}")
a=X[X.n_prev7<=1]; b=X[X.n_prev7>=2]
print(f"\nFisher single-aircraft week vs not, non-weather fail: p={fisher_exact([[a.nonwx_fail.sum(),len(a)-a.nonwx_fail.sum()],[b.nonwx_fail.sum(),len(b)-b.nonwx_fail.sum()]])[1]:.4f}")
a=X[X.n_today15==0]; b=X[X.n_today15>=1]
print(f"Fisher no 328 seen by 15:00 vs seen, non-weather fail: p={fisher_exact([[a.nonwx_fail.sum(),len(a)-a.nonwx_fail.sum()],[b.nonwx_fail.sum(),len(b)-b.nonwx_fail.sum()]])[1]:.4f}")
pd.set_option('display.width',200)
print("\nnon-weather failures with fleet context:"); print(X[X.nonwx_fail==1][['sched_arr','outcome','n_prev7','n_prev1','n_today15','cec_prev7']].to_string())
# single-aircraft periods
A=act.groupby('date').ac.nunique().reindex(pd.date_range(act.date.min(),act.date.max())).fillna(0)
r=A.rolling(7).max(); single=(r<=1)
print("\nweeks with at most one Dornier active (7-day window):"); 
blk=(single!=single.shift()).cumsum(); 
for _,g in single[single].groupby(blk[single]): print(f"   {g.index.min().date()} .. {g.index.max().date()} ({len(g)} days)")
X.to_csv('data/fleet_outcomes_joined.csv')
