"""Passenger-arrival risk: P(not in Crescent City by Monday morning) from reconstructed Sunday outcomes.
A landed Sunday evening; B diverted MFR/ACV; C no service (cancelled/no attempt); D failed to reach CEC by Monday morning.
P(D|cat) = P(B|cat)*(1-pB) + P(C|cat)*(1-pC), with pB,pC = P(reach CEC overnight | B or C) given as RANGES (unobserved)."""
import pandas as pd, numpy as np
from scipy.stats import beta
R=pd.read_csv('data/legacy/sunday_outcomes_joined.csv',parse_dates=['sunday'])
R['month']=R.sunday.dt.month; R['season']=np.where(R.month.isin([6,7,8,9]),'Jun-Sep','Oct-May')
det=R[~R.unknown].copy(); det['B']=det.outcome.str.startswith('missed'); det['C']=det.outcome.str.startswith('no_328')
def jeff(k,n): lo,hi=beta.ppf([0.025,0.975],k+0.5,n-k+0.5); return k/n if n else np.nan,lo,hi
print(f"Determinate Sundays: {len(det)} of {len(R)} (undetermined {int(R.unknown.sum())}); landed {int(det.reached_cec_evening.sum())}, diverted {int(det.B.sum())}, no service {int(det.C.sum())}")
print("\nRow = subset; columns = n, P(B) diverted, P(C) no service, P(B or C) with 95% Jeffreys intervals")
rows=[]
def line(name,Z):
    n=len(Z); b=int(Z.B.sum()); c=int(Z.C.sum()); f=b+c
    pb=jeff(b,n); pc=jeff(c,n); pf=jeff(f,n)
    rows.append(dict(subset=name,n=n,diverted=b,no_service=c,p_div=pb[0],p_nosvc=pc[0],p_fail=pf[0],lo=pf[1],hi=pf[2]))
    print(f"  {name:34s} n={n:3d}  B={b} ({pb[0]:.3f})  C={c} ({pc[0]:.3f})  fail={f} {pf[0]:.3f} [{pf[1]:.3f},{pf[2]:.3f}]")
line('all Sundays Mar 2024-Oct 2026',det)
for s in ['Oct-May','Jun-Sep']: line(f'{s}',det[det.season==s])
fc=det.dropna(subset=['p1'])
for c in ['green','yellow','red']: line(f'Friday forecast {c} (2025-26)',fc[fc.cat==c])
for s in ['Oct-May','Jun-Sep']:
    for c in ['green','red']: line(f'  {s} & forecast {c}',fc[(fc.season==s)&(fc.cat==c)])
line('weather at arrival OK',det[det.wx_worst==0]); line('weather marginal',det[det.wx_worst==1]); line('weather below mins',det[det.wx_worst==2])
pd.DataFrame(rows).to_csv('data/passenger_risk_table.csv',index=False)
# P(D) ranges
print("\nP(not in Crescent City by Monday morning) = P(B)(1-pB) + P(C)(1-pC); pB = P(reach CEC overnight | diverted), pC = P(reach | no service)")
for pB,pC,lab in [(0.8,0.7,'optimistic: self-drive from Medford, early cancel notice'),(0.5,0.4,'middle'),(0.2,0.1,'pessimistic: stuck at MFR/OAK')]:
    print(f"  [{lab}] pB={pB} pC={pC}")
    for name,Z in [('all',det),('Oct-May',det[det.season=='Oct-May']),('Jun-Sep',det[det.season=='Jun-Sep']),('fcst green',fc[fc.cat=='green']),('fcst red',fc[fc.cat=='red'])]:
        n=len(Z); b=Z.B.sum(); c=Z.C.sum()
        # posterior mean with Jeffreys-ish prior for each component
        pb=(b+0.25)/(n+0.5); pc=(c+0.25)/(n+0.5); pD=pb*(1-pB)+pc*(1-pC)
        print(f"     {name:11s} P(D)={pD:.3f}   (P(B)={pb:.3f}, P(C)={pc:.3f})")
# annual weekends
print("\nAnnual Oakland weekends vs expected stranded Mondays (52 Sundays; Oct-May 35, Jun-Sep 17). Using category shares from the 2025-26 rolling validation (all days) and empirical fail rates by category.")
share={'Oct-May':{'green':0.75,'yellow':0.11,'red':0.14},'Jun-Sep':{'green':0.25,'yellow':0.21,'red':0.54}}
rate={c:(fc[fc.cat==c].B.sum()+fc[fc.cat==c].C.sum()+0.25)/(len(fc[fc.cat==c])+0.5) for c in ['green','yellow','red']}
print("  empirical P(fail | category), posterior mean:",{k:round(v,3) for k,v in rate.items()})
for pol,cats in [('fly every weekend',['green','yellow','red']),('fly green+yellow',['green','yellow']),('fly green only',['green']),('fly green only, Oct-May; never in summer',None)]:
    wk=0; fails=0
    for s,nwk in [('Oct-May',35),('Jun-Sep',17)]:
        for c in (cats if cats else (['green'] if s=='Oct-May' else [])):
            w=nwk*share[s][c]; wk+=w; fails+=w*rate[c]
    print(f"  {pol:42s} weekends/yr={wk:4.1f}  expected not-landed Sunday evenings/yr={fails:.2f}  (x(1-p) for overnight recovery)")
