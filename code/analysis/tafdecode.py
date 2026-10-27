import re, pandas as pd, numpy as np
from datetime import timedelta
VIS_RE=re.compile(r'(?<!\S)(P6SM|(\d+)SM|(\d)/(\d)SM|(\d) (\d)/(\d)SM|M1/4SM)')
def parse_vis(s):
    m=VIS_RE.search(s)
    if not m: return np.nan
    t=m.group(0)
    if t.startswith('P6'): return 7.0
    if t.startswith('M1/4'): return 0.1
    t=t[:-2]
    if ' ' in t: a,b=t.split(); n,d=b.split('/'); return int(a)+int(n)/int(d)
    if '/' in t: n,d=t.split('/'); return int(n)/int(d)
    return float(t)
def parse_ceil(s):
    best=np.nan
    for cov,h in re.findall(r'(BKN|OVC|VV)(\d{3})',s):
        v=int(h)*100
        best=v if np.isnan(best) else min(best,v)
    return best
def decode(text, vfrom):
    """Return list of periods: (start, end, kind, vis, ceil). vfrom: pandas Timestamp UTC of validity start."""
    toks=text.split()
    # split into groups at FM/TEMPO/BECMG/PROB
    groups=[]; cur=[]
    for t in toks:
        if re.match(r'^(FM\d{6}|TEMPO|BECMG|PROB\d0)$',t):
            if cur: groups.append(cur)
            cur=[t]
        else: cur.append(t)
    if cur: groups.append(cur)
    def ts(dd,hh):
        d=vfrom.replace(hour=0,minute=0)
        # day-of-month relative to vfrom's month; handle month wrap
        base=vfrom.replace(day=1,hour=0,minute=0)
        try: t=base.replace(day=int(dd))+timedelta(hours=int(hh))
        except ValueError: t=(base+pd.offsets.MonthBegin(1)).replace(day=int(dd))+timedelta(hours=int(hh))
        if t<vfrom-timedelta(days=15): t=t+pd.offsets.MonthBegin(1)
        return t
    periods=[]; base_start=vfrom; prev_base=None
    for gi,g in enumerate(groups):
        s=' '.join(g); head=g[0]
        if head.startswith('FM'):
            start=ts(head[2:4],head[4:6])
            if prev_base is not None: prev_base['end']=start
            prev_base=dict(start=start,end=None,kind='BASE',vis=parse_vis(s),ceil=parse_ceil(s),text=s); periods.append(prev_base)
        elif head in('TEMPO','BECMG') or head.startswith('PROB'):
            m=re.search(r'(\d{2})(\d{2})/(\d{2})(\d{2})',s)
            if not m: continue
            kind=head if not head.startswith('PROB') else 'PROB'
            if len(g)>1 and g[1]=='TEMPO': kind='PROBTEMPO'
            periods.append(dict(start=ts(m.group(1),m.group(2)),end=ts(m.group(3),m.group(4)),kind=kind,vis=parse_vis(s),ceil=parse_ceil(s),text=s))
        else:
            prev_base=dict(start=vfrom,end=None,kind='BASE',vis=parse_vis(s),ceil=parse_ceil(s),text=s); periods.append(prev_base)
    return periods
def forecast_at(periods, t, valid_end):
    """Worst-case (conservative) and base forecast at time t."""
    base=[p for p in periods if p['kind']=='BASE' and p['start']<=t and (p['end'] is None or t<p['end'])]
    if not base: return None
    b=base[-1]; vis=b['vis']; ceil=b['ceil']
    bec=[p for p in periods if p['kind']=='BECMG' and p['start']<=t]
    for p in bec:   # BECMG: assume condition reached by end; treat as in effect after start (conservative)
        if not np.isnan(p['vis']): vis=p['vis']
        if not np.isnan(p['ceil']): ceil=p['ceil']
    out=dict(vis=vis,ceil=ceil,tempo_vis=vis,tempo_ceil=ceil,prob=False,tempo=False)
    for p in periods:
        if p['kind'] in('TEMPO','PROB','PROBTEMPO') and p['start']<=t<p['end']:
            out['tempo']=True; out['prob']=out['prob'] or p['kind']!='TEMPO'
            if not np.isnan(p['vis']): out['tempo_vis']=min(out['tempo_vis'] if not np.isnan(out['tempo_vis']) else 99,p['vis'])
            if not np.isnan(p['ceil']): out['tempo_ceil']=min(out['tempo_ceil'] if not np.isnan(out['tempo_ceil']) else 99999,p['ceil'])
    return out
if __name__=='__main__':
    df=pd.read_parquet('data/cec_taf_raw.parquet')
    r=df[df.issued=='2024-08-11T23:20:00Z'].iloc[0]
    vfrom=r.issued.replace(day=r.vfrom_d,hour=r.vfrom_h,minute=0) if r.vfrom_d>=r.issued.day-1 else None
    print(r.text); ps=decode(r.text,vfrom); [print(p) for p in ps]
    print(forecast_at(ps,pd.Timestamp('2024-08-12T02:00Z'),None))
