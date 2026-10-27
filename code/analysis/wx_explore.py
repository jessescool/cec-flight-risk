import pandas as pd, numpy as np
df = pd.read_csv('downloads/cec_asos_2019_2026.csv', na_values='M', low_memory=False)
df['valid']=pd.to_datetime(df['valid'],utc=True)
df['local']=df['valid'].dt.tz_convert('America/Los_Angeles')
# ceiling = lowest BKN/OVC/VV layer
def ceiling(r):
    for c,l in [('skyc1','skyl1'),('skyc2','skyl2'),('skyc3','skyl3')]:
        if r[c] in ('BKN','OVC','VV') and pd.notna(r[l]): return r[l]
    return np.nan
df['ceil']=df.apply(ceiling,axis=1)
df['vsby']=pd.to_numeric(df['vsby'],errors='coerce')
df.to_parquet('data/cec_metar.parquet')
print(df['local'].min(), df['local'].max(), len(df))
print("missing ceil layer info rows:", df[['skyc1']].isna().sum().item())
# yearly coverage
print(df.groupby(df['local'].dt.year).size())
# evening window Sunday 18:00-21:00 local
ev = df[(df.local.dt.hour>=18)&(df.local.dt.hour<=21)]
ev['below_ils'] = (ev.ceil<600)|(ev.vsby<2)   # placeholder thresholds; refine after approach research
ev['below_200']=(ev.ceil<200)|(ev.vsby<0.5)
g=ev.groupby(ev.local.dt.month).agg(n=('ceil','size'),ceil_lt600=('below_ils','mean'),ceil_lt200=('below_200','mean'))
print(g.round(3))
sun=ev[ev.local.dt.dayofweek==6]
print("Sunday evenings by month"); print(sun.groupby(sun.local.dt.month).agg(n=('ceil','size'),below=('below_ils','mean')).round(3))
