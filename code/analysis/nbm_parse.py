import re, glob, pandas as pd, numpy as np
rows=[]
for fn in glob.glob('data/nbm_kcec_nbs_12z*.txt'):
    txt=open(fn).read()
    for blk in txt.split('#### ')[1:]:
        lines=blk.split('\n'); key=lines[0].split()[0]
        hdr=[l for l in lines if l.startswith(' KCEC')]
        if not hdr: continue
        m=re.search(r'(\d+)/(\d+)/(\d{4})\s+(\d{2})00 UTC',hdr[0]); 
        if not m: continue
        init=pd.Timestamp(int(m.group(3)),int(m.group(1)),int(m.group(2)),int(m.group(4)),tz='UTC')
        def row(tag,width=3):
            for l in lines:
                if l.startswith(f' {tag} ') or l.startswith(f' {tag}\t'):
                    body=l[5:]
                    # fixed width 3-char fields
                    vals=[body[i:i+3] for i in range(0,len(body)-len(body)%3 if len(body)%3==0 else len(body),3)]
                    return [int(v) if v.strip() not in ('','-88') else (np.nan if v.strip()=='' else -88) for v in vals]
            return None
        fhr=row('FHR'); 
        if fhr is None: continue
        d={t:row(t) for t in ['CIG','VIS','IFC','IFV','LCB','TMP','DPT','SKY','WSP','WDR']}
        for i,f in enumerate(fhr):
            if f is None or (isinstance(f,float) and np.isnan(f)): continue
            r=dict(init=init,fhr=f,valid=init+pd.Timedelta(hours=f))
            for t,v in d.items():
                r[t]=v[i] if v is not None and i<len(v) else np.nan
            rows.append(r)
df=pd.DataFrame(rows).drop_duplicates(['init','fhr']).sort_values(['init','fhr'])
# units: CIG hundreds of ft (-88 unlimited), VIS tenths of miles? NBM text: VIS in miles*10 (e.g. 100 = 10 mi), IFC/IFV probabilities %
df['cig_ft']=np.where(df.CIG==-88,99999,df.CIG*100); df['vis_mi']=df.VIS/10.0
df.to_parquet('data/nbm_kcec.parquet')
print(len(df), df.init.min(), df.init.max(), df.init.nunique())
print(df[df.fhr.isin([60,63])].describe()[['CIG','VIS','IFC','IFV']].round(1))
