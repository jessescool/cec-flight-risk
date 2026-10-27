"""Flag day files that look incomplete: a 79-hex day with <30 traces, or any day missing an aircraft that was still
transmitting in the last 3 min of the previous UTC day (readsb carries such traces into the next file)."""
import glob,gzip,json,re,os,pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
AA={'a38cb0','a3a5cf','a49680','a3e542','a4955d','a49570','a49671','ac1b8b'}
def main():
    days=sorted(d.split('day_')[1] for d in glob.glob('data/raw/adsb_traces/day_*'))
    last={}; present={}
    for d in days:
        hs=set()
        for f in glob.glob(f'data/raw/adsb_traces/day_{d}/traces/*/*.json'):
            h=re.search(r'trace_full_(\w+)\.json',f).group(1); hs.add(h); raw=open(f,'rb').read()
            try: j=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
            except Exception: continue
            if j.get('trace'): last[(d,h)]=j['timestamp']+j['trace'][-1][0]
        present[d]=hs
    thin={}
    for d in days:
        n=(pd.Timestamp(d.replace('.','-'))+pd.Timedelta(days=1)).strftime('%Y.%m.%d')
        if n not in present: continue
        wide=len(present[n])>=8   # 79-hex pull vs 8-hex (Sun/Mon, Jul-Oct 2026) pull
        if wide and len(present[n])<30: thin[n]=f'only {len(present[n])} traces'
        end=pd.Timestamp(n.replace('.','-'),tz='UTC').timestamp()
        miss=[h for h in present[d] if last.get((d,h),0)>end-180 and h not in present[n] and (wide or h in AA)]
        if miss: thin[n]=(thin.get(n,'')+' missing boundary-active '+','.join(miss)).strip()
    for d in ['2024.12.31','2025.12.31','2026.05.06']: thin[d]='no archive'
    pd.Series(thin).sort_index().to_csv('data/adsb_thin_days.csv',header=['why'],index_label='day')
    print(len(thin)); print(pd.Series(thin).sort_index().to_string())

if __name__ == '__main__': main()
