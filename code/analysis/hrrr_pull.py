import requests, eccodes, sys, csv, os, datetime as dt
from concurrent.futures import ProcessPoolExecutor
LAT,LON=41.780,-124.237
B='https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
FIELDS={'VIS:surface':'vis','HGT:cloud ceiling':'ceil','LCDC:low cloud layer':'lcc'}
RUNS={0:[47,48],6:[44,45]}
S=requests
IDX=None
def point(msg):
    global IDX
    gid=eccodes.codes_new_from_message(msg)
    try:
        if IDX is None or eccodes.codes_get(gid,'numberOfPoints')!=IDX[1]:
            n=eccodes.codes_grib_find_nearest(gid,LAT,LON)[0]; IDX=(n.index,eccodes.codes_get(gid,'numberOfPoints')); return n.value
        return eccodes.codes_get_double_element(gid,'values',IDX[0])
    finally: eccodes.codes_release(gid)
def one(day):
    rows=[]
    init_day=(day-dt.timedelta(days=1)).strftime('%Y%m%d')
    for hh,fhs in RUNS.items():
        for fh in fhs:
            u=f'{B}/hrrr.{init_day}/conus/hrrr.t{hh:02d}z.wrfsfcf{fh:02d}.grib2'
            try:
                idx=S.get(u+'.idx',timeout=30)
                if idx.status_code!=200: rows.append((day,hh,fh,'missing',None)); continue
                L=[l.split(':') for l in idx.text.strip().split('\n')]
                for i,l in enumerate(L):
                    key=f'{l[3]}:{l[4]}'
                    if key in FIELDS:
                        a=int(l[1]); b=int(L[i+1][1])-1 if i+1<len(L) else ''
                        r=S.get(u,headers={'Range':f'bytes={a}-{b}'},timeout=60)
                        rows.append((day,hh,fh,FIELDS[key],point(r.content)))
            except Exception as e:
                rows.append((day,hh,fh,'err',str(e)[:80]))
    return rows
if __name__=='__main__':
    d0,d1=dt.date.fromisoformat(sys.argv[1]),dt.date.fromisoformat(sys.argv[2])
    out=sys.argv[3]; done=set()
    if os.path.exists(out):
        done={r[0] for r in csv.reader(open(out))}
    days=[d0+dt.timedelta(i) for i in range((d1-d0).days+1) if str(d0+dt.timedelta(i)) not in done][::-1]
    f=open(out,'a'); w=csv.writer(f)
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv)>4 else 6) as ex:
        for rows in ex.map(one,days):
            for r in rows: w.writerow(r)
            f.flush()
