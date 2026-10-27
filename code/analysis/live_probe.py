import requests, time, json, datetime as dt, sys
HEX={'a38cb0','a3a5cf','a49680','a3e542','a4955d','a49570','a49671','ac1b8b'}
LAT,LON=41.78,-124.237
NETS={
 'opensky': lambda: requests.get('https://opensky-network.org/api/states/all',params=dict(lamin=41.4,lamax=42.2,lomin=-124.7,lomax=-123.7),timeout=20).json().get('states') or [],
 'adsblol': lambda: requests.get(f'https://api.adsb.lol/v2/point/{LAT}/{LON}/40',timeout=20).json().get('ac',[]),
 'adsbfi': lambda: requests.get(f'https://opendata.adsb.fi/api/v2/lat/{LAT}/lon/{LON}/dist/40',timeout=20).json().get('aircraft',[]),
}
end=dt.datetime.fromisoformat(sys.argv[1]); out=open('downloads/live_probe.jsonl','a')
while dt.datetime.now()<end:
    t=dt.datetime.now().isoformat(timespec='seconds')
    for n,f in NETS.items():
        try:
            acs=f()
            for a in acs:
                if n=='opensky':
                    rec=dict(hex=a[0].strip().lower(),lat=a[6],lon=a[5],alt_ft=None if a[7] is None else a[7]*3.281,gs=a[9],gnd=a[8])
                else:
                    rec=dict(hex=a.get('hex','').lower(),lat=a.get('lat'),lon=a.get('lon'),alt_ft=a.get('alt_baro'),gs=a.get('gs'),gnd=a.get('alt_baro')=='ground')
                rec.update(t=t,net=n,n_all=len(acs),ours=rec['hex'] in HEX)
                if rec['ours'] or (rec['alt_ft'] not in (None,'ground') and isinstance(rec['alt_ft'],(int,float)) and rec['alt_ft']<6000):
                    out.write(json.dumps(rec)+'\n')
            out.write(json.dumps(dict(t=t,net=n,n_all=len(acs),hb=1))+'\n')
        except Exception as e:
            out.write(json.dumps(dict(t=t,net=n,err=str(e)[:120]))+'\n')
    out.flush(); time.sleep(20)
