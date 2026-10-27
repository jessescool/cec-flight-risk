import requests, json, itertools
base="https://previous-runs-api.open-meteo.com/v1/forecast"
models=["gfs_seamless","ecmwf_ifs025","icon_seamless","best_match","gem_seamless","meteofrance_seamless","ukmo_seamless","ncep_nbm_conus","gfs_hrrr","jma_seamless"]
dates=["2021-06-15","2022-06-15","2023-06-15","2024-02-15","2024-06-15","2025-06-15","2026-06-15"]
vars_=["temperature_2m","dew_point_2m","relative_humidity_2m","cloud_cover","cloud_cover_low","wind_speed_10m","wind_direction_10m","weather_code","surface_pressure","visibility"]
for m in models:
    row=[]
    for d in dates:
        hv=",".join(f"{v}_previous_day{n}" for v in vars_ for n in (1,3,7))
        r=requests.get(base,params=dict(latitude=41.78,longitude=-124.237,start_date=d,end_date=d,hourly=hv,models=m,timezone="America/Los_Angeles"),timeout=60)
        try: h=r.json()["hourly"]
        except Exception as e: row.append(f"{d}:ERR"); continue
        ok=[k.replace("_previous_day","@") for k,v in h.items() if k!="time" and any(x is not None for x in v)]
        # summarize: which vars have data at any lead
        vs=sorted({k.split("@")[0] for k in ok}); leads=sorted({k.split("@")[1] for k in ok})
        row.append(f"{d}: vars={len(vs)} leads={leads} {'cc_low' if 'cloud_cover_low' in vs else ''}{' vis' if 'visibility' in vs else ''}")
    print(m); [print("   ",x) for x in row]
