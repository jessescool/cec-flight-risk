import requests, pandas as pd, time, sys
base="https://previous-runs-api.open-meteo.com/v1/forecast"
vars_=["temperature_2m","dew_point_2m","relative_humidity_2m","cloud_cover","wind_speed_10m","wind_direction_10m","weather_code","surface_pressure"]
plan={ # model: (start, leads)
 "gfs_seamless":("2024-02-01",range(1,8)),
 "ecmwf_ifs025":("2024-02-01",range(1,8)),
 "gem_seamless":("2024-02-01",range(1,8)),
 "icon_seamless":("2024-02-01",range(1,4)),
 "jma_seamless":("2021-01-01",range(1,8)),
 "ncep_nbm_conus":("2025-01-01",range(1,8)),
 "gfs_hrrr":("2024-02-01",range(1,2)),
}
end=pd.Timestamp("2026-10-18")
for model,(start,leads) in plan.items():
    frames=[]
    for s in pd.date_range(start,end,freq="90D"):
        e=min(s+pd.Timedelta(days=89),end)
        hv=",".join(f"{v}_previous_day{n}" for v in vars_ for n in leads)
        for attempt in range(5):
            r=requests.get(base,params=dict(latitude=41.78,longitude=-124.237,start_date=s.date(),end_date=e.date(),hourly=hv,models=model,timezone="UTC"),timeout=120)
            if r.status_code==200: break
            time.sleep(5*(attempt+1))
        j=r.json()
        if "hourly" not in j: print(model,s.date(),"ERR",j.get("reason")); continue
        f=pd.DataFrame(j["hourly"]); frames.append(f)
        time.sleep(0.5)
    df=pd.concat(frames); df["time"]=pd.to_datetime(df["time"],utc=True)
    df.to_parquet(f"data/prevruns_{model}.parquet")
    nn=df.drop(columns="time").notna().mean().round(2)
    print(model,len(df),"nonnull frac min/max",nn.min(),nn.max(),flush=True)
