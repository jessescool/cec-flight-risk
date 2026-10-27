# cec-flight-risk

Probability that Advanced Air's evening OAK → CEC (Crescent City) flight does not land at Crescent City, from public
forecasts 1–7 days ahead. Built for one passenger's go/no-go decision. Everything runs on demand; nothing is scheduled.

## The number

```
P(no landing) = Σ_t  P(weather tier t at arrival | forecast) · P(flight fails | tier t)
```

| Stage | Source | Out-of-sample skill |
|---|---|---|
| Weather tier at arrival | Logistic regression on ensemble features (ECMWF, GFS, GEM, ICON via Open-Meteo; ICON only at leads 1–3), 979 days since Feb 2024 | Rolling CV on 650 days: AUC 0.76–0.89 for below minimums across leads 7–1 (49 events), 0.85–0.89 at leads 1–3 |
| P(fail \| tier) | 541 evening flights Oct 2024 – Oct 2026 with outcomes reconstructed from ADS-B (536 have forecast features) | fog 9/40, low cloud 1/57 (floored at the clear rate), clear 15/439 |
| Whole model | Rolling quarterly CV 2025Q1–2026Q4 of the shipped configuration, 477 flights, 24 failures | AUC 0.62–0.74 across leads 7–1, Brier skill +0.03 to +0.05 vs the training-period base rate |

Tiers follow the KCEC ILS/LPV 12 minimums: below minimums = ceiling < 300 ft or visibility < 3/4 mi; low cloud = ceiling ≤ 500 ft or visibility ≤ 1.5 mi.

The clear-weather term is a floor no forecast can see (the aircraft never came). It is the trailing-12-month rate; at lead 1 only, it is split on whether the previous evening's flight landed (8/161 vs 3/16). The mixture is forced monotone in weather. Within 1.5 h of arrival a persistence nowcast from the latest METAR is shown next to the forecast figure; from 2 h out the forecast is the better predictor (`data/nowcast_vs_ensemble.txt`).

## Run

```
pip install -r requirements.txt
python3 -m pytest tests                       # unit tests for the shared logic
python3 code/weekly_check.py                  # next Sunday
python3 code/weekly_check.py --date 2026-11-01 [--yesterday-cancelled]   # the flag matters at lead 1 only
python3 code/track_outcomes.py                # log the airline's live status feed into data/flight_log.csv
```

Rebuild from data:
```
python3 code/adsb/join_weather.py             # data/flight_outcomes.csv from the reconstructed outcome tables + METARs
python3 code/train_outcome.py                 # models/outcome_lead{1..7}.joblib, data/outcome_model_results.txt
python3 code/nowcast.py                       # data/nowcast_table.csv
```
Rebuild the outcome tables from the ADS-B archive (2.1 GB in `data/raw/adsb_traces/`, not in git):
```
python3 code/adsb/adsb_events.py              # data/adsb_events.csv
python3 code/adsb/thin_days.py                # data/adsb_thin_days.csv (incomplete archive days)
python3 code/adsb/evening_outcomes.py sundays
python3 code/adsb/evening_outcomes.py weekdays
```

## Layout

```
code/common.py            tiers, METAR parsing, schedule by era (hard-coded dates; update on a timetable change), Jeffreys intervals, forecast features
code/outcome.py           crew stage: P(fail | tier), floor variants, monotone mixture
code/weekly_check.py      the live check
code/train_outcome.py     train + rolling-origin validation of the outcome model
code/nowcast.py           persistence table from the METAR archive
code/track_outcomes.py    poll the airline status feed
code/adsb/                ADS-B archive pull, gap-event reconstruction, outcome tables, weather join
code/analysis/            one-off studies (HRRR, NBM, TAF, fleet, nowcast-vs-ensemble); legacy/ holds the Oct-2026 weather-proxy generation
tests/                    pytest
data/                     inputs and model outputs; data/raw/ holds downloads (METAR archive, TAFs, charts, airport packets)
models/                   outcome_lead{N}.joblib (shipped); lead{N}.joblib (legacy weather-proxy models)
reports/                  audits, inquiries, write-ups
```

## What is and isn't supported by the data

Supported (`reports/stat_audit_2026-10-26.md`, `reports/audit_2026-10-26.md`):
- forecast features predict the airport tier at arrival (650 out-of-sample days, AUC 0.85–0.89 for below minimums at leads 1–3, 0.76 at lead 7);
- fog at arrival raises the failure rate about 6x over clear weather (9/40 vs 15/439, p < 0.0001);
- out of sample Jan 2025 – Oct 2026, a red forecast (P(low cloud) ≥ 0.30) at lead 3 preceded all 10 weather-coincident failures (9 of 10 at lead 1); a green one (< 0.15) preceded none at lead 3 and one at lead 1 (`data/outcome_model_results.txt`, ledger lines).

Not separable in this sample, reported as sensitivities only:
- Sunday vs arrival-time effect on the fog failure rate (5/9 vs 4/31 and 6/14 vs 3/26; nominal p = 0.016 and 0.044, 11 tests run). Pre-registered rule in `reports/audit_2026-10-26.md`.

Shipped on borderline evidence, flagged as such:
- the previous-evening split of the floor (3/16 vs 8/161, p ≈ 0.007, borderline after correction), used at lead 1 only.

Known limits:
- outcomes are inferred from ADS-B coverage gaps; 8 low-confidence absences and 33 undetermined evenings are excluded (they do not cluster on fog nights, Fisher p = 0.73);
- the airport's own monthly reports match our counts for the two months both exist (Nov 2024, Dec 2025) but show daytime diversions we do not score;
- the clear-weather floor rose from 3/203 (1.5%) in 2025 to 11/184 (6.0%) in 2026 and the carrier is under performance review by the airport board; expect it to move;
- the model's top bin (predicted 0.10–0.20) ran about one in three observed out of sample at leads 1–3 (n = 10–15 per lead; sparser beyond), so a printed 0.12 reads as roughly one in three; the check prints the matching bin;
- NBM ceiling/visibility guidance reaches 72 h only; HRRR's own ceiling/visibility fields scored worse than the ensemble (AUC 0.77–0.80 vs 0.90) and are not used;
- a failed forecast fetch is imputed with training medians and flagged with a WARNING line; a failed schedule fetch falls back to the era default arrival time, also flagged.
