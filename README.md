# cec-flight-risk

Probability that Advanced Air's evening OAK → CEC (Crescent City) flight does not land at Crescent City, from public
forecasts 1–7 days ahead. Built for one passenger's go/no-go decision. Everything runs on demand; nothing is scheduled.

## The number

```
P(no landing) = Σ_t  P(weather tier t at arrival | forecast) · P(flight fails | tier t)
```

| Stage | Source | Out-of-sample skill |
|---|---|---|
| Weather tier at arrival | Logistic regression on ensemble features (ECMWF, GFS, GEM, ICON via Open-Meteo; ICON only at leads 1–3), 979 days since Feb 2024 | Rolling CV 2025Q4–2026Q4 on 380 days: AUC for below minimums in `data/outcome_model_results.txt` per lead (0.89 at lead 1) |
| P(fail \| tier) | 541 evening flights Oct 2024 – Oct 2026 with outcomes reconstructed from ADS-B (536 have forecast features) | clear 15/439; low cloud 1/57 (floored at the clear rate); fog 9/40 pooled, split by whether the forecast saw it coming (per lead, in the results file) |
| Whole model | Rolling quarterly CV 2025Q4–2026Q4 of the shipped configuration, 294 flights, 21 failures (earlier quarters have too few fog flights with outcomes to score the crew stage) | AUC 0.60–0.67, Brier skill +0.03 to +0.07 vs the training-period base rate |

Tiers follow the KCEC ILS/LPV 12 minimums: below minimums = ceiling < 300 ft or visibility < 3/4 mi; low cloud = ceiling ≤ 500 ft or visibility ≤ 1.5 mi.

The clear-weather term is a floor no forecast can see (the aircraft never came). It is the trailing-12-month rate; at lead 1 only, it is split on whether the previous evening's flight landed (8/161 vs 3/14). The fog term is conditioned on the forecast's own out-of-sample P(below minimums): fog the forecast saw coming (P ≥ 0.3) is the persistent kind and crews fail about half the time; surprise fog is brief and crews usually land. The split was identified on the 2025–2026 out-of-sample reliability table, where the 0.10–0.20 bin ran about 2.3x its predicted rate, so that table cannot confirm it; a held-out check (rates from before July 2026, scored on 87 flights after) favours the split at leads 1, 2, 3, 5 and 6, is neutral at 7 and slightly against at 4, on 1–3 confident-fog flights per cell (`data/fog_split_holdout.txt`). At lead 7 the data point the other way (0/1 vs 7/22) and the code keeps the pooled rate there. The mixture is forced monotone in weather. Within 1.5 h of arrival a persistence nowcast from the latest METAR is shown next to the forecast figure; from 2 h out the forecast is the better predictor (`data/nowcast_vs_ensemble.txt`).

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
code/outcome.py           crew stage: P(fail | tier), floor variants, confidence-conditioned fog rate, monotone mixture
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

Supported (`reports/audit_2026-10-26.md`; the earlier `reports/stat_audit_2026-10-26.md` carries pre-rebuild counts and is kept for its test ledger):
- forecast features predict the airport tier at arrival (380 out-of-sample days 2025Q4–2026Q4; AUC per lead in the results file);
- fog at arrival raises the failure rate about 6x over clear weather (9/40 vs 15/439, p < 0.0001);
- out of sample Oct 2025 – Oct 2026, a red forecast (P(low cloud) ≥ 0.30) at lead 3 preceded all 8 weather-coincident failures (7 of 8 at lead 1); a green one (< 0.15) preceded none at lead 3 and one at lead 1 (`data/outcome_model_results.txt`, ledger lines).

Not separable in this sample, reported as sensitivities only:
- Sunday vs arrival-time effect on the fog failure rate. Within forecastable fog in the scored period (lead 3): Sunday 3/3 vs weekday 3/8 (Fisher p = 0.18); arrival ≥ 19:00 4/5 vs 2/6 (p = 0.24). Sunday is the leading residual hypothesis (an earlier, longer out-of-sample run gave 4/4 vs 4/14, p = 0.02) but on three events it is not shipped; the pre-registered rule in `reports/audit_2026-10-26.md` decides when it would be.

Shipped on weak evidence, flagged as such:
- the previous-evening split of the floor (trailing 12 months: 3/14 after a non-landing vs 8/161 after a landing, Fisher p = 0.046, nominal only after 11 tests), used at lead 1 only because the mechanism (multi-day disruption) is plausible and the cost of being wrong is one evening's number.

Known limits:
- outcomes are inferred from ADS-B coverage gaps; 33 evenings are excluded as undetermined, 8 of them low-confidence absences (they do not cluster on fog nights, Fisher p = 0.73);
- the airport's own monthly reports (`data/airport_reports_vs_adsb.csv`) overlap our reconstruction in five months, counting segments where we count evenings: Nov 2024, Mar 2025 and Dec 2025 are consistent, Jun 2025 disagrees (airport 86/86 completed, our reconstruction one scored failure), and Dec 2024 shows two daytime diversions we do not score;
- the clear-weather floor rose from 3/203 (1.5%) in 2025 to 11/184 (6.0%) in 2026 and the airport board's 2026 packets carry a standing item on the carrier's on-time performance plus direction to a consultant on alternatives; expect it to move;
- calibration: reliability bins per lead are in the results file; the check prints the bin its number falls in, with the count, so the reader sees how thin it is;
- NBM ceiling/visibility guidance reaches 72 h only; HRRR's own ceiling/visibility fields scored worse than the ensemble (AUC 0.77–0.80 vs 0.90) and are not used;
- a failed forecast fetch is imputed with training medians and flagged with a WARNING line; a failed schedule fetch falls back to the era default arrival time, also flagged; a dead status feed is flagged because a same-day cancellation would then go unseen. Upstream delays are listed, not scored: there is no history to validate a rule.
