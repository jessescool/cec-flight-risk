# Statistical validity audit, outcome model (2026-10-26)

Scope: every quantitative claim made while building `code/train_outcome.py`, `code/analysis/fleet_floor.py` and the weekly_check output. Method: Fisher exact tests on each 2x2 the session relied on, Jeffreys 95% intervals, test-count reconciliation, independence and measurement checks.

## Test ledger (11 explicit comparisons; Bonferroni alpha for 11 = 0.0045)

| Comparison | Counts | Fisher p | Survives correction |
|---|---|---|---|
| fail given below-mins vs given clear | 9/40 vs 15/438 | <0.0001 | yes |
| red forecast catches all fog failures | 10/10 with red on 25% of flights | ~1e-6 | yes |
| green forecast, fog failures | 0/333, upper 95% 0.008 | n/a | yes |
| prev-day cancelled -> next-day cancel | 3/15 vs 12/515 | 0.007 | no (borderline) |
| Sunday vs weekday fail given below-mins | 5/9 vs 4/31 | 0.016 | no |
| floor 2026 vs 2025 | 11/232 vs 3/249 (pre-rebuild tables; the canonical evening-only table gives 11/184 vs 3/203) | 0.028 | no |
| arrival >=19:00 vs <19:00 fail given below-mins | 6/14 vs 3/26 | 0.044 | no |
| red Sunday vs red weekday any-fail | 5/27 vs 6/106 | 0.046 | no |
| 3 aircraft active vs fewer | 0/119 vs 15/420 | 0.051 | no |
| no 328 seen by 15:00 | 1/6 vs 14/518 | 0.16 | no |
| Wednesday vs other weekdays | 5/102 vs 10/437 | 0.18 | no |
| single-aircraft week | 2/63 vs 13/357 | 1.0 | no |

## Findings

| # | Category | Evidence | Severity | Consequence |
|---|---|---|---|---|
| 1 | Multiple comparisons, uncontrolled | 11 explicit tests plus implicit splits (season, year, dow, hour); nothing pre-specified | moderate | every p between 0.007 and 0.05 above is nominal; expect ~0.5 false positives at that level among 11 tests |
| 2 | HARKing | the Sunday-vs-arrival-time switch was made after seeing both splits; the within-slot test (3/7 vs 1/5, p=0.58) cannot discriminate them | moderate | the 0.13 / 0.43 rates are a modelling choice justified by mechanism, not a finding; the alternative (Sunday 0.55) is equally consistent with the data |
| 3 | Non-independence | fog episodes span consecutive days (Oct 5-7 2026, Aug 11-13 2025); cancellations pair (Sep 27-28, Mar 2-3, May 20-21); Fisher assumes independent trials | moderate | all p-values optimistic; effective n for the fail-given-fog rate is nearer 25 episodes than 40 flights |
| 4 | Small-cell Type M risk | fail-given-below-mins rests on 9 events; the arrival-time split on 6 and 3 | moderate | the 0.43 late-arrival rate is a Jeffreys mean with interval 0.20-0.68; its magnitude is unreliable, its sign relative to early arrivals is nominal at p=0.044 |
| 5 | Measurement error in target | outcomes are inferred from ADS-B coverage gaps with confidence grades; 15 of 25 failures are "no Dornier near CEC" (absence), not observed events; 3 days have no archive | moderate | a misread gap inflates the clear-weather floor; the direction of bias is toward over-counting cancellations |
| 6 | Floor drift treated as regime change | 2025 vs 2026 p=0.028 nominal; could be fleet shrinkage, Hawthorne expansion, or coverage artefact in the 2026 ADS-B pull | minor-moderate | trailing-12-month floor (0.060) is defensible as recency weighting, not as proof of a structural change |
| 7 | Prev-day flag | 3/15, p=0.007; survives none of 11 corrections at 0.0045 but is the strongest secondary signal; mechanism plausible (multi-day disruption) | minor | ship as a flag, not a learned coefficient; 15 events |
| 8 | Weather stage | 978 days, 75 below-mins events, rolling-origin quarterly CV, AUC 0.86-0.90, Brier skill +0.13 to +0.19, calibration checked | none | adequately powered, pre-specified target, honest CV; this is the sound part |
| 9 | Fleet and dow splits | p = 1.0, 0.16, 0.18, 0.051 | none (correctly not shipped) | the 3-aircraft 0/119 is confounded with era and the aircraft is gone |

## Corrected inference

Supported: forecast features predict airport fog at arrival (large n, out of sample). Fog at arrival raises the chance the flight does not land, roughly 5x over clear weather (p<0.0001, intervals do not overlap). A red forecast preceded every fog failure in two years; a green forecast has preceded none in 333 flights.

Partially supported: later arrivals fail in fog more often than earlier ones (nominal p=0.044, 9 events, direction plausible, magnitude unreliable). A cancellation raises next-day cancellation odds (nominal p=0.007, 15 events). The clear-weather cancellation rate rose in 2026 (nominal p=0.028).

Insufficiently supported: that the Sunday excess is a Sunday effect rather than an arrival-time effect, or vice versa. Fleet availability as a predictor. Any day-of-week effect.

Overall confidence: medium. The verdict on the weather stage is high and would not change if any single excerpt were removed. The verdict on the crew stage is low-medium: every conditional rate rests on under 10 events, and the choice of conditioning variable is not identifiable from this sample.

## Weakest link and next discriminating check

Weakest link: P(fail | below minimums) and its split. Alternative most likely to overturn: the Sunday excess is a crew/scheduling effect that persists at 18:42, in which case the current 0.13 understates Sunday risk by ~4x on fog nights.

Discriminating check: the next 10 below-minimums arrivals at the new 18:42 Sunday slot. Three or more failures among them favours the Sunday story; zero or one favours arrival time. Alternatively the airline's monthly activity reports (records request drafted in `reports/inquiries.md`) give exact diversion counts by day and remove the ADS-B measurement error entirely.
