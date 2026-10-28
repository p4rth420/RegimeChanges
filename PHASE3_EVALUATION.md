# Phase 3 — Validation benchmark and held-out evaluation

## Executive summary

This phase compares two frozen sequential PELT detectors and a three-state Gaussian hidden Markov model (HMM) for NIFTY 50 regime descriptions. The PELT configuration was selected before opening the test period and was run once on the held-out test segment. The HMM was fit on training data and compared with PELT on validation only; it was not run or tuned on test.

On validation, HMM labels aligned more closely with PELT–RBF (72.7% agreement; Cohen's κ=0.55) than with PELT–L2 (60.2%; κ=0.38). The HMM made 23 validation transitions, so it changed states more often than either PELT detector's confirmed alerts.

On test, PELT–L2 issued 8 alerts and PELT–RBF issued 9. Their estimated break dates aligned particularly closely in 2025 and early/mid 2026, while they differed more in 2024 and at the end of the test period. Median estimated-break-to-alert delays were 23.5 sessions for L2 and 26 sessions for RBF. There are no ground-truth regime labels, so these results describe detector behavior and cross-method agreement; they do not establish detection accuracy or prove real-world regime changes.

## Experimental design

The Phase 1 chronological split contains 2,632 complete training rows (through 2020-12-31), 741 validation rows (2021-01-01 through 2023-12-29), and 696 test rows (2024-01-01 through 2026-10-27). Each feature row is lagged, so a decision dated *t* uses market information through the prior observed session.

| Component | Procedure |
|---|---|
| PELT–L2 | `vol_20`; L2 cost; penalty 10 |
| PELT–RBF | `log_return` and `vol_20`; RBF cost; penalty 5 |
| Shared PELT settings | Window 252; minimum segment size 20; jump 5; update every 5 sessions; 2 confirmations; tolerance 5; maximum boundary age 60; cooldown 20 |
| HMM benchmark | Three-state diagonal Gaussian HMM; fit on training only; 10 initializations; select highest training log likelihood; order state labels by training probability-weighted mean `vol_20` |
| HMM validation inference | Forward-filtered state probabilities, carrying the filter through from training; no backward smoothing |

The PELT training-fitted scales, RBF bandwidth, and descriptive-volatility thresholds remained frozen. The detector state was carried forward chronologically from validation into test. Test results were not used to revise any setting.

## Validation HMM comparison

The HMM fit converged. Comparison used 741 shared validation dates and the PELT daily descriptive labels. Break proximity counts use the PELT estimated feature break date and the nearest HMM state transition, with a five-feature-session tolerance.

| Validation comparison | Label agreement | Cohen's κ | Adjusted Rand index | PELT breaks near HMM transition |
|---|---:|---:|---:|---:|
| HMM vs PELT–L2 | 60.2% | 0.38 | 0.28 | 4/6 |
| HMM vs PELT–RBF | 72.7% | 0.55 | 0.46 | 7/9 |

The median nearest-transition gap was three feature sessions for both PELT methods. The HMM made 23 validation transitions, compared with 6 confirmed L2 alerts and 9 confirmed RBF alerts. Its higher agreement with RBF is descriptive method alignment, not evidence that RBF or HMM is correct.

## Held-out PELT results

The final run covered 696 test feature dates and 139 scheduled updates per detector.

| Detector | Test alerts | Median estimated-break-to-alert delay | Delay range | Test descriptive regime occupancy |
|---|---:|---:|---:|---|
| PELT–L2 | 8 | 23.5 sessions | 21–31 | 496 calm, 95 moderate, 105 stressed |
| PELT–RBF | 9 | 26 sessions | 21–26 | 376 calm, 215 moderate, 105 stressed |

### Test alert records

Dates below are the dates alerts became available in the as-of simulation. The event CSVs also record the estimated feature break, its source date, regime label, and delay.

| Detector | Alert date | Estimated feature break | Delay (sessions) | Descriptive label at detection |
|---|---|---|---:|---|
| L2 | 2024-06-21 | 2024-05-22 | 21 | stressed |
| L2 | 2024-07-22 | 2024-06-05 | 31 | stressed |
| L2 | 2024-08-20 | 2024-07-04 | 31 | calm |
| L2 | 2025-05-13 | 2025-04-08 | 21 | stressed |
| L2 | 2025-07-08 | 2025-06-09 | 21 | calm |
| L2 | 2026-03-23 | 2026-02-05 | 31 | moderate |
| L2 | 2026-05-04 | 2026-03-20 | 26 | stressed |
| L2 | 2026-06-02 | 2026-04-30 | 21 | moderate |
| RBF | 2024-02-27 | 2024-01-18 | 26 | moderate |
| RBF | 2024-06-28 | 2024-05-29 | 21 | stressed |
| RBF | 2024-08-12 | 2024-07-04 | 26 | calm |
| RBF | 2025-05-13 | 2025-04-08 | 21 | stressed |
| RBF | 2025-07-08 | 2025-06-09 | 21 | calm |
| RBF | 2026-03-16 | 2026-02-05 | 26 | moderate |
| RBF | 2026-04-24 | 2026-03-20 | 21 | stressed |
| RBF | 2026-06-09 | 2026-04-30 | 26 | moderate |
| RBF | 2026-09-17 | 2026-08-11 | 26 | calm |

### Timeline review

The saved timelines display the full post-training sequence from 2021 onward; the test period is the section from 2024 onward. In that section:

- Both methods identify the same estimated feature breaks on 2025-04-08 and 2025-06-09, and again on 2026-02-05, 2026-03-20, and 2026-04-30. Their alert dates differ by several sessions for some of these, consistent with different feature sets and update/confirmation paths.
- The 2024 break estimates are less uniform. Both find the 2024-07-04 break; other nearby estimates in May/June differ by about a week, and RBF also alerts on a January break that L2 does not.
- RBF issues a further alert for an 2026-08-11 estimated break; L2 has no nearby test alert. The detectors therefore do not tell an identical story throughout the test period.
- Elevated `vol_20` episodes broadly coincide with stressed labels, but the labels are segment descriptions produced by the detector, not independent event classifications.

These are visual and algorithmic observations only. Without independently defined event dates or labeled regimes, no false-positive rate, precision, recall, or statistical superiority can be calculated.

## Interpretation and limitations

1. PELT is an offline segmentation method rerun on trailing information available at each check. The logs approximate as-of use; they are not a live production feed.
2. Alert dates are later than estimated feature break dates because the system updates every five sessions and requires confirmation. Delay is operational lag, not an estimate of error against a known true break.
3. L2 uses only trailing volatility, while RBF uses returns and volatility. Differences may come from feature choice as well as cost function.
4. The three HMM labels are ordered by training volatility. The names calm/moderate/stressed are convenient descriptive labels, not externally validated market states.
5. HMM and PELT agreement is not independent validation because neither method supplies ground truth. HMM's 23 validation transitions also show that its state assignments can switch more often than confirmed PELT alerts.
6. Results depend on the downloaded historical price snapshot and data provider revisions. They should not be interpreted as causal explanations, investment advice, or a trading strategy evaluation.

## Conclusion

Keep the 252/20 PELT configuration recorded as the frozen research setting. The validation comparison favors RBF only in the limited sense that its labels aligned more closely with the HMM on this validation sample. The test timelines show several shared estimated breaks, especially in 2025 and early/mid 2026, but meaningful disagreements remain. The project has completed the planned detector comparison and one-time held-out run; any further performance claims would require independently labeled market events or a separately designed trading/backtesting study.

## Reproducibility artifacts

- `phase3_hmm.py` — HMM fitting and validation comparison.
- `tmp/full_nifty_20261028/phase3_hmm_validation/hmm_vs_pelt_validation.csv` — validation agreement metrics.
- `tmp/full_nifty_20261028/phase2_final_test/summary.csv` — split-level checks, alerts, and regime occupancy.
- `tmp/full_nifty_20261028/phase2_final_test/NIFTY50_l2_events.csv` and `NIFTY50_rbf_events.csv` — alert and estimated-break records.
- `tmp/full_nifty_20261028/phase2_final_test/NIFTY50_l2_timeline.png` and `NIFTY50_rbf_timeline.png` — full post-training timelines.
- `tmp/full_nifty_20261028/phase2_final_test/run_manifest.json` — frozen settings, input/output hashes, dependency versions, and test flag.
