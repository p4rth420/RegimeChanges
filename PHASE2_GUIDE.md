# Phase 2 — Sequential change point detection

## What you are building

This phase reads the completed Phase 1 dataset and compares two unsupervised change detection pipelines:

| Pipeline | Input features | Cost detects |
|---|---|---|
| PELT–L2 | `vol_20` | Changes in the average level of trailing volatility |
| PELT–RBF | `log_return`, `vol_20` | Broader changes in the distribution of the joint features |

PELT optimizes a penalized segmentation; it is not a supervised classifier. The RBF cost is a kernel method. These pipelines use different feature sets, so differences cannot be attributed solely to the cost function. Their performance must be evaluated in Phase 3. This phase does not implement the HMM benchmark or a portfolio backtest.

The implementation reruns an offline algorithm sequentially on trailing history. It is an **as-of simulation**, not an intrinsically online PELT algorithm. It never treats a retrospectively estimated break as a trade that could have happened on that historical date.

## Run in your Jupyter notebook

Save `phase2_cpd.py` in the notebook's working folder, or supply its full path. Your Phase 1 folder is `D:\phase1_bank_v2`.

Install dependencies into the active notebook environment:

```python
%pip install numpy pandas scipy matplotlib ruptures
```

Run verification first:

```python
%run phase2_cpd.py --self-test
```

Then run the validation experiment:

```python
%run phase2_cpd.py --input "D:\phase1_bank_v2" --output "D:\phase2_validation"
```

This reads NIFTY50 and BANK from your Phase 1 folder. There are no Yahoo downloads in Phase 2. Use the full Phase 1 folder, including `manifest.json` and the `features` subfolder. The earlier version of Phase 1 is supported: Phase 2 reads its unscaled feature files and manifest, not its CLI defaults.

From a terminal, replace `%run` with `python`. From an imported module:

```python
import phase2_cpd
phase2_cpd.main([
    "--input", r"D:\phase1_bank_v2",
    "--output", r"D:\phase2_validation"
])
```

Passing an explicit list avoids Jupyter's injected `-f` argument. Use an empty, new output folder for each experiment. A successful run ends with `Created ... | test excluded` and produces `run_manifest.json`. A failed run may leave partial outputs; do not treat those as complete.

## Timing and the prevention of look-ahead bias

1. **Verify the Phase 1 snapshot.** Feature CSV hashes must match the Phase 1 manifest. Dates must be ordered and unique. Each row's `source_date` must equal the previous observed Phase 1 session. Split labels must agree with manifest date boundaries. These checks validate the input contract, not the original provider's historical availability.
2. **Calibrate on training only.** Phase 2 independently fits a standard scaler on finite training rows for each pipeline. The RBF bandwidth is the inverse median positive squared distance in a deterministic subsample of at most 500 standardized training rows. It remains frozen during validation/test. No automatic bandwidth heuristic is allowed to see the whole evaluation series.
3. **Start after training ends.** No purported real-time decisions are generated during the period used to fit preprocessing. Default Phase 1 training ends on 2020-12-31.
4. **Slice the available prefix.** For a decision on date t, the detector sees only feature rows through t. Phase 1 already lagged those rows, so the newest source close is from the prior observed session. Each PELT instance and its cost cache are created fresh for the current window.
5. **Record decisions when they occur.** Break estimates can move on later runs. Earlier decision and event records are never backfilled or rewritten. The chart's red lines show alert dates, not historical break estimates.
6. **Keep the test period closed by default.** Only data through the Phase 1 validation end date enters calibration, detection and plots. Full-file bytes are hashed for integrity, but test feature values are excluded from computations unless `--include-test` is explicitly selected.

You can calculate an earlier estimated break while still being causal: the relevant action date is when the detector first had enough evidence to issue its alert. No parameter choice or future-confirmed label is carried backward into past decisions.

## Default experiment settings

| Argument | Default | Meaning |
|---|---:|---|
| `--window` | 252 | At most 252 latest feature rows per detection run |
| `--min-size` | 20 | Each segment requires at least 20 rows |
| `--jump` | 5 | Restrict candidate boundaries to a five-row grid in each window |
| `--every` | 5 | Recompute every five observed sessions, anchored at the first post-training row |
| `--penalty-l2` | 10.0 | L2 segmentation penalty |
| `--penalty-rbf` | 5.0 | RBF segmentation penalty; not numerically comparable to L2 |
| `--confirmations` | 2 | Require a nearby recent boundary in two successive detector updates |
| `--tolerance` | 5 | Allowed boundary difference from the first estimate in a confirmation cluster |
| `--max-age` | 60 | Only boundaries within the last 60 feature sessions may generate a new alert |
| `--cooldown` | 20 | At least 20 feature sessions between alerts |

These are initial engineering settings, **not optimized financial results**. The minimum segment size, trailing-volatility window, update cadence and confirmation rule introduce detection delay. Five-session updates mean signals are not refreshed daily. No statistical significance or false-alarm guarantee is attached to an alert.

A full window is not required: a check may run once at least `2 × min-size` consecutive finite feature rows are available. Training observations can supply the initial trailing window. This is permissible because they are already known when validation begins.

Because a rolling window moves, its candidate boundary grid moves too. Tolerance helps associate nearby estimates, but duplicate or missed real-world events remain possible. A new alert is allowed only if its estimated boundary is later than the previously alerted boundary plus tolerance. This conservative rule suppresses repeated reports; it may also miss closely spaced changes.

To try a different setting on validation only:

```python
%run phase2_cpd.py --input "D:\phase1_bank_v2" --output "D:\phase2_validation_p15" --penalty-l2 15 --penalty-rbf 7 --min-size 30
```

For a smaller initial run:

```python
%run phase2_cpd.py --input "D:\phase1_bank_v2" --output "D:\phase2_nifty_only" --indices NIFTY50
```

Penalty increases generally discourage segmentation. Do not pick settings simply because they produce more alerts or attractive charts. Predefine comparison criteria; detection accuracy, robustness and delay belong in Phase 3. Preserve all tried configurations to avoid selective reporting.

## Descriptive regime labels

At each detector update, the script averages the unscaled `vol_20` values in the latest estimated segment. It compares that average with the training-period 1/3 and 2/3 volatility quantiles:

- `calm`: average at or below the lower training cutoff.
- `moderate`: between the two cutoffs.
- `stressed`: average at or above the upper cutoff.
- `unknown`: missing features or insufficient consecutive observations after a gap.

The cutoffs are fitted using training history only, separately per index. These labels are descriptive volatility buckets, not supervised truth, forecasts or proof of a financial crisis. A PELT boundary can reflect changes other than volatility level, particularly with RBF.

The current label is refreshed at scheduled checks and carried forward between them. It can change without a confirmed alert, because alert confirmation and current segment description are separate outputs. If no boundary exists in the current window, the segment is the whole available window; this does not mean the market has never changed in its full history.

## Missing data

The script retains the Phase 1 calendar instead of dropping gaps and treating separated dates as consecutive. A missing model feature resets the consecutive-history count, pending boundary confirmation and current regime. Detection resumes only after sufficient consecutive finite observations have accumulated. L2 requires `vol_20`; RBF requires both `log_return` and `vol_20`.

Long exchange holidays do not themselves reset the window: session counts use observed Phase 1 dates. The completeness of the underlying benchmark calendar remains a Phase 1 limitation.

## Output files

Each index/model pair generates:

```text
NIFTY50_l2_decisions.csv
NIFTY50_l2_events.csv
NIFTY50_l2_calibration.json
NIFTY50_l2_timeline.png
```

Equivalent files are written for RBF and BANK. Global outputs are `summary.csv` and `run_manifest.json`.

### Decision log

One immutable as-of result per post-training date:

| Column | Meaning |
|---|---|
| `Date` | When the decision features were available before the session opens |
| `source_date` | Latest close date represented by the Phase 1 row |
| `split` | validation or test |
| `checked` | Whether PELT ran on this date |
| `status` | updated, carried_forward, warming_up or missing_features |
| `alert` | Whether a confirmed, eligible new break alert was issued |
| `regime` | Current descriptive volatility bucket |
| `segment_mean_vol_20` | Current segment's average trailing annualized volatility |
| `last_update_date` | Date of the most recent detector update |
| `estimated_break_feature_date` | First feature-row date in the newest estimated segment |
| `estimated_break_source_date` | Source close date represented by that boundary feature |

The estimated break columns are carried forward with the last check. Missing values mean no current estimated boundary or no current usable state.

### Event log

Contains only confirmed alerts. `decision_date` is the date the signal could first be acted on under this simulation. `lagged_feature_delay_sessions` counts feature sessions between the estimated boundary and alert. It is **not** an accuracy measure against the true, unknown market break. Because features are lagged, the estimated source date is one observed session earlier than the feature boundary date. `confirmation_checks` records consecutive matching detector updates.

PELT returns a terminal endpoint as well as internal boundaries. The script explicitly removes that endpoint; it must never be interpreted as a new change at the end of every window.

### Summary and manifest

`summary.csv` reports checks, alerts and regime occupancy separately by index, pipeline and evaluation split. These are operational diagnostics, not Sharpe ratios or detection precision. `run_manifest.json` records configuration, input/output hashes, dependency versions and whether test data was enabled. Calibration JSON files preserve fitted feature scales, bandwidth and volatility thresholds.

## Validation performed on the code

`--self-test` checks both pipelines with a synthetic series containing a strong, known volatility-level change:

- Detection of an estimated break near the synthetic boundary.
- Identical historical decision/event logs when future rows are appended.
- Identical earlier decisions after future features are perturbed.
- Identical training calibration when future data is changed or truncated.
- Correct lagged source/decision chronology.
- Reset after missing model features.
- Exclusion of the final PELT endpoint on a constant signal.

A separate synthetic Phase 1 → Phase 2 integration run checks manifest integrity, file loading, output generation and default exclusion of test data. Synthetic checks demonstrate software behavior, not performance on your actual downloaded Indian market data. Your local Phase 1 folder has not been uploaded, so those financial results are not available here.

## Opening the test period later

Only after you freeze the feature sets, penalties, window and alert rules:

```python
%run phase2_cpd.py --input "D:\phase1_bank_v2" --output "D:\phase2_final" --include-test
```

Use the same frozen arguments from the selected validation experiment. The current implementation keeps training-fitted preprocessing frozen even in test; it does not automatically retrain on validation. Detection history and alert state continue chronologically from validation into test, which is appropriate for an ongoing system. Do not run this command repeatedly while tuning.

## Scope and remaining limitations

This is a research detector, not a completed trading strategy. A Phase 3 benchmark can add an HMM with forward-filtered probabilities and documented historical-event comparisons. Phase 4 must use executable instruments, correct fill timing and transaction costs. Never multiply an opening signal by a return that includes an overnight move before the fill.

Current historical Yahoo downloads can contain revisions or pre-launch backfilled index history. Phase 2 preserves Phase 1's timing contract but cannot create a historical data vintage that was not provided. Overlapping rolling-volatility observations are serially dependent; generic RBF distribution-change interpretation and penalties do not establish calibrated financial significance.

## Technical sources

- PELT API and segmentation parameters: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/
- RBF cost and bandwidth behavior: https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costrbf/
- Killick, Fearnhead and Eckley (2012), *Optimal Detection of Changepoints With a Linear Computational Cost*.

For a code walkthrough, read `fit_training`, `breakpoints`, then `walk_forward` in that order.

## Windows snapshot compatibility

Phase 1 may serialize manifest hash keys with Windows backslashes, such as `features\NIFTY50_features.csv`. Phase 2 normalizes these keys to forward slashes before lookup. Actual SHA-256 verification remains enabled; do not delete the integrity check or regenerate hashes merely to bypass a mismatch. Both native Windows and POSIX manifest paths are supported.
