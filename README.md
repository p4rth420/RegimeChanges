# NIFTY 50 Regime Change Detection

A time-series project that studies changes in the statistical behaviour of the Indian equity market. The first analysis focuses on the NIFTY 50. It derives causal daily return and volatility features, detects candidate structural changes with PELT, and evaluates the detector in chronological order.

The project has two stages:

1. **Phase 1 — Data and features:** Download or import index closes, validate and align the dates, calculate trailing features, and create chronological train/validation/test splits.
2. **Phase 2 — Change detection:** Compare PELT with an L2 cost against PELT with an RBF kernel. The detector is rerun on the history available at each decision date and writes an as-of decision log.

This is an unsupervised change-point project: there are no ground-truth regime labels in the input. Detected segments and volatility labels are analytical estimates, not predictions or proof of market events.

## Focus and data

Start with NIFTY 50 only. The Phase 1 script currently defaults to both NIFTY 50 and Bank NIFTY, so explicitly select NIFTY 50 when running it:

```bash
python phase1_nifty.py --indices NIFTY50 --output phase1_nifty50
```

Yahoo Finance symbol: `^NSEI`. The pipeline uses the daily closing index level with automatic price adjustment disabled. It is a price-index series, not a total-return series with reinvested dividends. The data are used for research and may be revised by the provider; they are not a point-in-time vintage.

The default date splits are chronological:

- Training through 31 December 2020.
- Validation from 1 January 2021 through 31 December 2023.
- Test from 1 January 2024 onward.

Do not shuffle time-series rows. Use training to fit transformations, validation to select model settings, and the test period once for final evaluation.

## Requirements

Python 3.10 or newer is recommended. Install the dependencies into the same environment used by Jupyter:

```bash
python -m pip install numpy pandas matplotlib yfinance tzdata scipy ruptures
```

In a notebook, use `%pip install ...` so packages install into the active kernel.

## Run in Jupyter on Windows

Place `phase1_nifty.py` and `phase2_cpd.py` in the notebook's working directory, or give their full paths. Run the Phase 1 verification and data build:

```python
%run phase1_nifty.py --self-test
%run phase1_nifty.py --indices NIFTY50 --output "D:\phase1_nifty50"
```

Then run Phase 2 verification and validation analysis:

```python
%run phase2_cpd.py --self-test
%run phase2_cpd.py --input "D:\phase1_nifty50" --indices NIFTY50 --output "D:\phase2_nifty50_validation"
```

The Phase 2 default excludes test-period observations. After choosing settings on validation and freezing the design, run the final evaluation once:

```python
%run phase2_cpd.py --input "D:\phase1_nifty50" --indices NIFTY50 --output "D:\phase2_nifty50_final" --include-test
```

Use a new, empty output folder for each run. The scripts refuse to overwrite nonempty output folders so earlier snapshots and decision logs are preserved.

## Run the complete workflow in Jupyter

Open `Regime_Change_Pipeline.ipynb` from the project root and run its cells from top to bottom. It installs the dependencies, runs both scripts’ synthetic self-tests, builds a NIFTY 50-only Phase 1 snapshot, inspects its coverage, then runs Phase 2 on validation data with the test period excluded.

In the setup cell, choose `DATA_SOURCE = "yahoo"` to download NIFTY 50 using `^NSEI`, or `DATA_SOURCE = "local_csv"` and set `LOCAL_CSV_DIR` to a folder containing `NIFTY50.csv` with `Date,Close` columns. Change `PHASE1_OUTPUT` and `PHASE2_OUTPUT` when rerunning so each run writes to a fresh folder. The optional final cell shows how to open the test period once the model design is frozen.

## Run from a terminal

```bash
python phase1_nifty.py --self-test
python phase1_nifty.py --indices NIFTY50 --output phase1_nifty50
python phase2_cpd.py --self-test
python phase2_cpd.py --input phase1_nifty50 --indices NIFTY50 --output phase2_nifty50_validation
```

For a reproducible project run, keep the downloaded Phase 1 folder unchanged and point Phase 2 at that folder. Phase 2 does not download market prices. Phase 1 downloads when run unless `--input-dir` is provided to use local CSVs.

## Dataset and features

Phase 1 writes:

```text
phase1_nifty50/
  raw/NIFTY50.csv
  aligned_close.csv
  features/NIFTY50_features.csv
  features/NIFTY50_scaler.json
  splits/NIFTY50_train.csv
  splits/NIFTY50_validation.csv
  splits/NIFTY50_test.csv
  plots/overview.png
  manifest.json
```

`raw/NIFTY50.csv` stores source closes. `aligned_close.csv` is an unlagged price-level archive; do not use its same-day close as a before-open feature. The model inputs are in `features/NIFTY50_features.csv`:

- `log_return` and `abs_return` describe the latest available session move.
- `vol_20` and `vol_60` are trailing sample standard deviations of returns, annualized by √252.
- `mean_20` is the trailing average daily log return.
- `momentum_20` is the trailing 20-session percentage change in index level.
- `drawdown` is the decline from the highest index level observed since the dataset start.
- `corr_nifty_60` compares NIFTY 50 with itself. It is effectively one and is redundant for the NIFTY-only model; Phase 2 excludes it.

Each feature row dated `t` is shifted so that it uses information through the previous observed session. `source_date` records that session. No forward-fill, backward-fill, interpolation or random split is used. Initial rows are incomplete while rolling windows warm up; use rows where `complete` is true for models that need all features. In a NIFTY-only analysis, `corr_nifty_60` can be excluded rather than treated as an informative predictor.

The scaled split files use statistics fitted on training rows only. Phase 2 independently fits its own scaler, RBF bandwidth and descriptive volatility cutoffs using training data and freezes them for later dates.

## Phase 2 methods

- **PELT–L2:** applied to `vol_20` to identify changes in its mean level.
- **PELT–RBF:** applied to `log_return` and `vol_20` to find broader distribution changes in the joint features.

Phase 2 repeatedly runs an offline PELT algorithm on a trailing prefix of history. By default, it updates every five sessions, uses a 252-session maximum window, requires a minimum 20-session segment and applies confirmation/cooldown rules before emitting an alert. These are starting settings, not optimized or statistically calibrated market parameters. Test model settings on validation only.

The estimated break date can move as new observations arrive. The logged `decision_date` records when an alert was actually available; past alerts are not backdated. Because Phase 1 features are lagged, a decision on date `t` uses data through the preceding session.

## Look-ahead controls

- Feature windows are trailing and shifted one session.
- Chronological train, validation and test splits are not shuffled.
- Scaling and descriptive thresholds are fitted on training only.
- The RBF bandwidth is estimated from training data only.
- PELT sees only the prefix available at the current decision date; the terminal endpoint is not called a change point.
- Test data are excluded from Phase 2 by default.
- The Phase 1 manifest records data hashes; Phase 2 verifies source feature hashes, including Windows-style manifest paths.

An offline segmentation of the entire history is retrospective: its historical boundaries can use observations from later dates. Do not present those as real-time historical alerts. This project's sequential logs are an as-of simulation, not a live market feed or scheduled production service.

## Outputs

Phase 2 writes for each index/model pair:

```text
NIFTY50_l2_decisions.csv
NIFTY50_l2_events.csv
NIFTY50_l2_calibration.json
NIFTY50_l2_timeline.png
```

It also writes equivalent RBF files, `summary.csv`, and `run_manifest.json`. The decisions file records every as-of date, current descriptive volatility bucket, detector update status and latest estimated boundary. The events file records confirmed alert dates separately from estimated break dates. The timeline marks actual alert dates in red.

## Validation and interpretation

Run each script's `--self-test` before the data workflow. The tests use synthetic data and check causal feature timing, future-data invariance, training-only calibration, missing-data handling and detection of a known artificial break. They verify software behaviour, not financial performance.

A change point indicates a statistical shift under a chosen feature, cost and penalty. It does not identify the event's cause. Segment count and regime labels can change with the window, penalty, minimum segment length and input features. Report those settings and discuss detection delay and false alarms; do not interpret alerts as investment advice.

## Project files

- `phase1_nifty.py` — data download, preprocessing and feature generation.
- `PHASE1_GUIDE.md` — Phase 1 methodology and data dictionary.
- `phase2_cpd.py` — sequential PELT–L2 and PELT–RBF runs.
- `PHASE2_GUIDE.md` — model timing, outputs, settings and limitations.
- `README.md` — project overview and quick start.
- `Regime_Change_Pipeline.ipynb` — ordered Jupyter workflow for both phases.

## Sources

- Supplied project synopsis: *Identifying Regime Changes in Financial Markets*.
- NIFTY Indices, sectoral index catalogue: https://niftyindices.com/indices/equity/sectoral-indices
- yfinance download API: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- `ruptures` PELT documentation: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/
- `ruptures` RBF cost documentation: https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costrbf/
