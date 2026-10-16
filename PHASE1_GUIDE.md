# Phase 1 — Indian market data and causal features

## Purpose

Prepare NIFTY 50 and selected NIFTY sector index histories for the regime-change ML class project. This phase produces data, trailing features, chronological partitions and plots. It does not fit PELT/HMM models, generate regime labels or backtest trades.

Default run: **NIFTY 50 and Bank NIFTY only**. Other sectors are opt-in until their download coverage is verified. Available sectors: Bank, IT, Auto, FMCG, Pharma, Metal and Realty. These are index-level histories, not today's constituents reconstructed backward. This is a selected sector panel, not every NIFTY sector index.

## Run

Python 3.10+ is required. Install packages in your virtual environment or Anaconda environment:

```bash
python -m pip install numpy pandas matplotlib yfinance tzdata
python phase1_nifty.py --self-test
python phase1_nifty.py --start 2010-01-01 --end 2026-10-15 --output phase1_output
```

The end date is **exclusive**. The example includes sessions before 15 October 2026. Omit `--end` to exclude today automatically using the India timezone. An end date beyond today is rejected, even if requested explicitly. Default splits are training through 2020-12-31, validation from 2021-01-01 through 2023-12-31, and test from 2024-01-01 onward. These are initial research choices: freeze them before inspecting test performance.

To explicitly request additional sectors:

```bash
python phase1_nifty.py --indices BANK IT AUTO --output phase1_small
```

NIFTY50 is always included. Use an empty output folder for each run; the script refuses to overwrite a previous snapshot. Any empty download or insufficient partition fails explicitly. A failed run can leave partial outputs: only consider a run complete when it prints success and writes `manifest.json`.

### Source symbols

| Internal name | Yahoo symbol |
|---|---|
| NIFTY50 | `^NSEI` |
| BANK | `^NSEBANK` |
| IT | `^CNXIT` |
| AUTO | `^CNXAUTO` |
| FMCG | `^CNXFMCG` |
| PHARMA | `^CNXPHARMA` |
| METAL | `^CNXMETAL` |
| REALTY | `^CNXREALTY` |

The default downloads use `^NSEI` and `^NSEBANK`. Other table entries are optional, not newly verified replacement symbols. An empty response alone does not establish that a ticker is wrong.

Symbols are configurable in `SYMBOLS` or with repeatable `--ticker NAME=SYMBOL` arguments. Overrides must refer to the same intended index; the manifest records the mapping. For example, `--indices BANK --ticker BANK=^NSEBANK` explicitly sets the existing Bank mapping. An override does not automatically add an index: also select it with `--indices`. Provider availability and historical coverage may vary. Verify each downloaded series against the intended index before analysis. Never silently substitute an ETF or another index if a symbol fails. This implementation uses unadjusted `Close`, with automatic adjustment, back adjustment and automated repair disabled. It models price index returns, not total returns with reinvested dividends. Index volume is not used.

### Offline / alternative-source mode

If Yahoo is unavailable, obtain daily index closes from a suitable source such as NIFTY Indices and convert them to canonical CSV files. Create one file for every requested internal name (e.g. `NIFTY50.csv`, `BANK.csv`, `IT.csv`):

```csv
Date,Close
2019-01-02,10792.50
2019-01-03,10672.25
```

The two rows illustrate the format only. Supply sufficient history for all three partitions.

```bash
python phase1_nifty.py --input-dir index_csvs --indices BANK IT --output phase1_offline
```

Dates must be ISO dates; closes must be positive numeric values or missing. Duplicate dates, nonnumeric values and infinite values raise errors. No assumption is made that different source providers are interchangeable; record the actual provenance separately when using local CSV files.

## Exact timing contract

**A feature row with Date = t is available before session t opens. It uses closes through the preceding observed NIFTY 50 session only.** `source_date` records that preceding session. The feature calculations first use trailing information at each close and then shift every feature forward by one session. There are no future-return targets in the output.

For an illustrative Monday/Tuesday pair: Monday's final close and trailing statistics may inform Tuesday's opening decision. Tuesday's close cannot inform Tuesday's opening decision. The aligned close file is an archival/diagnostic input and must not be joined into the model as a same-day feature.

| Feature | Formula before the one-session lag | Interpretation |
|---|---|---|
| `log_return` | log(C_s / C_(s-1)) | Latest completed-session log return |
| `abs_return` | absolute log return | Daily movement magnitude |
| `vol_20`, `vol_60` | sample standard deviation of last w log returns × sqrt(252) | Annualized trailing volatility |
| `mean_20` | mean of last 20 log returns | Trailing average daily return |
| `momentum_20` | C_s / C_(s-20) − 1 | Simple 20-session return; requires all 21 closes |
| `drawdown` | C_s / max(C_0,...,C_s) − 1 | Fall from the highest observed close since dataset start |
| `corr_nifty_60` | trailing 60-return Pearson correlation with NIFTY50 | Sector co-movement; NIFTY50 self-correlation is essentially 1 |

All return/volatility values are decimals: 0.01 means 1%; volatility 0.20 means 20% annualized. Annualization assumes 252 sessions. Drawdown depends on the chosen history start and is not an all-time measure. Windows count observed NIFTY sessions rather than calendar days. NIFTY self-correlation is redundant and can be removed from the Phase 2 model.

## Missing data and alignment

- NIFTY50's dates with observed closes define the calendar. Sector series are reindexed to that calendar, never to future availability or a full-sample common intersection.
- No forward fill, backward fill, interpolation, centered rolling window or future-based outlier replacement is used.
- Missing sector closes remain missing. Returns touching the missing close are missing, avoiding multi-session returns being mistaken for one-session returns.
- Rolling mean, volatility and correlation require a complete return window. Momentum requires a complete price window. Initial warm-up and subsequent gaps therefore create incomplete rows.
- Full feature files retain incomplete rows and a `complete` flag. Split exports contain only complete rows. Coverage is reported separately for each index.
- Do not concatenate sector split exports using their row positions. Join by Date; any later multivariate model must explicitly handle missing features. Dropping rows changes spacing, which matters for CPD segment lengths. Preserve original dates and inspect gaps in Phase 2.
- The benchmark itself can have unobserved missing sessions. Without an independent exchange calendar, the pipeline cannot distinguish every absent benchmark observation from a holiday. Inspect source coverage; this is not a certified exchange-calendar validation.

## Chronology and scaling

Partitions use fixed date boundaries, without random shuffling. Past observations may enter validation/test rolling windows, as they would in real time. Future observations never enter earlier windows.

For each index, the scaler's mean and population standard deviation are fitted on complete training rows only and then frozen. Constant columns use scale 1. Scaler parameters and fit dates are saved separately. Validation and test values never affect fitted parameters.

**Scaled training rows are retrospective training inputs.** The frozen scaler uses the whole training partition, so it must not be used to simulate trades earlier within that partition. Evaluate the resulting trained pipeline only after training ends. For a training-period backtest, fit preprocessing on a strictly preceding calibration window and refit forward as needed. The unscaled features themselves are causal throughout.

Before changing penalties, feature sets or models, use validation data. Open the test partition only for final evaluation; repeated tuning against test charts or results creates selection leakage. Whole-history EDA plots are provided for reporting; restrict research decisions to train/validation data.

## Files generated by a successful run

```text
phase1_output/
  raw/NAME.csv                         # canonical source Close before alignment
  aligned_close.csv                    # unlagged archival prices; not model features
  features/NAME_features.csv           # lagged features, source_date, split, complete
  features/NAME_scaler.json            # training-only fitted parameters
  splits/NAME_train.csv                # complete unscaled feature rows
  splits/NAME_validation.csv
  splits/NAME_test.csv
  splits/NAME_train_scaled.csv         # same rows, training-fitted scaling
  splits/NAME_validation_scaled.csv
  splits/NAME_test_scaled.csv
  plots/overview.png                   # prices, lagged NIFTY return and volatility
  manifest.json                       # coverage, versions, timing, fetch times, hashes
```

NAME is an internal name from the symbol table. Metadata columns (`source_date`, `split`, `complete`) are not model inputs. Use only the names in the script's `FEATURES` constant. Snapshots preserve the downloaded canonical closes and the manifest hashes so later downloads do not silently replace the research dataset. Library versions are recorded; preserve those versions when reproducing a run.

## Verification

`--self-test` uses generated prices only, without market downloads. It verifies:

1. A feature at session t equals the appropriate statistic through t−1.
2. Appending future prices leaves existing feature rows unchanged (prefix invariance).
3. Changing future prices and future benchmark prices leaves earlier features unchanged.
4. Changing today's close cannot change today's exported features.
5. Missing prices break affected returns rather than being filled.
6. Altering validation/test features cannot change the training-fitted scaler or scaled training data.
7. Chronological partitions, CSV/MultiIndex parsing and duplicate-date rejection work.

Each normal run also executes causal feature checks per index. A separate synthetic end-to-end run was used to verify file generation and split exports. Synthetic data is a software test, not financial evidence. Live Yahoo downloads were not executed in the authoring environment; run locally and inspect the manifest before treating the dataset as ready.

## Limits of the no-look-ahead claim

The implemented feature transformations are causal and scaling respects the training boundary. **A current download of historical prices is not a point-in-time historical data vintage.** Providers may revise past values; index histories may include backfilled periods. Disabling automatic adjustment does not eliminate these issues. Strict historical availability guarantees require vintage data and index launch/backfill metadata. Freeze the current snapshot, document this limitation, and do not claim a fully point-in-time dataset from these files alone.

Later phases must preserve the same timing contract:

- Full-history offline PELT segmentation is retrospective and can be used for descriptive analysis. It cannot supply tradable historical labels.
- A trading experiment must rerun detection using only data available at the decision date, then record both the estimated historical break date and the actual detection/decision date. A break found today cannot trigger a trade last month.
- HMM inference must use forward filtering on the available prefix, not full-sequence smoothing/Viterbi states presented as real-time decisions. Retrain using only the available history.
- Features dated t can inform a trade at t's open. For execution, use a tradeable instrument and explicit next-open/fill prices, costs and turnover. Multiplying an opening decision by the entire prior-close-to-current-close return incorrectly includes overnight performance before the fill.
- NIFTY index levels are analysis benchmarks, not directly executable prices. A later ETF/futures backtest needs appropriate data and costs.

## Sources

- Project brief: *Identifying Regime Changes in Financial Markets* (supplied PDF).
- yfinance download API: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html — explicit inclusive start, exclusive end and adjustment options.
- Official NIFTY sector list: https://niftyindices.com/indices/equity/sectoral-indices
- Official historical-data entry point: https://niftyindices.com/reports/historical-data

## Next phase

Start with NIFTY50 unscaled or training-scaled return/volatility features. Establish a descriptive PELT baseline, then implement prefix-only detection for prospective evaluation. Choose penalties and minimum segment lengths on training/validation data. Keep the final test partition untouched until the design is frozen.

## Running in Jupyter

Keep the script as a `.py` file and run:

```python
%run phase1_nifty.py --self-test
%run phase1_nifty.py --output phase1_nifty_bank
```

Alternatively, import it and pass an explicit argument list to avoid Jupyter's injected `-f` argument:

```python
import phase1_nifty
phase1_nifty.main(["--output", "phase1_nifty_bank"])
```

Use a new output directory when retrying. No live download availability guarantee is made; a listed Yahoo quote may still have unavailable historical downloads. Do not choose sectors based on test-period investment performance.
