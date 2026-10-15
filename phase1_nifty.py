#!/usr/bin/env python3
"""Causal Phase 1 data pipeline. Run --self-test for network-free verification."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SYMBOLS = {
    'NIFTY50': '^NSEI', 'BANK': '^NSEBANK', 'IT': '^CNXIT',
    'AUTO': '^CNXAUTO', 'FMCG': '^CNXFMCG', 'PHARMA': '^CNXPHARMA',
    'METAL': '^CNXMETAL', 'REALTY': '^CNXREALTY',
}
FEATURES = ['log_return', 'abs_return', 'vol_20', 'vol_60',
            'mean_20', 'momentum_20', 'drawdown', 'corr_nifty_60']


def clean_close(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Accept canonical Date,Close CSV or a single-ticker yfinance result."""
    frame = frame.copy()
    if 'Date' in frame.columns:
        frame = frame.set_index('Date')
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs('Close', level=0, axis=1)
        if frame.shape[1] != 1:
            raise ValueError('Expected one ticker per download')
        s = frame.iloc[:, 0]
    else:
        s = frame['Close']
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    if s.index.tz is not None:
        s.index = s.index.tz_convert('Asia/Kolkata').tz_localize(None)
    s.index = s.index.normalize()
    s = s.sort_index()
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]
    if s.index.has_duplicates:
        raise ValueError('Duplicate dates: resolve at source; never silently aggregate')
    s = pd.to_numeric(s, errors='raise').astype(float)
    if ((s <= 0) & s.notna()).any() or np.isinf(s).any():
        raise ValueError('Close must be positive and finite when present')
    s.name = 'Close'
    s.index.name = 'Date'
    return s


def causal_features(close: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """All exported numeric features at date t use information through t-1."""
    r = np.log(close / close.shift(1))  # A missing close breaks both adjacent returns.
    b = np.log(benchmark / benchmark.shift(1))
    f = pd.DataFrame(index=close.index)
    f['log_return'] = r
    f['abs_return'] = r.abs()
    for w in (20, 60):
        f[f'vol_{w}'] = r.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(252)
    f['mean_20'] = r.rolling(20, min_periods=20).mean()
    # Require a complete window, including intermediate prices.
    f['momentum_20'] = (close / close.shift(20) - 1).where(
        close.rolling(21, min_periods=21).count() == 21)
    f['drawdown'] = close / close.cummax() - 1
    f['corr_nifty_60'] = r.rolling(60, min_periods=60).corr(b)
    return f[FEATURES].shift(1).replace([np.inf, -np.inf], np.nan)


def split_scale(features: pd.DataFrame, train_end: str, validation_end: str):
    """No shuffling, interpolation, full-sample fitting, or target columns."""
    if pd.Timestamp(train_end) >= pd.Timestamp(validation_end):
        raise ValueError('train_end must precede validation_end')
    complete = features.notna().all(axis=1)
    labels = pd.Series('test', index=features.index, name='split')
    labels.loc[features.index <= pd.Timestamp(validation_end)] = 'validation'
    labels.loc[features.index <= pd.Timestamp(train_end)] = 'train'
    train = features.loc[complete & labels.eq('train')]
    if len(train) < 60:
        raise ValueError('Need at least 60 complete training rows; adjust dates/history')
    mean = train.mean()
    std = train.std(ddof=0)
    std = std.mask(std.abs() < 1e-12, 1.0)  # Constant NIFTY self-correlation is OK.
    scaled = (features - mean) / std
    params = {'mean': mean.to_dict(), 'scale': std.to_dict(),
              'fit_rows': len(train), 'fit_first': str(train.index.min().date()),
              'fit_last': str(train.index.max().date())}
    return labels, complete, scaled, params


def assert_causal(prices: pd.Series, benchmark: pd.Series):
    """Appending or modifying the future must not change an earlier feature."""
    full = causal_features(prices, benchmark)
    for k in sorted(set([min(90, len(prices)), len(prices)//2, len(prices)-1])):
        if k < 2:
            continue
        prefix = causal_features(prices.iloc[:k], benchmark.iloc[:k])
        pd.testing.assert_frame_equal(prefix, full.iloc[:k])
        altered = prices.copy()
        altered.iloc[k:] *= 7.0
        altered_b = benchmark.copy()
        altered_b.iloc[k:] *= 0.4
        pd.testing.assert_frame_equal(full.iloc[:k],
                                     causal_features(altered, altered_b).iloc[:k])
    # Even today's close must not influence today's decision features.
    k = len(prices)//2
    altered = prices.copy()
    altered.iloc[k] *= 2.0
    pd.testing.assert_series_equal(full.iloc[k], causal_features(altered, benchmark).iloc[k])


def self_test():
    rng = np.random.default_rng(17)
    dates = pd.bdate_range('2015-01-01', periods=1100)
    benchmark = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates)))), index=dates)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .015, len(dates)))), index=dates)
    close.iloc[150] = np.nan
    assert_causal(close, benchmark)
    f = causal_features(close, benchmark)
    assert pd.isna(f['log_return'].iloc[151])
    assert pd.isna(f['log_return'].iloc[152])
    np.testing.assert_allclose(f['log_return'].iloc[100], np.log(close.iloc[99]/close.iloc[98]))
    np.testing.assert_allclose(f['vol_20'].iloc[100],
        np.log(close/close.shift()).iloc[80:100].std(ddof=1)*np.sqrt(252))
    te, ve = str(dates[550].date()), str(dates[800].date())
    labels, valid, scaled, params = split_scale(f, te, ve)
    changed = f.copy()
    changed.loc[labels.ne('train')] *= 10
    _, _, scaled2, params2 = split_scale(changed, te, ve)
    assert params == params2
    pd.testing.assert_frame_equal(scaled.loc[labels.eq('train')], scaled2.loc[labels.eq('train')])
    assert labels.loc[:te].eq('train').all()
    assert all(valid.groupby(labels).sum() > 0)
    # Check CSV and yfinance MultiIndex parsing plus duplicate rejection.
    raw = pd.DataFrame({'Date': dates, 'Close': benchmark.to_numpy()})
    clean = clean_close(raw, '2015-01-01', '2020-01-01')
    multi = pd.DataFrame(benchmark.to_numpy(), index=dates,
                         columns=pd.MultiIndex.from_tuples([('Close', '^NSEI')]))
    pd.testing.assert_series_equal(clean, clean_close(multi, '2015-01-01', '2020-01-01'), check_freq=False)
    try:
        clean_close(pd.concat([raw, raw.iloc[:1]]), '2015-01-01', '2020-01-01')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate dates not rejected')
    print('PASS: lag timing, trailing volatility, prefix/future invariance, missing data, '
          'chronological splits, train-only scaling, CSV/MultiIndex parsing, duplicate rejection')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', help='Exclusive end date; defaults to today in India (today excluded)')
    p.add_argument('--train-end', default='2020-12-31')
    p.add_argument('--validation-end', default='2023-12-31')
    p.add_argument('--indices', nargs='+', choices=list(SYMBOLS), default=list(SYMBOLS))
    p.add_argument('--input-dir', type=Path, help='Offline NAME.csv files, each with Date,Close')
    p.add_argument('--output', type=Path, default=Path('phase1_output'))
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    today = datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()
    end = args.end or today
    if not (pd.Timestamp(args.start) < pd.Timestamp(args.train_end)
            < pd.Timestamp(args.validation_end) < pd.Timestamp(end)):
        p.error('Require start < train-end < validation-end < end')
    if pd.Timestamp(end) > pd.Timestamp(today):
        p.error('end cannot exceed today; incomplete current-session prices are excluded')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use an empty output directory to preserve the previous data snapshot')
    names = list(dict.fromkeys(['NIFTY50'] + args.indices))
    series, fetched = {}, {}
    for name in names:
        if args.input_dir:
            frame = pd.read_csv(args.input_dir / f'{name}.csv')
        else:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise SystemExit('Install yfinance or supply --input-dir. See the guide.') from exc
            frame = yf.download(SYMBOLS[name], start=args.start, end=end,
                interval='1d', auto_adjust=False, back_adjust=False, repair=False,
                keepna=True, progress=False, threads=False)
        if frame is None or frame.empty:
            raise ValueError(f'{name}: empty data; verify symbol or use a local index CSV')
        series[name] = clean_close(frame, args.start, end)
        fetched[name] = datetime.now(ZoneInfo('UTC')).isoformat()
    # Benchmark observed sessions define the calendar. Never forward/backfill.
    calendar = series['NIFTY50'].dropna().index
    if len(calendar) < 200:
        raise ValueError('Insufficient NIFTY50 history')
    prices = pd.DataFrame({n: s.reindex(calendar) for n, s in series.items()})
    source_date = pd.Series(calendar, index=calendar).shift(1)
    out = args.output
    for directory in ('raw', 'features', 'splits', 'plots'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    quality = {}
    for name in names:
        # Preserve canonical pre-alignment data, including missing closes.
        series[name].to_csv(out / 'raw' / f'{name}.csv')
        f = causal_features(prices[name], prices['NIFTY50'])
        assert_causal(prices[name], prices['NIFTY50'])
        labels, complete, scaled, params = split_scale(f, args.train_end, args.validation_end)
        counts = labels.loc[complete].value_counts().to_dict()
        if any(counts.get(s, 0) == 0 for s in ('train', 'validation', 'test')):
            raise ValueError(f'{name}: missing a complete train/validation/test partition')
        metadata = pd.DataFrame({'source_date': source_date, 'split': labels,
                                 'complete': complete}, index=calendar)
        metadata.index.name = 'Date'
        metadata.join(f).to_csv(out / 'features' / f'{name}_features.csv')
        for split in ('train', 'validation', 'test'):
            mask = complete & labels.eq(split)
            metadata.loc[mask].join(f.loc[mask]).to_csv(out / 'splits' / f'{name}_{split}.csv')
            metadata.loc[mask].join(scaled.loc[mask]).to_csv(out / 'splits' / f'{name}_{split}_scaled.csv')
        (out / 'features' / f'{name}_scaler.json').write_text(json.dumps(params, indent=2))
        quality[name] = {'ticker': SYMBOLS[name], 'source_rows': len(series[name]),
            'missing_on_nifty_calendar': int(prices[name].isna().sum()),
            'off_calendar_rows': int((~series[name].index.isin(calendar)).sum()),
            'first_valid_close': str(prices[name].first_valid_index()),
            'last_valid_close': str(prices[name].last_valid_index()),
            'incomplete_feature_rows': int((~complete).sum()), 'split_rows': counts}
    prices.index.name = 'Date'
    prices.to_csv(out / 'aligned_close.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for name in names:
        f = causal_features(prices[name], prices['NIFTY50'])
        axes[0].plot(prices.index, prices[name], label=name, linewidth=.7)
        if name == 'NIFTY50':
            axes[1].plot(f.index, f.log_return, linewidth=.6)
            axes[2].plot(f.index, f.vol_20, linewidth=.7)
    axes[0].set_ylabel('Index level'); axes[0].legend(ncol=4)
    axes[1].set_ylabel('NIFTY50 lagged log return')
    axes[2].set_ylabel('Lagged 20-session vol (annual)')
    for ax in axes:
        for boundary in (args.train_end, args.validation_end):
            ax.axvline(pd.Timestamp(boundary), color='black', linestyle='--', linewidth=.8)
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'plots' / 'overview.png', dpi=150); plt.close(fig)
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'matplotlib', 'yfinance'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    hashes = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in out.rglob('*') if path.is_file()}
    manifest = {'start_inclusive': args.start, 'end_exclusive': end,
        'train_end_inclusive': args.train_end, 'validation_end_inclusive': args.validation_end,
        'source': 'local CSV' if args.input_dir else 'Yahoo Finance via yfinance',
        'retrieved_utc': fetched, 'versions': versions, 'quality': quality,
        'feature_timing': 'date t uses closes through preceding NIFTY observed session only',
        'scaler_policy': 'training rows only, frozen thereafter; not valid before training is complete',
        'limitations': 'Historical download is not a point-in-time vintage; no guarantee against provider revisions.',
        'sha256': hashes}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Created {out.resolve()} | causal checks passed | inspect manifest.json for coverage')


if __name__ == '__main__':
    main()
