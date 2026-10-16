#!/usr/bin/env python3
"""Phase 2: prefix-only rolling PELT on Phase 1 features; no market downloads."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.spatial.distance import pdist

MODEL_FEATURES = {'l2': ['vol_20'], 'rbf': ['log_return', 'vol_20']}


@dataclass(frozen=True)
class Config:
    window: int = 252
    min_size: int = 20
    jump: int = 5
    every: int = 5
    confirmations: int = 2
    tolerance: int = 5
    max_age: int = 60
    cooldown: int = 20
    penalty_l2: float = 10.0
    penalty_rbf: float = 5.0


def validate_config(c):
    if c.min_size < 2 or c.window < 2*c.min_size:
        raise ValueError('Require min-size >= 2 and window >= 2*min-size')
    if min(c.jump, c.every, c.confirmations) < 1:
        raise ValueError('jump, every and confirmations must be positive')
    if min(c.tolerance, c.cooldown) < 0 or c.max_age < c.min_size:
        raise ValueError('Invalid boundary tolerance/cooldown/max-age')
    if not all(np.isfinite(x) and x > 0 for x in (c.penalty_l2, c.penalty_rbf)):
        raise ValueError('Penalties must be finite and positive')


def load_features(root, name, manifest, verify=True):
    relative = f'features/{name}_features.csv'
    path = root / relative
    if verify:
        # Phase 1 on Windows serializes relative paths with backslashes.
        canonical_hashes = {key.replace(chr(92), '/'): value
                            for key, value in manifest.get('sha256', {}).items()}
        expected = canonical_hashes.get(relative)
        if not expected:
            raise ValueError(f'No Phase 1 hash for {relative}; use the original completed snapshot')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Phase 1 integrity mismatch: {relative}')
    f = pd.read_csv(path, parse_dates=['Date', 'source_date']).set_index('Date')
    required = ['log_return', 'vol_20', 'source_date', 'split']
    if any(x not in f for x in required) or f.empty:
        raise ValueError(f'{name}: missing required Phase 1 columns or empty file')
    if f.index.has_duplicates or not f.index.is_monotonic_increasing:
        raise ValueError(f'{name}: dates must be unique and increasing')
    if f.index.hasnans:
        raise ValueError(f'{name}: missing Date')
    expected_source = pd.Series(f.index, index=f.index).shift(1)
    pd.testing.assert_series_equal(f.source_date, expected_source,
                                  check_names=False, check_freq=False)
    train_end = pd.Timestamp(manifest['train_end_inclusive'])
    val_end = pd.Timestamp(manifest['validation_end_inclusive'])
    if not train_end < val_end:
        raise ValueError('Invalid chronological boundaries in Phase 1 manifest')
    expected_split = np.where(f.index <= train_end, 'train',
                              np.where(f.index <= val_end, 'validation', 'test'))
    if not np.array_equal(f.split.to_numpy(), expected_split):
        raise ValueError(f'{name}: split labels disagree with Phase 1 boundaries')
    return f


def fit_training(f, model, train_end):
    """Fit scale, RBF bandwidth and descriptive regime cutoffs on training only."""
    columns = MODEL_FEATURES[model]
    train = f.loc[f.index <= pd.Timestamp(train_end), columns]
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 100:
        raise ValueError('At least 100 complete training rows required')
    mean, scale = train.mean(), train.std(ddof=0)
    scale = scale.mask(scale.abs() < 1e-12, 1.0)
    gamma = None
    if model == 'rbf':
        # Deterministic training-only subsample bounds pairwise-distance memory.
        z = ((train - mean)/scale).to_numpy()
        z = z[np.linspace(0, len(z)-1, min(500, len(z)), dtype=int)]
        distances = pdist(z, metric='sqeuclidean')
        positive = distances[distances > 0]
        gamma = float(1/np.median(positive)) if len(positive) else 1.0
    vol = f.loc[f.index <= pd.Timestamp(train_end), 'vol_20']
    vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
    low, high = vol.quantile([1/3, 2/3]).to_numpy()
    return {'columns': columns, 'mean': mean.to_dict(), 'scale': scale.to_dict(),
            'gamma': gamma, 'vol_low': float(low), 'vol_high': float(high),
            'fit_rows': len(train), 'fit_last_date': str(train.index.max().date())}


def breakpoints(x, model, params, config):
    cost = (rpt.costs.CostL2() if model == 'l2'
            else rpt.costs.CostRbf(gamma=params['gamma']))
    penalty = config.penalty_l2 if model == 'l2' else config.penalty_rbf
    # A fresh instance prevents cached Gram matrices from a different prefix.
    ends = rpt.Pelt(custom_cost=cost, min_size=config.min_size,
                    jump=config.jump).fit(x).predict(pen=penalty)
    return [int(b) for b in ends if b < len(x)]  # Terminal endpoint is NOT a break.


def walk_forward(f, model, params, config, train_end):
    """Never reads future rows when estimating a signal on the current date."""
    validate_config(config)
    columns = params['columns']
    x = (f[columns] - pd.Series(params['mean'])) / pd.Series(params['scale'])
    array = x.to_numpy(dtype=float)
    first = int(f.index.searchsorted(pd.Timestamp(train_end), side='right'))
    streak = 0
    pending, pending_hits = None, 0
    last_alert_break, last_alert_i = -10**9, -10**9
    state, state_vol = 'unknown', np.nan
    rows, events = [], []
    estimate, estimate_source, update_date = pd.NaT, pd.NaT, pd.NaT
    for i, date in enumerate(f.index):
        finite = np.isfinite(array[i]).all()
        streak = streak + 1 if finite else 0
        if i < first:
            continue
        checked, alert = False, False
        if not finite:
            state, state_vol = 'unknown', np.nan
            estimate, estimate_source, update_date = pd.NaT, pd.NaT, pd.NaT
            pending, pending_hits = None, 0
        if finite and (i-first) % config.every == 0 and streak >= 2*config.min_size:
            checked = True
            start = max(0, i+1-config.window, i+1-streak)
            ends = breakpoints(array[start:i+1], model, params, config)
            b = start + ends[-1] if ends else None
            segment_start = b if b is not None else start
            state_vol = float(f.vol_20.iloc[segment_start:i+1].mean())
            state = ('calm' if state_vol <= params['vol_low'] else
                     'stressed' if state_vol >= params['vol_high'] else 'moderate')
            update_date = date
            estimate = f.index[b] if b is not None else pd.NaT
            estimate_source = f.source_date.iloc[b] if b is not None else pd.NaT
            eligible = (b is not None and i-b <= config.max_age and
                        b > last_alert_break + config.tolerance)
            if eligible:
                if pending is not None and abs(b-pending) <= config.tolerance:
                    pending_hits += 1
                else:
                    pending, pending_hits = b, 1
                # Anchor the confirmation cluster at its first estimate (no drift).
                if pending_hits >= config.confirmations and i-last_alert_i >= config.cooldown:
                    alert = True
                    last_alert_break, last_alert_i = b, i
                    events.append({'decision_date': date, 'estimated_break_feature_date': f.index[b],
                        'estimated_break_source_date': f.source_date.iloc[b],
                        'lagged_feature_delay_sessions': i-b,
                        'regime_at_detection': state, 'segment_mean_vol_20': state_vol,
                        'confirmation_checks': pending_hits})
                    pending, pending_hits = None, 0
            else:
                pending, pending_hits = None, 0
        if checked or not finite:
            status = 'updated' if checked else 'missing_features'
        elif state == 'unknown':
            status = 'warming_up'
        else:
            status = 'carried_forward'
        rows.append({'Date': date, 'source_date': f.source_date.iloc[i],
            'split': f.split.iloc[i], 'checked': checked, 'status': status,
            'alert': alert, 'regime': state, 'segment_mean_vol_20': state_vol,
            'last_update_date': update_date, 'estimated_break_feature_date': estimate,
            'estimated_break_source_date': estimate_source})
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError('No post-training rows to evaluate')
    daily = daily.set_index('Date')
    event_columns = ['decision_date', 'estimated_break_feature_date',
        'estimated_break_source_date', 'lagged_feature_delay_sessions',
        'regime_at_detection', 'segment_mean_vol_20', 'confirmation_checks']
    return daily, pd.DataFrame(events, columns=event_columns)


def self_test():
    rng = np.random.default_rng(2026)
    n = 420
    dates = pd.bdate_range('2019-01-01', periods=n)
    vol = np.r_[rng.normal(.10,.004,220), rng.normal(.45,.004,n-220)]
    f = pd.DataFrame({'vol_20': vol, 'log_return': rng.normal(0,.01,n),
        'source_date': pd.Series(dates).shift().to_numpy(), 'split': 'validation'}, index=dates)
    train_end = dates[159]
    f.loc[:train_end,'split'] = 'train'
    c = Config(window=100, min_size=10, every=5, jump=1, confirmations=2,
               max_age=60, penalty_l2=10, penalty_rbf=3)
    for model in MODEL_FEATURES:
        params = fit_training(f, model, train_end)
        daily, events = walk_forward(f, model, params, c, train_end)
        assert not events.empty, f'{model} failed to detect the strong synthetic change'
        near = events.loc[events.estimated_break_feature_date.between(dates[210], dates[230])]
        assert not near.empty, f'{model}: no break near synthetic ground truth'
        assert (events.decision_date > events.estimated_break_source_date).all()
        assert (events.lagged_feature_delay_sessions >= c.min_size-1).all()
        prefix = f.iloc[:310]
        assert fit_training(prefix, model, train_end) == params
        pdaily, pevents = walk_forward(prefix, model, params, c, train_end)
        pd.testing.assert_frame_equal(pdaily, daily.loc[:prefix.index[-1]])
        pd.testing.assert_frame_equal(pevents.reset_index(drop=True),
            events.loc[events.decision_date <= prefix.index[-1]].reset_index(drop=True))
        changed = f.copy()
        changed.iloc[310:, changed.columns.get_loc('vol_20')] *= 100
        changed.iloc[310:, changed.columns.get_loc('log_return')] += 5
        assert fit_training(changed, model, train_end) == params
        cdaily, _ = walk_forward(changed, model, params, c, train_end)
        pd.testing.assert_frame_equal(daily.loc[:dates[309]], cdaily.loc[:dates[309]])
        missing = f.copy(); missing.loc[dates[300], ['vol_20','log_return']] = np.nan
        mdaily, _ = walk_forward(missing, model, params, c, train_end)
        assert mdaily.loc[dates[300], 'status'] == 'missing_features'
        assert mdaily.loc[dates[301], 'regime'] == 'unknown'
        assert breakpoints(np.zeros((100,len(params['columns']))), model, params, c) == []
        print(f'PASS {model}: synthetic break, prefix/future invariance, frozen calibration, gap reset, terminal removal')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('phase1_bank_v2'))
    p.add_argument('--output', type=Path, default=Path('phase2_output'))
    p.add_argument('--indices', nargs='+', default=['NIFTY50','BANK'])
    p.add_argument('--models', nargs='+', choices=list(MODEL_FEATURES), default=list(MODEL_FEATURES))
    p.add_argument('--include-test', action='store_true', help='Explicitly open test only after design is frozen')
    p.add_argument('--self-test', action='store_true')
    for field, default in asdict(Config()).items():
        p.add_argument('--'+field.replace('_','-'), type=type(default), default=default)
    a = p.parse_args(argv)
    if a.self_test:
        self_test(); return
    config = Config(**{k:getattr(a,k) for k in asdict(Config())})
    validate_config(config)
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use an empty output folder; prior decision logs must not be overwritten')
    manifest_path = a.input/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    train_end = manifest['train_end_inclusive']
    val_end = manifest['validation_end_inclusive']
    inputs, summary, hashes = {}, [], {}
    # Verify full inputs; test data are then excluded from all computations by default.
    for name in dict.fromkeys(a.indices):
        if name not in manifest['quality']:
            p.error(f'{name} is absent from Phase 1; available: {list(manifest["quality"])}')
        f = load_features(a.input, name, manifest)
        if not a.include_test:
            f = f.loc[:val_end]
        if not (f.index > pd.Timestamp(train_end)).any():
            raise ValueError(f'{name}: no evaluation rows')
        inputs[name] = f
        rel = f'features/{name}_features.csv'
        hashes[rel] = hashlib.sha256((a.input / rel).read_bytes()).hexdigest()
    a.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, f in inputs.items():
        for model in dict.fromkeys(a.models):
            print(f'{name} / {model}: sequential detection ({len(f)} feature rows)', flush=True)
            params = fit_training(f, model, train_end)
            daily, events = walk_forward(f, model, params, config, train_end)
            stem = a.output/f'{name}_{model}'
            daily.to_csv(str(stem)+'_decisions.csv')
            events.to_csv(str(stem)+'_events.csv', index=False)
            Path(str(stem)+'_calibration.json').write_text(json.dumps(params, indent=2))
            for split in ('validation','test'):
                part = daily.loc[daily.split.eq(split)]
                if part.empty:
                    continue
                summary.append({'index':name,'model':model,'split':split,'rows':len(part),
                    'checks':int(part.checked.sum()), 'alerts':int(part.alert.sum()),
                    'unknown_rows':int(part.regime.eq('unknown').sum()),
                    'calm_rows':int(part.regime.eq('calm').sum()),
                    'moderate_rows':int(part.regime.eq('moderate').sum()),
                    'stressed_rows':int(part.regime.eq('stressed').sum())})
            fig, axes = plt.subplots(2,1,figsize=(13,7),sharex=True)
            axes[0].plot(f.loc[daily.index,'vol_20'],label='Lagged annualized volatility',linewidth=.8)
            for _, event in events.iterrows():
                axes[0].axvline(event.decision_date,color='red',alpha=.45,linewidth=.8)
            axes[0].axhline(params['vol_low'],color='grey',linestyle='--')
            axes[0].axhline(params['vol_high'],color='grey',linestyle='--')
            levels = daily.regime.map({'calm':0,'moderate':1,'stressed':2,'unknown':np.nan})
            axes[1].step(daily.index,levels,where='post',linewidth=.8)
            axes[1].set_yticks([0,1,2],['calm','moderate','stressed'])
            axes[0].set_title(f'{name} / PELT-{model.upper()} — red lines: actual alert dates')
            axes[0].set_ylabel('Annualized volatility');axes[1].set_ylabel('As-of descriptive regime')
            for ax in axes: ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(str(stem)+'_timeline.png',dpi=140);plt.close(fig)
            print(f'  saved {len(daily)} decisions, {len(events)} alerts', flush=True)
    pd.DataFrame(summary).to_csv(a.output/'summary.csv', index=False)
    output_hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in a.output.iterdir() if f.is_file()}
    metadata = {'config':asdict(config), 'training_end':train_end, 'validation_end':val_end,
        'test_enabled':a.include_test, 'feature_sets':MODEL_FEATURES,
        'phase1_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'input_sha256':hashes, 'output_sha256':output_hashes,
        'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','ruptures','scipy','matplotlib']},
        'timing':'Decision date t uses Phase 1 lagged features through source date t-1. No retroactive alerts.',
        'interpretation':'Sequential reruns of offline PELT; not an intrinsically online CPD algorithm. '
                         'Initial parameters are illustrative, not validated market-performance optima.'}
    (a.output/'run_manifest.json').write_text(json.dumps(metadata,indent=2))
    print(f'Created {a.output.resolve()} | test {"included" if a.include_test else "excluded"}')


if __name__ == '__main__':
    main()
