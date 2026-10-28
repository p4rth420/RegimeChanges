#!/usr/bin/env python3
"""Train-only Gaussian HMM benchmark with causal filtering on validation data."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from hmmlearn.hmm import GaussianHMM

FEATURES = ['log_return', 'vol_20']
LABELS = ['calm', 'moderate', 'stressed']


def forward_filter(model, X, initial_log_alpha=None):
    """Return forward-only posteriors; never uses backward smoothing."""
    means = np.asarray(model.means_, dtype=float)
    covars = np.asarray(model.covars_, dtype=float)
    variances = np.diagonal(covars, axis1=1, axis2=2) if covars.ndim == 3 else covars
    variances = np.maximum(variances, 1e-8)
    log_start = np.log(np.maximum(model.startprob_, 1e-300))
    log_trans = np.log(np.maximum(model.transmat_, 1e-300))
    log_alpha = None if initial_log_alpha is None else np.asarray(initial_log_alpha, dtype=float).copy()
    out = np.empty((len(X), len(means)), dtype=float)
    for i, x in enumerate(X):
        delta = x[None, :] - means
        emission = -0.5 * np.sum(np.log(2.0 * np.pi * variances) + (delta * delta) / variances, axis=1)
        if log_alpha is None:
            candidate = log_start + emission
        else:
            candidate = logsumexp(log_alpha[:, None] + log_trans, axis=0) + emission
        log_alpha = candidate - logsumexp(candidate)
        out[i] = np.exp(log_alpha)
    return out, log_alpha


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Phase 1 output folder')
    parser.add_argument('--phase2', type=Path, required=True, help='Fixed 252/20 Phase 2 validation output')
    parser.add_argument('--output', type=Path, required=True, help='New empty output folder')
    parser.add_argument('--states', type=int, default=3)
    parser.add_argument('--restarts', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args(argv)
    if args.states != 3:
        parser.error('This comparison currently maps three HMM states to calm/moderate/stressed')
    if args.restarts < 1:
        parser.error('--restarts must be at least 1')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output folder; existing outputs are preserved')

    phase1_manifest_path = args.input / 'manifest.json'
    feature_path = args.input / 'features' / 'NIFTY50_features.csv'
    scaler_path = args.input / 'features' / 'NIFTY50_scaler.json'
    phase2_manifest_path = args.phase2 / 'run_manifest.json'
    for path in (phase1_manifest_path, feature_path, scaler_path, phase2_manifest_path):
        if not path.is_file():
            parser.error(f'Missing required input: {path}')
    phase1_manifest = json.loads(phase1_manifest_path.read_text(encoding='utf-8'))
    phase2_manifest = json.loads(phase2_manifest_path.read_text(encoding='utf-8'))
    if phase2_manifest.get('test_enabled') is not False:
        parser.error('Phase 2 baseline must have test_enabled=false')
    config = phase2_manifest.get('config', {})
    expected = {'window': 252, 'min_size': 20, 'penalty_l2': 10.0, 'penalty_rbf': 5.0}
    for key, value in expected.items():
        if config.get(key) != value:
            parser.error(f'Phase 2 baseline must use {key}={value}; found {config.get(key)}')
    if phase2_manifest.get('phase1_manifest_sha256') != sha256(phase1_manifest_path):
        parser.error('Phase 2 output does not match this Phase 1 manifest')
    input_hash = phase2_manifest.get('input_sha256', {}).get('features/NIFTY50_features.csv')
    if input_hash != sha256(feature_path):
        parser.error('Phase 2 baseline does not use this Phase 1 feature file')

    # Keep only complete training and validation rows. Test values are never used.
    frame = pd.read_csv(feature_path, parse_dates=['Date', 'source_date'])
    if frame['Date'].duplicated().any() or not frame['Date'].is_monotonic_increasing:
        parser.error('Feature dates must be unique and chronological')
    complete = frame['complete'].astype(str).str.lower().eq('true')
    frame = frame.loc[complete & frame['split'].isin(['train', 'validation'])].copy()
    train = frame.loc[frame['split'].eq('train')].copy()
    validation = frame.loc[frame['split'].eq('validation')].copy()
    if train.empty or validation.empty:
        parser.error('Need complete train and validation rows')
    if not np.isfinite(train[FEATURES].to_numpy(dtype=float)).all() or not np.isfinite(validation[FEATURES].to_numpy(dtype=float)).all():
        parser.error('Selected features contain non-finite values')

    scaler = json.loads(scaler_path.read_text(encoding='utf-8'))
    means = pd.Series(scaler['mean'], dtype=float).reindex(FEATURES)
    scales = pd.Series(scaler['scale'], dtype=float).reindex(FEATURES)
    if means.isna().any() or scales.isna().any() or (scales <= 0).any():
        parser.error('Invalid Phase 1 training-only scaler')
    X_train = ((train[FEATURES].astype(float) - means) / scales).to_numpy()
    X_val = ((validation[FEATURES].astype(float) - means) / scales).to_numpy()

    candidates = []
    best_model = None
    best_loglik = -np.inf
    for i in range(args.restarts):
        seed = args.seed + i
        model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=500,
                            tol=1e-4, min_covar=1e-3, random_state=seed,
                            implementation='log')
        model.fit(X_train)
        train_loglik = float(model.score(X_train))
        converged = bool(model.monitor_.converged)
        candidates.append({'seed': seed, 'train_log_likelihood': train_loglik,
                           'converged': converged, 'iterations': int(model.monitor_.iter)})
        if train_loglik > best_loglik:
            best_model, best_loglik = model, train_loglik
    if best_model is None:
        raise RuntimeError('No HMM restart completed')

    train_prob, last_train_log_alpha = forward_filter(best_model, X_train)
    val_prob_raw, _ = forward_filter(best_model, X_val, last_train_log_alpha)
    raw_vol = train['vol_20'].to_numpy(dtype=float)
    state_mean_vol = np.array([
        np.average(raw_vol, weights=np.maximum(train_prob[:, state], 1e-12))
        for state in range(args.states)
    ])
    state_order = np.argsort(state_mean_vol)
    ordered_prob = val_prob_raw[:, state_order]
    regime = np.array(LABELS)[np.argmax(ordered_prob, axis=1)]
    val_out = validation[['Date', 'source_date', 'split']].reset_index(drop=True).copy()
    for j, label in enumerate(LABELS):
        val_out[f'prob_{label}'] = ordered_prob[:, j]
    val_out['hmm_regime'] = regime
    val_out['max_state_probability'] = ordered_prob.max(axis=1)
    prior_rank = int(np.flatnonzero(state_order == np.argmax(train_prob[-1]))[0])
    val_out['is_state_change'] = np.r_[regime[0] != LABELS[prior_rank], regime[1:] != regime[:-1]]

    # Compare with the as-of PELT descriptive labels on the exact validation dates.
    comparisons = []
    transition_rows = []
    prior_regime = LABELS[prior_rank]
    for row in val_out.itertuples(index=False):
        if row.hmm_regime != prior_regime:
            transition_rows.append({'Date': row.Date, 'source_date': row.source_date,
                                    'from_regime': prior_regime, 'to_regime': row.hmm_regime,
                                    'new_state_probability': row.max_state_probability})
        prior_regime = row.hmm_regime
    transitions = pd.DataFrame(transition_rows, columns=['Date','source_date','from_regime','to_regime','new_state_probability'])
    session_pos = {d: i for i, d in enumerate(frame['Date'])}
    transition_dates = pd.to_datetime(transitions['Date']) if len(transitions) else pd.Series(dtype='datetime64[ns]')
    metrics = []
    for model_name in ('l2', 'rbf'):
        decisions_path = args.phase2 / f'NIFTY50_{model_name}_decisions.csv'
        events_path = args.phase2 / f'NIFTY50_{model_name}_events.csv'
        decisions = pd.read_csv(decisions_path, parse_dates=['Date'])
        joined = val_out[['Date','hmm_regime']].merge(decisions[['Date','regime']], on='Date', how='inner', validate='one_to_one')
        agreement = float((joined['hmm_regime'] == joined['regime']).mean())
        kappa = float(cohen_kappa_score(joined['regime'], joined['hmm_regime'], labels=LABELS))
        ari = float(adjusted_rand_score(joined['regime'], joined['hmm_regime']))
        pelt_events = pd.read_csv(events_path, parse_dates=['estimated_break_feature_date'])
        matched = 0
        gaps = []
        for d in pelt_events['estimated_break_feature_date']:
            if len(transition_dates):
                gap = min(abs(session_pos.get(pd.Timestamp(d), -10**9) - session_pos.get(pd.Timestamp(x), 10**9)) for x in transition_dates)
                gaps.append(gap)
                matched += gap <= 5
        metrics.append({'comparison': f'HMM vs PELT-{model_name.upper()}',
                        'validation_dates': len(joined),
                        'label_agreement': agreement,
                        'cohen_kappa': kappa,
                        'adjusted_rand_index': ari,
                        'pelt_confirmed_alerts': len(pelt_events),
                        'pelt_estimated_breaks_with_hmm_transition_within_5_sessions': f'{matched}/{len(pelt_events)}',
                        'median_nearest_hmm_transition_gap_sessions': float(np.median(gaps)) if gaps else None})

    args.output.mkdir(parents=True, exist_ok=True)
    val_out.to_csv(args.output / 'NIFTY50_hmm_validation_probabilities.csv', index=False)
    transitions.to_csv(args.output / 'NIFTY50_hmm_validation_transitions.csv', index=False)
    pd.DataFrame(candidates).to_csv(args.output / 'hmm_training_restarts.csv', index=False)
    pd.DataFrame(metrics).to_csv(args.output / 'hmm_vs_pelt_validation.csv', index=False)
    state_info = []
    for rank, raw_state in enumerate(state_order):
        state_info.append({'regime_label': LABELS[rank], 'raw_hmm_state': int(raw_state),
                           'training_probability_weighted_mean_vol_20': float(state_mean_vol[raw_state]),
                           'state_mean_scaled_log_return': float(best_model.means_[raw_state, 0]),
                           'state_mean_scaled_vol_20': float(best_model.means_[raw_state, 1])})
    metadata = {
        'model': 'GaussianHMM', 'states': 3, 'covariance_type': 'diag',
        'features': FEATURES, 'feature_scaler': 'Phase 1 training-only scaler',
        'fit_period': {'first': str(train['Date'].min().date()), 'last': str(train['Date'].max().date()), 'rows': len(train)},
        'validation_period': {'first': str(validation['Date'].min().date()), 'last': str(validation['Date'].max().date()), 'rows': len(validation)},
        'test_used': False, 'filtering': 'forward-only recursive probabilities; no backward smoothing',
        'restart_selection': 'maximum training log likelihood only',
        'best_training_log_likelihood': best_loglik,
        'best_restart_seed': int(candidates[int(np.argmax([x['train_log_likelihood'] for x in candidates]))]['seed']),
        'best_restart_converged': bool(best_model.monitor_.converged),
        'states_ordered_by_training_probability_weighted_mean_vol_20': state_info,
        'validation_comparison': metrics,
        'limitations': 'HMM state transitions and PELT alerts are not ground-truth changes; label agreement measures method agreement only.'
    }
    (args.output / 'hmm_run_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Created {args.output.resolve()} | HMM fit on training only | forward-filtered validation | test excluded')
    for item in metrics:
        print(item)


if __name__ == '__main__':
    main()
