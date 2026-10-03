"""
Significance tests for the Top-K ranker comparison.

Families of tests
  A. base model vs every other model     -> compare_base_vs_models   (delta = base - other, >0 => base better)
  B. every model vs its own universe     -> compare_models_vs_universe (delta = top-k - universe, >0 => beats universe)
  C. Sharpe difference of two equity curves -> sharpe_diff_block_bootstrap

Entry point: run_analysis(dfs, base_model, mode='base' | 'universe' | 'both')

Why blocks: target_magnitude is a 21-day forward return / trailing vol, so targets of
adjacent days share 20 of 21 days and daily deltas are strongly autocorrelated. All tests
resample BLOCKS of consecutive days (default 21). `step=21` is a stricter cross-check that
keeps only every 21st date (non-overlapping samples, less power). With `step`, `block`
is counted in rebalance periods (default 1) and `offset` (0..step-1) picks the start position.
Use plot_autocorr() to choose the block size, offset_sensitivity() to check step results.
Delta units: vol-adjusted return units, not % return.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHUNK = 1000  # resamples per chunk, keeps memory small


# ================================================================ internal helpers
def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def _block_signs(rng, n, block, size):
    """Random +-1 signs, one per block of consecutive days, shape (size, n)."""
    n_blocks = int(np.ceil(n / block))
    s = rng.integers(0, 2, size=(size, n_blocks)) * 2 - 1
    return np.repeat(s, block, axis=1)[:, :n]


def _block_idx(rng, n, block, size):
    """Moving-block bootstrap indices, shape (size, n)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(size, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(size, -1)[:, :n]


def _sharpe_rows(a, annualization_factor, rf_annual):
    a = a - rf_annual / annualization_factor
    sd = a.std(axis=1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sd == 0, np.nan, a.mean(axis=1) / sd * np.sqrt(annualization_factor))


# ================================================================ 1. daily series builders
def topk_daily_avg(df, k=20, date_col='date', score_col='pred_score', target_col='target_magnitude'):
    """Mean target of the top-k scored rows per day (vectorized; ties broken by row order)."""
    d = df[[date_col, score_col, target_col]].dropna()
    rank = d.groupby(date_col)[score_col].rank(method='first', ascending=False)
    return d[rank <= k].groupby(date_col)[target_col].mean().rename('topk_avg')


def universe_daily_avg(df, date_col='date', target_col='target_magnitude'):
    """Mean target over ALL rows per day."""
    return df.dropna(subset=[target_col]).groupby(date_col)[target_col].mean().rename('universe_avg')


def _pair(a, b):
    m = pd.concat([a.rename('avg_1'), b.rename('avg_2')], axis=1, join='inner').sort_index()
    m['delta'] = m['avg_2'] - m['avg_1']
    return m


def compute_delta_base_vs_model(df_base, df_other, k=20, date_col='date', score_col='pred_score',
                                target_col='target_magnitude'):
    """delta = base top-k avg - other top-k avg. Positive => base is better."""
    return _pair(topk_daily_avg(df_other, k, date_col, score_col, target_col),
                 topk_daily_avg(df_base, k, date_col, score_col, target_col))


def compute_delta_vs_universe(df, k=20, date_col='date', score_col='pred_score', target_col='target_magnitude'):
    """delta = top-k avg - universe avg. Positive => model beats the universe."""
    return _pair(universe_daily_avg(df, date_col, target_col),
                 topk_daily_avg(df, k, date_col, score_col, target_col))


# ================================================================ 2. core tests
def paired_permutation_test(delta, n_iter=10000, alternative='two-sided', block=1, seed=None):
    """Block sign-flip permutation test on per-day deltas (block=1 -> classic test)."""
    rng = np.random.default_rng(seed)
    d = _clean(delta)
    n = len(d)
    obs = d.mean()
    null = np.concatenate([
        (_block_signs(rng, n, block, min(CHUNK, n_iter - i)) * d).mean(axis=1)
        for i in range(0, n_iter, CHUNK)])
    eps = 1e-12
    if alternative == 'two-sided':
        hits = np.sum(np.abs(null) >= abs(obs) - eps)
    elif alternative == 'greater':
        hits = np.sum(null >= obs - eps)
    elif alternative == 'less':
        hits = np.sum(null <= obs + eps)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {'observed_mean_delta': obs, 'n_days': n, 'p_value': (hits + 1) / (n_iter + 1),
            'null_distribution': null}


def paired_bootstrap_ci(delta, n_boot=10000, ci=0.95, block=1, seed=None):
    """Moving-block bootstrap CI for the mean delta (block=1 -> ordinary bootstrap)."""
    rng = np.random.default_rng(seed)
    d = _clean(delta)
    n = len(d)
    boots = np.concatenate([
        d[_block_idx(rng, n, block, min(CHUNK, n_boot - i))].mean(axis=1)
        for i in range(0, n_boot, CHUNK)])
    a = 1 - ci
    lo, hi = np.percentile(boots, [100 * a / 2, 100 * (1 - a / 2)])
    return {'observed_mean_delta': d.mean(), 'n_days': n, 'ci_lo': lo, 'ci_hi': hi,
            'boot_distribution': boots}


def run_comparison(name, delta, block=None, step=None, offset=0, n_iter=10000, n_boot=10000,
                   alternative='two-sided', seed=42):
    """
    Permutation p-value + bootstrap CI for one delta series.
    step   : if set (e.g. 21), keep every step-th date starting at `offset` (0..step-1).
    block  : block length. Default None -> 21 days (no step) or 1 rebalance period (with step).
             With step, block counts rebalance periods (e.g. step=21, block=3).
    p_value: at the chosen block size (the one to report)
    p_naive: block=1, i.e. ignoring any dependence (shown for comparison)
    """
    d = pd.Series(delta).dropna()
    if step:
        d = d.iloc[offset::step]
    if block is None:
        block = 1 if step else 21
    v = d.values
    if block > len(v):
        raise ValueError(f'block={block} is longer than the series ({len(v)} points)')
    perm = paired_permutation_test(v, n_iter, alternative, block, seed)
    boot = paired_bootstrap_ci(v, n_boot, 0.95, block, seed)
    p_naive = perm['p_value'] if block == 1 else paired_permutation_test(v, n_iter, alternative, 1, seed)['p_value']
    return {'comparison': name, 'mean_delta': perm['observed_mean_delta'], 'n_days': perm['n_days'],
            'block': block, 'p_value': perm['p_value'], 'p_naive': p_naive,
            'ci_lo': boot['ci_lo'], 'ci_hi': boot['ci_hi']}


def add_holm_correction(results, alpha=0.05):
    """Holm-adjusted p-values within ONE family of tests (call once per table)."""
    res = results.copy()
    p = res['p_value'].values
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    res['p_holm'] = adj
    res['significant'] = res['p_value'] < alpha
    res['significant_holm'] = res['p_holm'] < alpha
    return res


# ================================================================ 3. comparison functions
def compare_base_vs_models(dfs, base_model='ndcg', k=20, block=None, step=None, offset=0,
                           date_col='date', score_col='pred_score', target_col='target_magnitude', **kw):
    """Base vs each other model. delta = base - other, so positive => base better."""
    rows = []
    for name, df in dfs.items():
        if name == base_model:
            continue
        m = compute_delta_base_vs_model(dfs[base_model], df, k, date_col, score_col, target_col)
        rows.append(run_comparison(f'{base_model} vs {name}', m['delta'], block=block, step=step,
                                   offset=offset, **kw))
    return add_holm_correction(pd.DataFrame(rows))


def compare_models_vs_universe(dfs, k=20, block=None, step=None, offset=0, date_col='date',
                               score_col='pred_score', target_col='target_magnitude', **kw):
    """Every model's top-k vs its own universe. delta = top-k - universe."""
    rows = [run_comparison(f'{name} vs universe',
                           compute_delta_vs_universe(df, k, date_col, score_col, target_col)['delta'],
                           block=block, step=step, offset=offset, **kw)
            for name, df in dfs.items()]
    return add_holm_correction(pd.DataFrame(rows))


# ================================================================ 4. diagnostics
def autocorr_report(delta, max_lag=25):
    """Autocorrelation of a delta series by lag, plus a rough effective sample size."""
    s = pd.Series(delta).dropna()
    ac = pd.Series({lag: s.autocorr(lag) for lag in range(1, max_lag + 1)}, name='autocorr')
    n = len(s)
    n_eff = n / max(1.0, 1 + 2 * ac.clip(lower=0).sum())
    print(f'n = {n}, rough effective n ~ {n_eff:.0f}')
    print(ac.round(3).to_string())
    return ac


def outlier_report(df, date_col='date', target_col='target_magnitude', top_frac=0.01):
    """How much of the target's variance comes from the most extreme stock-days."""
    x = df[target_col].dropna()
    dev2 = ((x - x.mean()) ** 2).sort_values(ascending=False)
    n_top = max(1, int(len(dev2) * top_frac))
    out = {'n_obs': len(x), 'mean': x.mean(), 'std': x.std(),
           'min': x.min(), 'p1': x.quantile(0.01), 'p99': x.quantile(0.99), 'max': x.max(),
           f'var_share_top_{top_frac:.0%}': dev2.iloc[:n_top].sum() / dev2.sum()}
    for k_, v in out.items():
        print(f'{k_:>20}: {v:.4f}' if isinstance(v, float) else f'{k_:>20}: {v}')
    return out


def _collect_deltas(dfs, base_model, mode, k, date_col, score_col, target_col):
    """{comparison name: daily delta series} for the chosen mode."""
    out = {}
    if mode in ('base', 'both'):
        for name, df in dfs.items():
            if name != base_model:
                out[f'{base_model} vs {name}'] = compute_delta_base_vs_model(
                    dfs[base_model], df, k, date_col, score_col, target_col)['delta']
    if mode in ('universe', 'both'):
        for name, df in dfs.items():
            out[f'{name} vs universe'] = compute_delta_vs_universe(
                df, k, date_col, score_col, target_col)['delta']
    return out


def _acf(series, max_lag):
    s = pd.Series(series).dropna()
    max_lag = max(1, min(max_lag, len(s) // 2))
    return pd.Series({lag: s.autocorr(lag) for lag in range(1, max_lag + 1)}), len(s)


def suggest_block(ac, n, min_block=1, quiet_run=3):
    """
    First lag where `quiet_run` consecutive autocorrelations are inside the +-1.96/sqrt(n) band.
    Dependence lasts up to the lag before it, so a block of that length covers it.
    Returns (suggested_block or None if never quiet within max_lag, band).
    """
    band = 1.96 / np.sqrt(n)
    inside = (ac.abs() < band).values
    lags = ac.index.values
    for i in range(len(lags) - quiet_run + 1):
        if inside[i:i + quiet_run].all():
            return max(min_block, int(lags[i])), band
    return None, band


def plot_autocorr(dfs, base_model='ndcg', mode='both', k=20, max_lag=60, step=None, offset=0,
                  date_col='date', score_col='pred_score', target_col='target_magnitude',
                  save_path=None, show=True):
    """
    Autocorrelation bars of every delta series (one panel per comparison) with the +-1.96/sqrt(n)
    noise band, plus a suggested block size. Blue bars are outside the band.
    Daily mode: lags are days, red dashed line = 21-day target horizon, block never suggested < 21.
    step mode : lags are rebalance periods (thinned series, starting at `offset`).
    """
    deltas = _collect_deltas(dfs, base_model, mode, k, date_col, score_col, target_col)
    min_block = 1 if step else 21
    n_pan = len(deltas)
    ncols = min(3, n_pan)
    nrows = int(np.ceil(n_pan / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.2 * nrows), squeeze=False, sharey=True)

    rows = []
    for ax, (name, d) in zip(axes.ravel(), deltas.items()):
        s = d.iloc[offset::step] if step else d
        ac, n = _acf(s, max_lag)
        sug, band = suggest_block(ac, n, min_block)
        ax.bar(ac.index, ac.values, width=0.8,
               color=['tab:blue' if abs(v) > band else 'lightgray' for v in ac.values])
        ax.axhspan(-band, band, color='tab:orange', alpha=0.15)
        ax.axhline(0, color='black', linewidth=0.8)
        if not step and ac.index.max() >= 21:
            ax.axvline(21, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'{name} (n={n})', fontsize=9)
        ax.set_xlabel('Lag (rebalance periods)' if step else 'Lag (days)')
        rows.append({'comparison': name, 'n': n, 'band': round(band, 3),
                     'suggested_block': sug if sug is not None else f'> {int(ac.index.max())}'})
    for ax in axes.ravel()[n_pan:]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel('Autocorrelation')
    fig.suptitle('Autocorrelation of daily top-k deltas' if not step
                 else f'Autocorrelation of deltas, every {step}th date (offset {offset})', fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f'(orange band = noise range; with few points it is wide, so "inside the band" means '
          f'"not detectable", not "absent". Use the largest suggested block, or a bit more.)')
    return table


def offset_sensitivity(dfs, base_model='ndcg', family='universe', step=21, block=None, k=20,
                       offsets=None, n_iter=2000, n_boot=2000, date_col='date',
                       score_col='pred_score', target_col='target_magnitude'):
    """
    Re-run the step-subsampled tests from every starting offset (0..step-1) and show how stable
    the p-values are. family = 'universe' or 'base'. Returns (p-value table, summary).
    Raw p-values only (no Holm); n_iter/n_boot are lower by default to keep it fast.
    """
    offsets = range(step) if offsets is None else offsets
    cols = {}
    for o in offsets:
        kw = dict(k=k, block=block, step=step, offset=o, date_col=date_col, score_col=score_col,
                  target_col=target_col, n_iter=n_iter, n_boot=n_boot)
        t = (compare_models_vs_universe(dfs, **kw) if family == 'universe'
             else compare_base_vs_models(dfs, base_model, **kw))
        cols[o] = t.set_index('comparison')['p_value']
    P = pd.DataFrame(cols)
    summary = pd.DataFrame({'p_min': P.min(axis=1), 'p_median': P.median(axis=1), 'p_max': P.max(axis=1),
                            'share_p<0.05': (P < 0.05).mean(axis=1)})
    print(summary.round(4).to_string())
    return P, summary


# ================================================================ 5. output
def print_table(results):
    t = results.copy()
    for c in ['mean_delta', 'p_value', 'p_naive', 'p_holm', 'ci_lo', 'ci_hi']:
        if c in t:
            t[c] = t[c].round(4)
    print(t.to_string(index=False))
    return t


def _forest(ax, results, title, sig_col='significant_holm'):
    df = results.sort_values('mean_delta').reset_index(drop=True)
    labels = df['comparison']
    sig = df[sig_col] if sig_col in df else df['p_value'] < 0.05
    colors = ['tab:blue' if s else 'lightgray' for s in sig]
    y = np.arange(len(df))

    ax.hlines(y, df['ci_lo'], df['ci_hi'], color=colors, linewidth=2)
    ax.scatter(df['mean_delta'], y, color=colors, zorder=3, s=60)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean delta (target_magnitude, volatility-adjusted units)')
    ax.set_title(title, fontsize=10)

    for i, row in df.iterrows():
        txt = f"p={row['p_value']:.3f}"
        if 'p_holm' in df:
            txt += f" (Holm {row['p_holm']:.3f})"
        if sig.iloc[i]:
            txt += ' *'
        ax.annotate(txt, (row['ci_hi'], i), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9)

    lo, hi = min(df['ci_lo'].min(), 0), max(df['ci_hi'].max(), 0)
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.05 * span, hi + 0.5 * span)  # room for the p-value labels


def _fig_height(*tables):
    return 0.7 * max(len(t) for t in tables) + 1.8


def plot_base_vs_models(results, save_path=None, show=True,
                        title='Base model vs other models (95% block-bootstrap CI)'):
    fig, ax = plt.subplots(figsize=(9, _fig_height(results)))
    _forest(ax, results, title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_vs_universe(results, save_path=None, show=True,
                     title='Each model top-k vs universe (95% block-bootstrap CI)'):
    fig, ax = plt.subplots(figsize=(9, _fig_height(results)))
    _forest(ax, results, title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def run_analysis(dfs, base_model='ndcg', mode='both', k=20, block=None, step=None, offset=0,
                 date_col='date', score_col='pred_score', target_col='target_magnitude',
                 save_dir=None, show=True, **kw):
    """
    mode = 'base'     : base vs other models (table + plot)
           'universe' : each model vs universe (table + plot)
           'both'     : both tables, two panels in one figure
    save_dir : if given, plots are saved there as PNG.
    block=None -> 21 days (no step) or 1 rebalance period (with step); offset = start position for step.
    Returns dict of result tables.
    """
    if mode not in ('base', 'universe', 'both'):
        raise ValueError("mode must be 'base', 'universe' or 'both'")
    common = dict(k=k, block=block, step=step, offset=offset, date_col=date_col, score_col=score_col,
                  target_col=target_col, **kw)
    out = {}
    path = (lambda f: f'{save_dir}/{f}') if save_dir else (lambda f: None)

    if mode in ('base', 'both'):
        out['base_vs_models'] = compare_base_vs_models(dfs, base_model, **common)
        print(f'\n=== {base_model} vs other models (delta = {base_model} - other; >0 => {base_model} better) ===')
        print_table(out['base_vs_models'])
    if mode in ('universe', 'both'):
        out['vs_universe'] = compare_models_vs_universe(dfs, **common)
        print('\n=== each model top-k vs universe (delta = top-k - universe; >0 => beats universe) ===')
        print_table(out['vs_universe'])

    if mode == 'base':
        plot_base_vs_models(out['base_vs_models'], path('base_vs_models.png'), show)
    elif mode == 'universe':
        plot_vs_universe(out['vs_universe'], path('vs_universe.png'), show)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(17, _fig_height(out['base_vs_models'], out['vs_universe'])))
        _forest(axes[0], out['base_vs_models'], f'{base_model} vs other models')
        _forest(axes[1], out['vs_universe'], 'Top-k vs universe')
        plt.tight_layout()
        if save_dir:
            fig.savefig(path('combined.png'), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    return out


# ================================================================ 6. Sharpe difference
def sharpe_ratio(returns, annualization_factor=252, rf_annual=0.0):
    """Annualized Sharpe from 1D daily returns (rf_annual: annual risk-free rate, e.g. 0.04)."""
    r = _clean(returns) - rf_annual / annualization_factor
    sd = r.std(ddof=1)
    return np.nan if sd == 0 else r.mean() / sd * np.sqrt(annualization_factor)


def sharpe_diff_block_bootstrap(history_1, history_2, block=21, n_boot=10000, annualization_factor=252,
                                rf_annual=0.0, date_col='date', value_col='total_value', seed=42):
    """
    Paired moving-block bootstrap for Sharpe(history_1) - Sharpe(history_2).
    Same block positions for both series (pairing preserved). Two-sided p-value from the
    bootstrap distribution re-centred at 0, with +1 correction.
    """
    rng = np.random.default_rng(seed)
    r1 = history_1.set_index(date_col)[value_col].pct_change().dropna().rename('r1')
    r2 = history_2.set_index(date_col)[value_col].pct_change().dropna().rename('r2')
    m = pd.concat([r1, r2], axis=1, join='inner').dropna().sort_index()
    n = len(m)
    a1, a2 = m['r1'].values, m['r2'].values

    s1 = sharpe_ratio(a1, annualization_factor, rf_annual)
    s2 = sharpe_ratio(a2, annualization_factor, rf_annual)
    obs = s1 - s2

    diffs = []
    for i in range(0, n_boot, CHUNK):
        idx = _block_idx(rng, n, block, min(CHUNK, n_boot - i))
        diffs.append(_sharpe_rows(a1[idx], annualization_factor, rf_annual)
                     - _sharpe_rows(a2[idx], annualization_factor, rf_annual))
    boot = np.concatenate(diffs)
    boot = boot[~np.isnan(boot)]

    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p = (np.sum(np.abs(boot - boot.mean()) >= abs(obs)) + 1) / (len(boot) + 1)
    return {'sharpe_1': s1, 'sharpe_2': s2, 'observed_diff': obs, 'p_value': p,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n_days': n, 'boot_distribution': boot}


# ---------------- Example usage ----------------
# dfs = {'ndcg': df_ndcg, 'mse': df_mse, 'lstm': df_lstm, 'linear': df_linear}
# res = run_analysis(dfs, base_model='ndcg', mode='both', k=20, block=21, save_dir='.')
# run_analysis(dfs, 'ndcg', mode='base')                 # only base vs models
# run_analysis(dfs, 'ndcg', mode='universe', block=42)   # only vs universe, sensitivity
# run_analysis(dfs, 'ndcg', mode='both', step=21)        # independent-sample cross-check
# plot_autocorr(dfs, 'ndcg', mode='both')                       # choose the block size (daily)
# plot_autocorr(dfs, 'ndcg', mode='universe', step=21, offset=0) # leftover dependence after thinning
# run_analysis(dfs, 'ndcg', mode='both', step=21, block=3)       # 3 consecutive rebalance periods
# P, summ = offset_sensitivity(dfs, 'ndcg', family='universe', step=21)
# autocorr_report(compute_delta_vs_universe(df_ndcg)['delta']); outlier_report(df_ndcg)
# out = sharpe_diff_block_bootstrap(history_p1, history_p4, block=21)