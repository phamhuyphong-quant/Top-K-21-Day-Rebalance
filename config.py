final_features= [
    'dist_52w_high',
    'log_ret_skip1m',
    'WQ_Alpha_024',
    'log_ret_6m',
    'RSI_14',
    'log_ret_3m',
    'price_vol_divergence',
    'dist_SMA_50',
    'dist_SMA_21',
    'log_ret_1w',
    'log_ret_1m',
    'WQ_Alpha_028',
    'volume_surge_weekly',
    'log_ret_1y',
    'volatility_shock_monthly',
    'volume_surge_monthly',
    'volatility_shock_weekly',
    'volatility_1w',
    'volatility_1m',
    'volatility_6m'
]


candidate_features = [
    # --- Return features ---
    "log_ret_1w",
    "log_ret_1m",
    "log_ret_3m",
    "log_ret_6m",
    "log_ret_1y",

    # --- Volatility ---
    "volatility_1w",
    "volatility_1m",
    "volatility_3m",
    "volatility_6m",
    "volatility_shock_monthly",
    "volatility_shock_weekly",

    # --- Moving average distances ---
    "dist_SMA_9", "dist_SMA_21", "dist_SMA_50", "dist_SMA_100", "dist_SMA_200",
    "dist_EMA_9", "dist_EMA_21", "dist_EMA_50", "dist_EMA_100", "dist_EMA_200",

    # --- Volume / price-volume ---
    "volume_surge_monthly",
    "volume_surge_weekly",
    "obv_trend",
    "price_vol_divergence",

    # --- Momentum / technical ---
    "RSI_14",
    "dist_52w_high",
    "log_ret_skip1m",

    # --- WQ Alpha signals (return-predicting, eligible for deep combiner) ---
    "WQ_Alpha_001",   # 12-1m momentum
    "WQ_Alpha_002",   # Volume momentum vs intraday return rank correlation
    "WQ_Alpha_006",   # -corr(open, volume, 10)
    "WQ_Alpha_007",   # Conditional momentum. Only fires on high-volume days.
    "WQ_Alpha_013",   # Close-volume co-movement. Stocks where price and volume co-move get shorted.
    "WQ_Alpha_016",   # Captures breakout/blowoff tops (same as 013 but uses 'high' instead of 'close').
    "WQ_Alpha_024",   # conditional 100d mean-reversion
    "WQ_Alpha_028",   # corr(adv20, low, 5) + midprice - close
    "WQ_Alpha_040",   # Penalizes high-volatility stocks that also have high-volume correlation (avoids blow-off tops).
    "WQ_Alpha_101",   # BAB: negative rolling beta to equal-weighted market
    "WQ_Alpha_103",   # monthly price acceleration (21d second derivative)
    "WQ_Alpha_200",   # Value proxy: distance from 52w high (far below = cheap)
    "WQ_Alpha_201",   # Volume trend confirmation: price momentum × volume momentum
    "WQ_Alpha_202",   # Residual momentum: 12-1m excess return vs equal-weighted market

    # --- Structural / context features (XGBoost only, NOT deep combiner) ---
    # monthly-compatible but not cross-sectional return signals
    #"turnover_12m",       # 252d/504d volume ratio — structural liquidity level
    #"limit_bias_60d",     # up-limit minus down-limit days (60d) — demand/supply pressure
    #"herding_dispersion", # cross-sectional return std — market regime indicator
    #"amihud_illiquidity", # log1p(|ret|/volume, 21d) — illiquidity premium proxy
]
BASE_MODEL_PARAMS= {
    'device':             'cuda',
    'tree_method':        'hist',
    'objective':          'rank:ndcg',
    'random_state':       42,
    # Tree structure
    'n_estimators':       100,
    'max_depth':          4,
    'min_child_weight':   5,
    # Learning
    'learning_rate':      0.05,
    # Sampling
    'subsample':          0.7,
    'colsample_bytree':   0.6,
    # Regularization
    'reg_lambda':         2.0,
    'reg_alpha':          0.5,
    # Ranking
    'lambdarank_pair_method':         'topk',
    'lambdarank_num_pair_per_sample':  60,
}


FEATURE_GROUPS: dict[str, list[str]] = {
    "returns": [
        "log_ret_1w",
        "log_ret_1m",
        "log_ret_3m",
        "log_ret_6m",
        "log_ret_1y",
    ],
    "volatility": [
        "volatility_1w",
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_shock_monthly",
        "volatility_shock_weekly",
    ],
    "moving_average": [
        "dist_SMA_9", "dist_SMA_21", "dist_SMA_50","dist_SMA_100","dist_SMA_200",
        "dist_EMA_9", "dist_EMA_21", "dist_EMA_50","dist_EMA_100","dist_EMA_200",
    ],
    "volume": [
        "volume_surge_monthly",
        "volume_surge_weekly",
        "obv_trend",
        "price_vol_divergence",
    ],
    "rsi": [
        "RSI_14",
    ],
    "wq_features": [
        "WQ_Alpha_001",
        "WQ_Alpha_002",
        "WQ_Alpha_006",
        "WQ_Alpha_007",
        "WQ_Alpha_013",
        "WQ_Alpha_016",
        "WQ_Alpha_024",
        "WQ_Alpha_028",
        "WQ_Alpha_040",
        "WQ_Alpha_101",
        "WQ_Alpha_103",
        "WQ_Alpha_200",
        "WQ_Alpha_201",
        "WQ_Alpha_202"
    ],
    
    "price_structure": [
    "dist_52w_high",
    "log_ret_skip1m",]

}

ALL_GROUP_NAMES: list[str] = list(FEATURE_GROUPS.keys())


# ---------------------------------------------------------------------------
# Helper: flatten a list of group names → feature column list
# ---------------------------------------------------------------------------
def groups_to_features(group_names: list[str]) -> list[str]:
    features: list[str] = []
    for g in group_names:
        features.extend(FEATURE_GROUPS[g])
    return features
