final_features= ['log_ret_6m',
 'volatility_1w',
 'volatility_1m',
 'volatility_shock_monthly',
 'volatility_shock_weekly',
 'dist_EMA_9',
 'volume_surge_monthly',
 'volume_surge_weekly',
 'WQ_Alpha_012',
 'WQ_Alpha_024',
 'WQ_Alpha_053',
 'dist_EMA_100']


candidate_features = [
    "log_ret_1w",
        "log_ret_1m",
        "log_ret_3m",
        "log_ret_6m",
        "log_ret_1y",

        "volatility_1w",
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_shock_monthly",
        "volatility_shock_weekly",

        "dist_SMA_9", "dist_SMA_21", "dist_SMA_50","dist_SMA_100","dist_SMA_200",
        "dist_EMA_9", "dist_EMA_21", "dist_EMA_50","dist_EMA_100","dist_EMA_200",

        "volume_surge_monthly",
        "volume_surge_weekly",
        "obv_trend",
        "price_vol_divergence",

        "RSI_14",

        "WQ_Alpha_012","WQ_Alpha_024","WQ_Alpha_028","WQ_Alpha_053","WQ_Alpha_060",

        "dist_52w_high",
        "log_ret_skip1m",
]
