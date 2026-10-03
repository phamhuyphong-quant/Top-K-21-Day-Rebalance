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
    #"dist_52w_high",
    #"log_ret_skip1m",

    # --- WQ Alpha signals (return-predicting, eligible for deep combiner) ---
    
    "WQ_Alpha_002",   # Volume momentum vs intraday return rank correlation
    "WQ_Alpha_006",   # -corr(open, volume, 10)
    "WQ_Alpha_007",   # Conditional momentum. Only fires on high-volume days.
    "WQ_Alpha_013",   # Close-volume co-movement. Stocks where price and volume co-move get shorted.
    "WQ_Alpha_016",   # Captures breakout/blowoff tops (same as 013 but uses 'high' instead of 'close').
    "WQ_Alpha_024",   # conditional 100d mean-reversion
    "WQ_Alpha_028",   # corr(adv20, low, 5) + midprice - close
    "WQ_Alpha_040",   # Penalizes high-volatility stocks that also have high-volume correlation (avoids blow-off tops).
    
    "WQ_Alpha_102",   # monthly price acceleration (21d second derivative)
    "WQ_Alpha_103",   # Value proxy: distance from 52w high (far below = cheap)
    "WQ_Alpha_104",   # BAB: negative rolling beta to equal-weighted market
    "WQ_Alpha_105",   # 12-1m momentum
    #"WQ_Alpha_201",   # Volume trend confirmation: price momentum × volume momentum
    #"WQ_Alpha_202",   # Residual momentum: 12-1m excess return vs equal-weighted market

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
    'n_estimators':       150,
    'max_depth':          3,
    'min_child_weight':   5,
    # Learning
    'learning_rate':      0.03,
    # Sampling
    'subsample':          0.7,
    'colsample_bytree':   0.6,
    # Regularization
    'reg_lambda':         5.0,
    'reg_alpha':          1,
    # Ranking
    'lambdarank_pair_method':         'topk',
    'lambdarank_num_pair_per_sample':  20,
    'n_jobs':          -1,
    'ndcg_exp_gain':False,

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
        
        "WQ_Alpha_002",
        "WQ_Alpha_006",
        "WQ_Alpha_007",
        "WQ_Alpha_013",
        "WQ_Alpha_016",
        "WQ_Alpha_024",
        "WQ_Alpha_028",
        "WQ_Alpha_040",

        "WQ_Alpha_102",
        "WQ_Alpha_103",
        "WQ_Alpha_104",
        "WQ_Alpha_105",
        #"WQ_Alpha_201",
        #"WQ_Alpha_202"
    ],
    
    #"price_structure": [
    #"dist_52w_high",
    #"log_ret_skip1m",]

}

ALL_GROUP_NAMES: list[str] = list(FEATURE_GROUPS.keys())


usedSymbols = ['AAA', 'ACB', 'ACC', 'ACL', 'ADG', 'ADS', 'AGG', 'AGR', 'ANV',
       'APG', 'APH', 'ASM', 'ASP', 'AST', 'BAF', 'BCE', 'BCM', 'BFC',
       'BIC', 'BID', 'BKG', 'BMC', 'BMI', 'BMP', 'BSI', 'BSR', 'BTP',
       'BVH', 'BWE', 'C32', 'CCC', 'CCL', 'CDC', 'CHP', 'CIG', 'CII',
       'CKG', 'CLL', 'CMG', 'CMX', 'CNG', 'CRC', 'CRE', 'CSM', 'CSV',
       'CTD', 'CTF', 'CTG', 'CTI', 'CTR', 'CTS', 'D2D', 'DAH', 'DBC',
       'DBD', 'DC4', 'DCL', 'DCM', 'DGW', 'DHA', 'DHC', 'DHM', 'DIG',
       'DLG', 'DMC', 'DPG', 'DPM', 'DPR', 'DRC', 'DSC', 'DSE', 'DTA',
       'DVP', 'DXG', 'DXS', 'EIB', 'ELC', 'EVE', 'EVF', 'EVG', 'FCM',
       'FCN', 'FIR', 'FIT', 'FMC', 'FPT', 'FRT', 'FTS', 'GAS', 'GDT',
       'GEE', 'GEG', 'GEX', 'GIL', 'GMD', 'GSP', 'GVR', 'HAG', 'HAH',
       'HAP', 'HAR', 'HAX', 'HCD', 'HCM', 'HDB', 'HDC', 'HDG', 'HHP',
       'HHS', 'HHV', 'HID', 'HII', 'HMC', 'HPG', 'HPX', 'HQC', 'HSG',
       'HSL', 'HT1', 'HTG', 'HTI', 'HTN', 'HUB', 'HVH', 'ICT', 'IDI',
       'IJC', 'ILB', 'IMP', 'ITC', 'ITD', 'JVC', 'KBC', 'KDC', 'KDH',
       'KHG', 'KHP', 'KMR', 'KOS', 'KSB', 'LBM', 'LCG', 'LGL', 'LHG',
       'LIX', 'LPB', 'LSS', 'MBB', 'MCM', 'MHC', 'MIG', 'MSB', 'MSH',
       'MSN', 'MWG', 'NAB', 'NAF', 'NBB', 'NCT', 'NHA', 'NHH', 'NKG',
       'NLG', 'NNC', 'NO1', 'NSC', 'NT2', 'NTL', 'NVL', 'OCB', 'OGC',
       'ORS', 'PAC', 'PAN', 'PC1', 'PDR', 'PET', 'PGC', 'PHC', 'PHR',
       'PLP', 'PLX', 'PNJ', 'POW', 'PPC', 'PTB', 'PTC', 'PTL', 'PVD',
       'PVP', 'PVT', 'QCG', 'RAL', 'REE', 'RYG', 'SAB', 'SAM', 'SAV',
       'SBG', 'SBT', 'SCR', 'SCS', 'SGN', 'SGR', 'SGT', 'SHA', 'SHB',
       'SHI', 'SIP', 'SJD', 'SJS', 'SKG', 'SMB', 'SSB', 'SSI', 'ST8',
       'STB', 'STK', 'SVD', 'SVT', 'SZC', 'SZL', 'TCB', 'TCH', 'TCI',
       'TCL', 'TCM', 'TCO', 'TDC', 'TDG', 'TDH', 'TDP', 'TEG', 'THG',
       'TIP', 'TLD', 'TLG', 'TLH', 'TMT', 'TNH', 'TNI', 'TNT', 'TPB',
       'TRC', 'TSC', 'TTA', 'TTF', 'TV2', 'TVB', 'TVS', 'UIC', 'VCB',
       'VCG', 'VCI', 'VDS', 'VFG', 'VGC', 'VHC', 'VHM', 'VIB', 'VIC',
       'VIP', 'VIX', 'VJC', 'VND', 'VNL', 'VNM', 'VOS', 'VPB', 'VPG',
       'VPH', 'VPI', 'VPL', 'VRC', 'VRE', 'VSC', 'VTO', 'VTP', 'YBM',
       'YEG',]

# ---------------------------------------------------------------------------
# Helper: flatten a list of group names → feature column list
# ---------------------------------------------------------------------------
def groups_to_features(group_names: list[str]) -> list[str]:
    features: list[str] = []
    for g in group_names:
        features.extend(FEATURE_GROUPS[g])
    return features
