import os, sys, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

R = "PhongHPham/vn_cross_sectional_ranking_data_storage"
rd = lambda n: pd.read_parquet(hf_hub_download(R, n, repo_type="dataset", token=os.environ["HF_TOKEN"]))
f, d = rd("market_data.parquet"), rd("market_data_dnse.parquet")

c = f.merge(d, on=["Symbol", "date"], suffixes=("_f", "_d"))
c = c[(c.volume_f > 0) & (c.volume_d > 0)]
last = c.date.max()
x = c[c.date == last]
share = (np.log(x.close_d / x.close_f).abs() > 0.005).mean()
print(f"common last date {last.date()} (FHSC last {f.date.max().date()}), n={len(x)}, mismatch share={share:.1%}")
if len(x) < 100 or share > 0.10:
    sys.exit("ALARM: sources disagree or too few common symbols, do not trade off DNSE prices today")