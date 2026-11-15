from vnstock import Reference
from vnstock import Market
import pandas as pd
from datetime import datetime
import numpy as np
import os
import re
import time
import hashlib
import logging

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_collect.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Hugging Face storage ──────────────────────────────────────────────────────
HF_REPO_ID = os.environ.get("HF_DATA_REPO", "PhongHPham/vn_cross_sectional_ranking_data_storage")
HF_TOKEN = os.environ.get("HF_TOKEN")


def _file_hash(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pull_from_hf(filename: str) -> None:
    """
    Downloads `filename` from the HF dataset repo into the working directory.

    Only a genuinely missing file (first run) is treated as "start fresh".
    Any other error (network, auth, rate limit) is re-raised: continuing from
    an empty local file would rebuild a partial history and then overwrite the
    full dataset on HF at upload time.
    """
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=".",
        )
        log.info("⬇️  Pulled %s from HF (%s)", filename, HF_REPO_ID)
    except EntryNotFoundError:
        log.warning("%s not found on HF — starting from scratch.", filename)


def push_to_hf(filename: str) -> None:
    HfApi(token=HF_TOKEN).upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"data: automated market update {pd.Timestamp.today():%Y-%m-%d}",
    )
    log.info("⬆️  Pushed %s to HF (%s)", filename, HF_REPO_ID)


# ── Symbols ───────────────────────────────────────────────────────────────────
def clean_symbols(symbol_list):
    return sorted(
        set(str(s).strip() for s in symbol_list if pd.notna(s))
    )


def get_tags(fetching=False):
    if fetching:
        ref = Reference()
        return ref.index.members("VNALL")

    return [
    "AAA", "AAM", "ABT", "ACB", "ACC", "ACL", "ADG", "ADP", "ADS", "AGG",
    "AGR", "ANV", "APG", "APH", "ASM", "ASP", "AST", "BAF", "BCE", "BCM",
    "BFC", "BIC", "BID", "BKG", "BMC", "BMI", "BMP", "BRC", "BSI", "BSR",
    "BTP", "BVH", "BWE", "C32", "CCC", "CCL", "CDC", "CHP", "CIG", "CII",
    "CKG", "CLL", "CMG", "CMX", "CNG", "CRC", "CRE", "CSM", "CSV", "CTD",
    "CTF", "CTG", "CTI", "CTR", "CTS", "D2D", "DAH", "DBC", "DBD", "DC4",
    "DCL", "DCM", "DGW", "DHA", "DHC", "DHM", "DIG", "DLG", "DMC", "DPG",
    "DPM", "DPR", "DRC", "DRL", "DSC", "DSE", "DSN", "DTA", "DVP", "DXG",
    "DXS", "DXV", "EIB", "ELC", "EVE", "EVF", "EVG", "FCM", "FCN", "FIR",
    "FIT", "FMC", "FPT", "FRT", "FTS", "GAS", "GDT", "GEE", "GEG", "GEX",
    "GIL", "GMD", "GSP", "GVR", "HAG", "HAH", "HAP", "HAR", "HAX", "HCD",
    "HCM", "HDB", "HDC", "HDG", "HHP", "HHS", "HHV", "HID", "HII", "HMC",
    "HPG", "HPX", "HQC", "HSG", "HSL", "HT1", "HTG", "HTI", "HTN", "HTV",
    "HUB", "HVH", "ICT", "IDI", "IJC", "ILB", "IMP", "ITC", "ITD", "JVC",
    "KBC", "KDC", "KDH", "KHG", "KHP", "KMR", "KOS", "KSB", "LAF", "LBM",
    "LCG", "LGL", "LHG", "LIX", "LPB", "LSS", "MBB", "MCM", "MCP", "MHC",
    "MIG", "MSB", "MSH", "MSN", "MWG", "NAB", "NAF", "NAV", "NBB", "NCT",
    "NHA", "NHH", "NKG", "NLG", "NNC", "NO1", "NSC", "NT2", "NTL", "NVL",
    "OCB", "OGC", "ORS", "PAC", "PAN", "PC1", "PDR", "PET", "PGC", "PHC",
    "PHR", "PIT", "PLP", "PLX", "PNJ", "POW", "PPC", "PTB", "PTC", "PTL",
    "PVD", "PVP", "PVT", "QCG", "RAL", "REE", "RYG", "SAB", "SAM", "SAV",
    "SBG", "SBT", "SCR", "SCS", "SFC", "SFI", "SGN", "SGR", "SGT", "SHA",
    "SHB", "SHI", "SIP", "SJD", "SJS", "SKG", "SMB", "SSB", "SSI", "ST8",
    "STB", "STK", "SVD", "SVT", "SZC", "SZL", "TCB", "TCH", "TCI", "TCL",
    "TCM", "TCO", "TCT", "TDC", "TDG", "TDH", "TDP", "TEG", "THG", "TIP",
    "TLD", "TLG", "TLH", "TMT", "TN1", "TNH", "TNI", "TNT", "TPB", "TRC",
    "TSC", "TTA", "TTF", "TV2", "TVB", "TVS", "UIC", "VCA", "VCB", "VCG",
    "VCI", "VDS", "VFG", "VGC", "VHC", "VHM", "VIB", "VIC", "VIP", "VIX",
    "VJC", "VND", "VNL", "VNM", "VOS", "VPB", "VPG", "VPH", "VPI", "VPL",
    "VPS", "VRC", "VRE", "VSC", "VSI", "VTB", "VTO", "VTP", "YBM", "YEG"
]


# ── Data cleaning ─────────────────────────────────────────────────────────────
MAX_MOVE = 0.15
P = ["open", "high", "low", "close"]

def flag_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df["f_nan"] = df[P + ["volume"]].isna().any(axis=1)
    df["f_bad_ohlc"] = (
        (df.high < df.low) | (df.high < df.open) | (df.high < df.close)
        | (df.low > df.open) | (df.low > df.close) | (df[P] <= 0).any(axis=1)
    )
    df["f_zero_vol"] = ~(df.volume > 0)  # also catches NaN
    df["f_big_move"] = df.groupby("Symbol")["close"].pct_change(fill_method=None).abs() > MAX_MOVE
    df["f_dup"] = df["f_dup"].fillna(False).astype(bool) if "f_dup" in df else False
    df["f_any"] = df[["f_nan", "f_bad_ohlc", "f_zero_vol", "f_big_move", "f_dup"]].any(axis=1)
    return df


# ── Market data ───────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, file_path: str) -> None:
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_parquet(file_path, index=False)


def _fetch_quote(symbol: str, start_d: str, end_d: str,source:str) -> pd.DataFrame | None:
    """
    Fetches OHLCV history for a single symbol from the given source.
    Returns a cleaned DataFrame on success, or None if the source fails.
    Retries on rate-limit responses; raises immediately on other errors
    so the caller can try a fallback source.
    """
    while True:
        df = Market(source).equity(symbol).ohlcv(start=start_d, end=end_d, interval="1D")

        if df is None or df.empty:
            log.warning("! %s — no data returned", symbol)
            return None

        df = df.rename(columns={"time": "date"})
        df["Symbol"] = str(symbol)
        df['source']=source
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        df["f_dup"] = df.duplicated(["date", "Symbol"], keep=False)
        df = df.drop_duplicates(["date", "Symbol"], keep="last").sort_values(["Symbol", "date"])
        # (remove: df = clean_ohlcv(df))
        if df.empty:
            log.warning("! %s — empty after cleaning", symbol)
            return None

        return df


def update_market_data(
    file_path: str,
    symbols: list,
    start_date: str = "2016-01-01",
    batch_size: int = 1,
    base_sleep: float = 0.3,
    filter_symbols: bool = True,
    source: str="dnse",
) -> None:
    """
    Fetches full stock history from start_date to today for every symbol.
    Existing data is preserved; freshly fetched rows overwrite on (date, Symbol).
    OHLC violations are fixed by ffill-ing the entire candle; volume is untouched.
    Newly fetched rows with volume <= 0 are dropped. The final DataFrame is
    sorted by Symbol and date, and saved to the given Parquet file.

    filter_symbols: if True (default), purges rows from existing data whose
    Symbol is no longer in the provided symbols list. Set to False to keep
    all previously stored symbols regardless.
    """
    today = pd.Timestamp.today().normalize()
    today_str = today.strftime("%Y-%m-%d")

    # Load existing file once
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        existing_df["date"] = (
            pd.to_datetime(existing_df["date"]).dt.normalize().dt.tz_localize(None)
        )

        if filter_symbols:
            initial_count = len(existing_df)
            existing_df = existing_df[existing_df["Symbol"].isin(symbols)]

            removed_count = initial_count - len(existing_df)
            if removed_count > 0:
                log.info("Purged %d rows: Symbols no longer in the current list.", removed_count)

        log.info("Loaded existing data: %d rows", len(existing_df))

        # Build a lookup: symbol -> latest date already stored
        latest_dates = existing_df.groupby("Symbol")["date"].max()
    else:
        existing_df = pd.DataFrame()
        latest_dates = pd.Series(dtype="datetime64[ns]")
    if os.environ.get("FORCE_REFRESH") == "1" and not existing_df.empty:
        existing_df = existing_df[~existing_df["Symbol"].isin(symbols)]
        latest_dates = latest_dates.iloc[0:0]
        log.warning("FORCE_REFRESH: discarded existing rows for %d symbols", len(symbols))

    # Last trading day (rolls back to Friday on weekends)
    # Last completed trading day:
    # - If market has closed today (after 14:45 ICT), today counts
    # - Otherwise, roll back to the previous weekday
    MARKET_CLOSE_HOUR = 14
    MARKET_CLOSE_MINUTE = 45

    now_ict = pd.Timestamp.now("Asia/Ho_Chi_Minh")
    market_closed_today = (
        now_ict.hour > MARKET_CLOSE_HOUR
        or (now_ict.hour == MARKET_CLOSE_HOUR and now_ict.minute >= MARKET_CLOSE_MINUTE)
    )

    if market_closed_today and today.weekday() < 5:
        last_trading_day = today
    else:
        last_trading_day = today - pd.offsets.BDay(1)

    new_data: list[pd.DataFrame] = []
    processed = 0

    for symbol in symbols:
        # Skip if already up to date
        if symbol in latest_dates.index:
            last = latest_dates[symbol]
            if last >= last_trading_day:
                log.info("⏭ %s — already up to date (%s), skipping", symbol, last.date())
                continue

        df = None

        try:
            log.info("→ %s fetching", symbol)
            df = _fetch_quote(symbol, start_date, today_str,source)

            if df is not None:
                log.info("✓ %s — %d rows fetched", symbol, len(df))
            else:
                log.warning("! %s — no usable data returned", symbol)

        except Exception as exc:
            msg = str(exc)
            log.error("✗ %s error: %s", symbol, msg)

            rate_match = re.search(r"(\d+)\s*(giây|seconds?)", msg, re.IGNORECASE)
            # in the except block of update_market_data
            if rate_match or "RATE_LIMIT" in msg.upper() or "429" in msg:
                wait = int(rate_match.group(1)) + 1 if rate_match else 15
                log.info("⏳ Rate limit — waiting %ds then retrying…", wait)
                time.sleep(wait)
                try:
                    df = _fetch_quote(symbol, start_date, today_str,source)
                    if df is not None:
                        log.info("✓ %s — %d rows after retry", symbol, len(df))
                except Exception as retry_exc:
                    log.error("✗ %s retry failed: %s", symbol, retry_exc)

        if df is not None:
            new_data.append(df)
        else:
            log.error("✗ %s — fetch failed, symbol skipped.", symbol)

        time.sleep(base_sleep)
        processed += 1

        # Batch save every `batch_size` symbols
        if processed % batch_size == 0 and new_data:
            existing_df = _merge_and_dedup(existing_df, new_data)
            _save(existing_df, file_path)
            log.info("💾 Batch save at %d symbols (%d total rows)", processed, len(existing_df))
            new_data = []

    # Final save
    if new_data:
        existing_df = _merge_and_dedup(existing_df, new_data)
        _save(existing_df, file_path)
        log.info("💾 Final save — %d total rows", len(existing_df))
    if os.environ.get("FORCE_REFRESH") == "1":
        have = set(existing_df["Symbol"]) if "Symbol" in existing_df else set()
        missing = sorted(set(symbols) - have)
        if missing:
            raise RuntimeError(f"FORCE_REFRESH incomplete, missing: {missing}")

def _merge_and_dedup(existing: pd.DataFrame, new_chunks: list) -> pd.DataFrame:
    """
    Merges existing data with new chunks.
    On (date, Symbol) conflict the new fetch wins (keep='last').
    After deduplication, rows having <= 0 volume
    are dropped per symbol (see _drop_before_last_zero_volume).
    """
    new_df = pd.concat(new_chunks, ignore_index=True)
    # close_raw = close as first seen (never restated). FHSC back-adjusts `close`
    # on every full refetch but leaves volume in raw shares, so close*volume is
    # only correct with close_raw. Carry it over from stored rows.
    if not existing.empty and "close_raw" not in existing.columns:
        raise RuntimeError("market_data has no close_raw: run migrate_hf.py before collecting.")
    if not existing.empty:
        prior = existing[["Symbol", "date", "close_raw"]].drop_duplicates(["Symbol", "date"])
        new_df = new_df.drop(columns=["close_raw"], errors="ignore").merge(prior, on=["Symbol", "date"], how="left")
    if "close_raw" not in new_df:
        new_df["close_raw"] = np.nan
    new_df["close_raw"] = new_df["close_raw"].fillna(new_df["close"])   # first sighting
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "Symbol"], keep="last")
    combined = combined[combined["date"] >= "2016-01-01"]
    combined = combined.sort_values(["Symbol", "date"]).reset_index(drop=True)
    combined = combined.sort_values(["Symbol", "date"]).reset_index(drop=True)
    return flag_ohlcv(combined)



# ── Index / indicator data ────────────────────────────────────────────────────
def fetch_indicator_data(
    ind_symbol: str,
    start_date: str,
    file_path: str,
    source  :str="dnse",
) -> None:
    """
    Fetches market index data (e.g. VNINDEX), cleans it, and saves to a Parquet file.
    """
    today_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    try:
        log.info("→ %s fetching", ind_symbol)
        df = Market(source).index(ind_symbol).ohlcv(start=start_date, end=today_str, interval="1D")

        if df is None or df.empty:
            log.error("✗ %s — no data returned.", ind_symbol)
            return

        df = df.rename(columns={"time": "date"})
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        df["f_nan"] = df.isna().any(axis=1)

        if df.empty:
            log.error("✗ %s — empty after cleaning, no data saved.", ind_symbol)
            return

        _save(df, file_path)
        log.info("✓ %s — %d rows saved to %s", ind_symbol, len(df), file_path)

    except Exception as exc:
        log.error("✗ %s error: %s — no data saved.", ind_symbol, exc)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    source = os.environ.get("VNSTOCK_SOURCE", "fhsc")
    sfx = "" if source == "fhsc" else f"_{source}"
    MARKET_PATH  = f"market_data{sfx}.parquet"
    VNINDEX_PATH = f"vnindex_data{sfx}.parquet"
    START_DATE     = "2016-01-01"

    log.info("🤖 Starting data collection…")
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN is not set — cannot read/write the HF dataset.")

    # Existing history lives on Hugging Face. Pull it first so update_market_data()
    # only fetches the missing days and merges into it.
    pull_from_hf(MARKET_PATH)
    pull_from_hf(VNINDEX_PATH)
    hash_before = {p: _file_hash(p) for p in (MARKET_PATH, VNINDEX_PATH)}

    symbols = get_tags()
    update_market_data(file_path=MARKET_PATH, symbols=symbols, start_date=START_DATE, source=source)

    fetch_indicator_data("VNINDEX", START_DATE, VNINDEX_PATH, source)

    # Upload only the files that actually changed.
    changed = False
    for p in (MARKET_PATH, VNINDEX_PATH):
        if _file_hash(p) is not None and _file_hash(p) != hash_before[p]:
            push_to_hf(p)
            changed = True

    # Tell the workflow whether to trigger precompute_model.yml.
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")

    log.info("✅ Done (changed=%s). Market data → %s | VNINDEX → %s", changed, MARKET_PATH, VNINDEX_PATH)