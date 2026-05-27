from vnstock import Reference
from vnstock.ui import Market
import pandas as pd
from datetime import datetime
import numpy as np
import os
import re
import time
import logging

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
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies OHLCV sanity rules — no rows are ever dropped:
      - Detect any OHLC violation per row (high < low, high < open/close,
        low > open/close, or any non-positive price)
      - If ANY violation is found on a row, forward-fill ALL four OHLC
        columns for that row from the previous valid row
      - Volume is left completely untouched
    """
    price_cols = ["open", "high", "low", "close"]

    # Mark rows with any OHLC violation
    violation = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"]  > df["open"])
        | (df["low"]  > df["close"])
        | (df[price_cols] <= 0).any(axis=1)
    )

    if violation.any():
        log.debug("  Flagging %d OHLC-violation rows for ffill", violation.sum())
        # Set all four OHLC columns to NaN on bad rows, then ffill
        df.loc[violation, price_cols] = np.nan
        df[price_cols] = df[price_cols].ffill()

    return df


# ── Market data ───────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, file_path: str) -> None:
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_parquet(file_path, index=False)


def _fetch_quote(symbol: str, start_d: str, end_d: str) -> pd.DataFrame | None:
    """
    Fetches OHLCV history for a single symbol from the given source.
    Returns a cleaned DataFrame on success, or None if the source fails.
    Retries on rate-limit responses; raises immediately on other errors
    so the caller can try a fallback source.
    """
    while True:
        df = Market().equity(symbol).ohlcv(start=start_d,end=end_d,
                                           count = (
                                        datetime.strptime(end_d, "%Y-%m-%d") -
                                          datetime.strptime(start_d, "%Y-%m-%d")).days
                                           
                                           ,interval='1D')

        if df is None or df.empty:
            log.warning("! %s [%s] — no data returned", symbol)
            return None

        df = df.rename(columns={"time": "date"})
        df["Symbol"] = str(symbol)
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        df = df.drop_duplicates(subset=["date", "Symbol"]).sort_values(["Symbol", "date"])
        df = clean_ohlcv(df)

        if df.empty:
            log.warning("! %s [%s] — empty after cleaning", symbol)
            return None

        return df


def update_market_data(
    file_path: str,
    symbols: list,
    start_date: str = "2016-01-01",
    batch_size: int = 1,
    base_sleep: float = 7,
    filter_symbols: bool = True,
) -> None:
    """
    Fetches full stock history from start_date to today for every symbol.
    Existing data is preserved; freshly fetched rows overwrite on (date, Symbol).
    OHLC violations are fixed by ffill-ing the entire candle; volume is untouched.
    No rows are dropped.

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
        # Roll back to the previous weekday (skips weekends)
        days_back = 1 if today.weekday() > 0 else 3  # Monday rolls back to Friday
        last_trading_day = today - pd.Timedelta(days=days_back)
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
            df = _fetch_quote(symbol, start_date, today_str)

            if df is not None:
                log.info("✓ %s — %d rows fetched", symbol, len(df))
            else:
                log.warning("! %s — no usable data returned", symbol)

        except Exception as exc:
            msg = str(exc)
            log.error("✗ %s error: %s", symbol, msg)

            rate_match = re.search(r"(\d+)\s*(giây|seconds?)", msg, re.IGNORECASE)
            if rate_match:
                wait = int(rate_match.group(1)) + 1
                log.info("⏳ Rate limit — waiting %ds then retrying…", wait)
                time.sleep(wait)
                try:
                    df = _fetch_quote(symbol, start_date, today_str)
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

def _drop_before_last_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol, finds the latest date where volume == 0, then drops all
    rows from the beginning up to and including that date.
    Symbols with no zero-volume rows are left completely untouched.

    Example
    -------
    Symbol ACB has zero-volume rows on 2017-03-10 and 2020-06-15.
    The latest is 2020-06-15, so every ACB row with date <= 2020-06-15 is removed.
    ACB rows from 2020-06-16 onwards are kept.
    """
    if df.empty or "volume" not in df.columns:
        return df

    # Find the latest zero-volume date per symbol
    zero_vol = df[df["volume"] == 0].groupby("Symbol")["date"].max().rename("cutoff")

    if zero_vol.empty:
        return df

    df = df.merge(zero_vol.reset_index(), on="Symbol", how="left")

    # Drop rows where date <= cutoff (NaN cutoff = no zero-volume row → keep all)
    mask_drop = df["cutoff"].notna() & (df["date"] <= df["cutoff"])
    dropped = mask_drop.sum()

    if dropped:
        symbols_affected = df.loc[mask_drop, "Symbol"].nunique()
        log.info(
            "🧹 Zero-volume trim: dropped %d rows across %d symbol(s)",
            dropped, symbols_affected,
        )
        for sym, cutoff in zero_vol.items():
            log.debug("  %s — cutoff date: %s", sym, cutoff.date())

    df = df[~mask_drop].drop(columns=["cutoff"])
    return df


def _merge_and_dedup(existing: pd.DataFrame, new_chunks: list) -> pd.DataFrame:
    """
    Merges existing data with new chunks.
    On (date, Symbol) conflict the new fetch wins (keep='last').
    After deduplication, rows up to and including the latest zero-volume date
    are dropped per symbol (see _drop_before_last_zero_volume).
    """
    new_df = pd.concat(new_chunks, ignore_index=True)
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "Symbol"], keep="last")
    combined = combined[combined["date"] >= "2016-01-01"]
    combined = combined.sort_values(["Symbol", "date"]).reset_index(drop=True)
    combined = _drop_before_last_zero_volume(combined)
    combined = combined.sort_values(["Symbol", "date"]).reset_index(drop=True)
    return combined



# ── Index / indicator data ────────────────────────────────────────────────────
def fetch_indicator_data(
    ind_symbol: str,
    start_date: str,
    file_path: str,
) -> None:
    """
    Fetches market index data (e.g. VNINDEX), cleans it, and saves to a Parquet file.
    """
    today_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    try:
        log.info("→ %s fetching", ind_symbol)
        df = Market().index(ind_symbol).ohlcv(start=start_date, end=today_str, interval="1D",count = 5000)

        if df is None or df.empty:
            log.error("✗ %s — no data returned.", ind_symbol)
            return

        df = df.rename(columns={"time": "date"})
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.normalize()
            .dt.tz_localize(None)
        )
        df = df.dropna()

        if df.empty:
            log.error("✗ %s — empty after cleaning, no data saved.", ind_symbol)
            return

        _save(df, file_path)
        log.info("✓ %s — %d rows saved to %s", ind_symbol, len(df), file_path)

    except Exception as exc:
        log.error("✗ %s error: %s — no data saved.", ind_symbol, exc)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MARKET_PATH    = "market_data.parquet"
    VNINDEX_PATH   = "vnindex_data.parquet"
    START_DATE     = "2016-01-01"

    log.info("🤖 Starting data collection…")
    symbols = get_tags()
    # Symbol list is derived from the unique tickers already in the parquet,
    # keeping the fetch in sync with stored data without manual hardcoding.
    update_market_data(
        file_path=MARKET_PATH,
        symbols=symbols,
        start_date=START_DATE,
    )

    fetch_indicator_data(
        ind_symbol="VNINDEX",
        start_date=START_DATE,
        file_path=VNINDEX_PATH,
    )

    log.info("✅ Done. Market data → %s | VNINDEX → %s", MARKET_PATH, VNINDEX_PATH)