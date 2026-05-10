from vnstock import Listing, Company, Quote
import pandas as pd
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


def build_vn100(fetching=False):
    if fetching:
        listing = Listing(source="KBS")
        vn30  = clean_symbols(listing.symbols_by_group("VN30"))
        vnmid = clean_symbols(listing.symbols_by_group("VNMidCap"))
        vn100 = sorted(set(vn30 + vnmid))
        log.info("VN30=%d  VNMID=%d  VN100=%d", len(vn30), len(vnmid), len(vn100))
        return vn100

    return [
    "AAA", "ACB", "ADS", "VRE", "AGR", "ANV", "BFC", "BHN", "BID", "BMI",
    "BMP", "BSI", "BVH", "BWE", "CII", "CMG", "CTS", "CSV", "CTD", "CTG",
    "DBC", "DCL", "DCM", "DGC", "DGW", "DHG", "DIG", "DMC", "DPM", "DPR",
    "DRC", "DVP", "DXG", "EIB", "FMC", "FPT", "FTS", "GAS", "GEX", "GMD",
    "HAG", "HCM", "HDB", "HDC", "HDG", "HPG", "HSG", "HT1", "HVN", "IMP",
    "KBC", "KDC", "KDH", "LPB", "MBB", "MSN", "MWG", "NKG", "NLG", "NT2",
    "NVL", "PAN", "PC1", "PDR", "PHR", "PLX", "PNJ", "POW", "PPC", "PTB",
    "PVD", "PVT", "RAL", "REE", "SAB", "SBT", "SCS", "SHB", "SJS", "SSI",
    "STB", "STK", "TBC", "TCB", "TCH", "TCM", "TMS", "TRA", "VCB", "VCF",
    "VCG", "VCI", "VGC", "VHC", "VIB", "VIC", "VJC", "VND", "VNM", "VPB"]


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


def _fetch_quote(symbol: str, start: str, end: str, source: str) -> pd.DataFrame | None:
    """
    Fetches OHLCV history for a single symbol from the given source.
    Returns a cleaned DataFrame on success, or None if the source fails.
    Retries on rate-limit responses; raises immediately on other errors
    so the caller can try a fallback source.
    """
    while True:
        df = Quote(symbol=symbol, source=source).history(
            start=start,
            end=end,
            interval="1D",
        )

        if df is None or df.empty:
            log.warning("! %s [%s] — no data returned", symbol, source)
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
            log.warning("! %s [%s] — empty after cleaning", symbol, source)
            return None

        return df


def update_market_data(
    file_path: str,
    symbols: list,
    start_date: str = "2019-01-01",
    batch_size: int = 5,
    base_sleep: float = 3.0,
    sources: list[str] = ("VCI", "KBS"),
    filter_symbols: bool = True,
) -> None:
    """
    Fetches full stock history from start_date to today for every symbol.
    Tries each source in `sources` order; falls back to the next source on
    any timeout, rate-limit, or other error.
    Existing data is preserved; freshly fetched rows overwrite on (date, Symbol).
    OHLC violations are fixed by ffill-ing the entire candle; volume is untouched.
    No rows are dropped.

    filter_symbols: if True (default), purges rows from existing data whose
    Symbol is no longer in the provided symbols list. Set to False to keep
    all previously stored symbols regardless.
    """
    today_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

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
    else:
        existing_df = pd.DataFrame()

    new_data: list[pd.DataFrame] = []
    processed = 0

    for symbol in symbols:
        df = None

        for source in sources:
            try:
                log.info("→ %s fetching from [%s]", symbol, source)
                df = _fetch_quote(symbol, start_date, today_str, source)

                if df is not None:
                    log.info("✓ %s [%s] — %d rows fetched", symbol, source, len(df))
                    break  # success; no need to try next source

                # Source returned empty — try next source immediately
                log.warning("! %s [%s] — no usable data, trying next source…", symbol, source)

            except Exception as exc:
                msg = str(exc)
                log.error("✗ %s [%s] error: %s", symbol, source, msg)

                # If it's a rate-limit with an explicit wait time, honour it and
                # retry the SAME source once before falling back.
                rate_match = re.search(r"(\d+)\s*(giây|seconds?)", msg, re.IGNORECASE)
                if rate_match:
                    wait = int(rate_match.group(1)) + 1
                    log.info("⏳ Rate limit on [%s] — waiting %ds then retrying…", source, wait)
                    time.sleep(wait)
                    try:
                        df = _fetch_quote(symbol, start_date, today_str, source)
                        if df is not None:
                            log.info("✓ %s [%s] — %d rows after retry", symbol, source, len(df))
                            break
                    except Exception as retry_exc:
                        log.error("✗ %s [%s] retry failed: %s", symbol, source, retry_exc)

                # Fall through to next source
                log.warning("⚠ %s [%s] failed — falling back to next source…", symbol, source)

        if df is not None:
            new_data.append(df)
        else:
            log.error("✗ %s — all sources exhausted, symbol skipped.", symbol)

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


def _merge_and_dedup(existing: pd.DataFrame, new_chunks: list) -> pd.DataFrame:
    """
    Merges existing data with new chunks.
    On (date, Symbol) conflict the new fetch wins (keep='last').
    """
    new_df = pd.concat(new_chunks, ignore_index=True)
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "Symbol"], keep="last")
    combined = combined[combined["date"] >= "2018-01-01"]          # ← add this line
    combined = combined.sort_values(["Symbol", "date"]).reset_index(drop=True)
    return combined



# ── Index / indicator data ────────────────────────────────────────────────────
def fetch_indicator_data(
    ind_symbol: str,
    start_date: str,
    file_path: str,
    sources: list[str] = ("VCI", "KBS"),
) -> None:
    """
    Fetches market index data (e.g. VNINDEX), cleans it, and saves to a Parquet
    file. Tries each source in `sources` order; falls back on any error.
    """
    today_str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    for source in sources:
        try:
            log.info("→ %s fetching from [%s]", ind_symbol, source)
            df = Quote(symbol=ind_symbol, source=source).history(
                start=start_date, end=today_str, interval="1D"
            )

            if df is None or df.empty:
                log.warning("! %s [%s] — no data returned, trying next source…", ind_symbol, source)
                continue

            df = df.rename(columns={"time": "date"})
            df["date"] = (
                pd.to_datetime(df["date"], errors="coerce")
                .dt.normalize()
                .dt.tz_localize(None)
            )
            df = df.dropna()

            if df.empty:
                log.warning("! %s [%s] — empty after cleaning, trying next source…", ind_symbol, source)
                continue

            _save(df, file_path)
            log.info("✓ %s [%s] — %d rows saved to %s", ind_symbol, source, len(df), file_path)
            return  # success

        except Exception as exc:
            log.error("✗ %s [%s] error: %s — trying next source…", ind_symbol, source, exc)

    log.error("✗ %s — all sources exhausted, no data saved.", ind_symbol)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MARKET_PATH    = "market_data.parquet"
    VNINDEX_PATH   = "vnindex_data.parquet"
    START_DATE     = "2018-01-01"

    log.info("🤖 Starting data collection…")

    vn100_symbols = build_vn100(fetching=True)

    update_market_data(
        file_path=MARKET_PATH,
        symbols=vn100_symbols,
        start_date=START_DATE,
    )

    fetch_indicator_data(
        ind_symbol="VNINDEX",
        start_date=START_DATE,
        file_path=VNINDEX_PATH,
    )

    log.info("✅ Done. Market data → %s | VNINDEX → %s", MARKET_PATH, VNINDEX_PATH)