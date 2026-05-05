from vnstock import Listing
from vnstock import Company
import pandas as pd
import numpy as np
import os
from vnstock import Quote
import re
import time


#VN100_CROSS_SECTIONAL_RANKING
def clean_symbols(symbol_list):
    return sorted(
        list(
            set(
                str(s).strip()
                for s in symbol_list
                if pd.notna(s)
            )
        )
    )

def build_vn100(fetching = False):
    if fetching:
        listing = Listing(source='KBS')

        vn30_raw = listing.symbols_by_group('VN30')
        vnmid_raw = listing.symbols_by_group('VNMidCap')

        vn30 = clean_symbols(vn30_raw)
        vnmid = clean_symbols(vnmid_raw)

        vn100 = sorted(list(set(vn30 + vnmid)))

        print("VN30:", len(vn30))
        print("VNMID:", len(vnmid))
        print("VN100:", len(vn100))

        return vn100
    return ["ACB", "ANV", "BCM", "BID", "BMP", "BSI", "BSR", "BVH"
            , "BWE", "CII", "CMG", "CTD", "CTG", "CTR", "CTS", "DBC"
            , "DCM", "DGC", "DGW", "DIG", "DPM", "DSE", "DXG", "DXS"
            , "EIB", "EVF", "FPT", "FRT", "FTS", "GAS", "GEE", "GEX"
            , "GMD", "GVR", "HAG", "HCM", "HDB", "HDC", "HDG", "HHV"
            , "HPG", "HSG", "HT1", "IMP", "KBC", "KDC", "KDH", "KOS"
            , "LPB", "MBB", "MSB", "MSN", "MWG", "NAB", "NKG", "NLG"
            , "NT2", "NVL", "OCB", "PAN", "PC1", "PDR", "PHR", "PLX"
            , "PNJ", "POW", "PVD", "PVT", "REE", "SAB", "SBT", "SCS"
            , "SHB", "SIP", "SJS", "SSB", "SSI", "STB", "SZC", "TCB"
            , "TCH", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM"
            , "VIB", "VIC", "VIX", "VJC", "VND", "VNM", "VPB", "VPI"
            , "VPL", "VRE", "VSC", "VTP"]



def update_market_data(file_path, symbols, start_date="2018-01-01", batch_size=5):
    """
    Fetches full stock history from start_date to today, ignoring 
    any existing dates in the database.
    """
    all_data = []
    processed_count = 0
    today = pd.Timestamp.today().normalize().tz_localize(None)

    # Load existing data just to preserve symbols NOT in the current 'symbols' list 
    # (Optional: if you want to completely wipe the file, set existing_df = pd.DataFrame())
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.normalize().dt.tz_localize(None)
    else:
        existing_df = pd.DataFrame()

    # --- FETCHING LOOP ---
    for company in symbols:
        # ALWAYS fetch from the absolute start_date
        fetch_start = start_date
        
        while True:
            try:
                df = Quote(symbol=company, source='VCI').history(
                    start=fetch_start,
                    end=today.strftime('%Y-%m-%d'),
                    interval="1D"
                )

                if df is not None and not df.empty:
                    # --- CLEANING DATA ---
                    df = df.rename(columns={"time": "date"})
                    df["Symbol"] = str(company)
                    
                    df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.normalize().dt.tz_localize(None)
                    df = df.drop_duplicates(subset=["date", "Symbol"]).sort_values(["Symbol", "date"])
                    df["close"] = df["close"].mask(df["close"] <= 0).ffill()
                    df = df.dropna(subset=["close", "date"])
                    if not df.empty:
                        all_data.append(df)
                        print(f"✓ Fetched FULL history for {company} ({len(df)} rows)")
                else:
                    print(f"! No data found for {company} since {start_date}")
                
                time.sleep(3) # Respect rate limits
                break

            except Exception as e:
                error_msg = str(e)
                print(f"Error on {company}: {error_msg}")
                match = re.search(r"(\d+)\s*(giây|seconds?)", error_msg, re.IGNORECASE)
                if match:
                    wait_seconds = int(match.group(1))
                    print(f"⏳ Rate limit hit! Waiting {wait_seconds} seconds...")
                    time.sleep(wait_seconds + 1)
                else:
                    break

        processed_count += 1

        # --- BATCH SAVING ---
        if processed_count % batch_size == 0 and all_data:
            new_batch_df = pd.concat(all_data, ignore_index=True)
            
            # Combine and deduplicate. Since we fetched full history, 
            # 'keep="last"' ensures the freshest data wins.
            if not existing_df.empty:
                existing_df = pd.concat([existing_df, new_batch_df], ignore_index=True)
            else:
                existing_df = new_batch_df
                
            existing_df = existing_df.drop_duplicates(subset=["date", "Symbol"], keep="last")
            existing_df = existing_df.sort_values(["Symbol", "date"])
            
            dir_name = os.path.dirname(file_path)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            existing_df.to_parquet(file_path, index=False)
            
            print(f"💾 Saved batch at {processed_count} symbols")
            all_data = [] 

    # --- FINAL SAVE ---
    if all_data:
        new_batch_df = pd.concat(all_data, ignore_index=True)
        existing_df = pd.concat([existing_df, new_batch_df], ignore_index=True)
        existing_df = existing_df.drop_duplicates(subset=["date", "Symbol"], keep="last")
        existing_df = existing_df.sort_values(["Symbol", "date"])
        existing_df.to_parquet(file_path, index=False)
        print("💾 Final full-refresh save completed.")

def fetch_indicator_data(ind_symbol, start_date, file_path):
    """
    Fetches market index data (e.g., VNINDEX), cleans it, 
    and saves it to a Parquet file.
    """
    # 1. Get today as a Timestamp
    today_ts = pd.Timestamp.today().normalize().tz_localize(None)
    
    # 2. Convert to string format 'YYYY-MM-DD' for the API!
    today_str = today_ts.strftime('%Y-%m-%d')
    
    try:
        quote = Quote(symbol=ind_symbol, source='VCI')
        
        # FIX: Pass today_str instead of the Timestamp object
        df = quote.history(start=start_date, end=today_str, interval='1D')
        
        if df is not None and not df.empty:
            df = df.rename(columns={'time': 'date'})
            
            # Standardize dates
            df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.normalize().dt.tz_localize(None)
            
            # Drop NAs
            df = df.dropna()
            
            if not df.empty:
                dir_name = os.path.dirname(file_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                df.to_parquet(file_path, index=False)
                print(f"✓ Successfully saved {ind_symbol} data ({len(df)} rows) to {file_path}")
            else:
                print(f"! {ind_symbol} data was empty after dropping NA values.")
        else:
            print(f"! No data returned from API for {ind_symbol}.")
            
    except Exception as e:
        print(f"Error fetching {ind_symbol}: {e}")


if __name__ == "__main__":
    import os
    
    # Define where the data should be saved
    # The GitHub Action expects it in data/market_data.parquet
    SAVE_PATH = "market_data.parquet"
    
    print("🤖 Robot starting data collection...")
    
    # 1. Get the current list of VN100 symbols
    # This ensures we pick up any new symbols added to the index
    vn100_symbols = build_vn100()
    
    # 2. Run the update function
    # This will fetch only the missing dates (delta) up to today
    update_market_data(
        file_path=SAVE_PATH, 
        symbols=vn100_symbols,
        start_date="2018-01-01"
    )
    
    print(f"✅ Success! Data saved to {SAVE_PATH}")