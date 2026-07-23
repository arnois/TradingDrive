# schwabttk/price_history.py
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from schwab.client import Client
from schwabttk.schwab_auth import get_client

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "futures")
SYMBOLS_FILE = os.path.join(BASE_DIR, "data", "schwab_futures_symbols.txt")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Storage helpers ───────────────────────────────────────────────────────────
def load_futures_symbols(path: str = SYMBOLS_FILE) -> dict:
    """
    Load futures symbols from txt file.
    Format: TICKER,/SYMBOL,Description
    Lines starting with # or empty are ignored.
    Returns dict {ticker: symbol}
    """
    futures = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                ticker, symbol = parts[0].strip(), parts[1].strip()
                futures[ticker] = symbol
    print(f"[INFO] Loaded {len(futures)} symbols from {path}")
    return futures

def get_storage_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}.csv")

def load_stored(ticker: str) -> pd.DataFrame:
    """Load existing data if it exists, else return empty DataFrame."""
    path = get_storage_path(ticker)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    return pd.DataFrame()

def save_stored(ticker: str, df: pd.DataFrame):
    """Save DataFrame to parquet, sorted and deduplicated."""
    df = df.drop_duplicates(subset="datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    df.to_csv(get_storage_path(ticker), index=False)

def get_missing_ranges(stored: pd.DataFrame,
                       start: datetime,
                       end: datetime) -> list[tuple]:
    """
    Compare requested date range against stored data.
    Returns list of (range_start, range_end) tuples that need to be fetched.
    Handles three cases:
      1. No stored data         → fetch full range
      2. Stored data exists     → fetch only missing head/tail
      3. Fully covered          → nothing to fetch
    """
    if stored.empty:
        print(f"    → No local data found. Will fetch full range.")
        return [(start, end)]

    stored_start = stored["datetime"].min().to_pydatetime()
    stored_end   = stored["datetime"].max().to_pydatetime()

    print(f"    → Local data: {stored_start.date()} to {stored_end.date()}")

    ranges = []

    # Gap at the beginning (requested start is before stored start)
    if start < stored_start - timedelta(days=1):
        gap_end = stored_start - timedelta(days=1)
        print(f"    → Missing head: {start.date()} to {gap_end.date()}")
        ranges.append((start, gap_end))

    # Gap at the end (requested end is after stored end)
    if end > stored_end + timedelta(days=1):
        gap_start = stored_end + timedelta(days=1)
        print(f"    → Missing tail: {gap_start.date()} to {end.date()}")
        ranges.append((gap_start, end))

    if not ranges:
        print(f"    → Fully covered. No fetch needed.")

    return ranges


# ── Schwab fetch ──────────────────────────────────────────────────────────────
def fetch_from_schwab(client: Client,
                      symbol: str,
                      start: datetime,
                      end: datetime,
                      frequency_type: str = "daily",
                      frequency: int = 1) -> pd.DataFrame:
    """Fetch OHLC from Schwab API for a specific date range."""
    try:
        response = client.get_price_history(
            symbol=symbol,
            period_type=client.PriceHistory.PeriodType.YEAR,
            frequency_type=client.PriceHistory.FrequencyType(frequency_type),
            frequency=client.PriceHistory.Frequency(frequency),
            start_datetime=start,
            end_datetime=end,
            need_extended_hours_data=True
        )
    except Exception as e:
        print(f"    [ERROR] Request failed: {e}")
        return pd.DataFrame()

    if response.status_code != 200:
        print(f"    [ERROR] HTTP {response.status_code}: {response.text}")
        return pd.DataFrame()

    data = response.json()

    if not data.get("candles"):
        print(f"    [WARN] No candles returned")
        return pd.DataFrame()

    df = pd.DataFrame(data["candles"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df


# ── Main logic ────────────────────────────────────────────────────────────────
def pull_all(start: datetime,
             end: datetime,
             frequency_type: str = "daily",
             frequency: int = 1) -> dict:
    """
    Smart pull: only fetches what is not already stored.
    Appends new data to existing parquet files.
    """
    FUTURES = load_futures_symbols()
    print(f"\n{'='*60}")
    print(f"  Requested range : {start.date()} → {end.date()}")
    print(f"  Frequency       : {frequency} {frequency_type}")
    print(f"  Symbols         : {list(FUTURES.keys())}")
    print(f"{'='*60}\n")

    client  = get_client()
    results = {}

    for ticker, symbol in FUTURES.items():
        print(f"  [{ticker}] {symbol}")

        # Load what we already have
        stored = load_stored(ticker)

        # Determine what needs to be fetched
        missing_ranges = get_missing_ranges(stored, start, end)

        if not missing_ranges:
            # Nothing to fetch — clip stored to requested range and return
            mask = (stored["datetime"] >= pd.Timestamp(start)) & \
                   (stored["datetime"] <= pd.Timestamp(end))
            results[ticker] = stored[mask].reset_index(drop=True)
            print(f"    ✓ Loaded {len(results[ticker])} bars from local storage\n")
            continue

        # Fetch each missing range from Schwab
        new_frames = []
        for (r_start, r_end) in missing_ranges:
            print(f"    → Fetching from Schwab: {r_start.date()} to {r_end.date()}")
            df_new = fetch_from_schwab(client, symbol, r_start, r_end,
                                       frequency_type, frequency)
            if not df_new.empty:
                new_frames.append(df_new)
                print(f"    ✓ Got {len(df_new)} new bars")

        if not new_frames:
            print(f"    [WARN] No new data retrieved\n")
            if not stored.empty:
                results[ticker] = stored
            continue

        # Merge new data with stored data
        combined = pd.concat([stored] + new_frames, ignore_index=True)
        save_stored(ticker, combined)

        # Return only the requested range
        combined["datetime"] = pd.to_datetime(combined["datetime"])
        mask = (combined["datetime"] >= pd.Timestamp(start)) & \
               (combined["datetime"] <= pd.Timestamp(end))
        results[ticker] = combined[mask].reset_index(drop=True)

        print(f"    ✓ Total stored: {len(combined)} bars | "
              f"Returned: {len(results[ticker])} bars\n")

    print(f"{'='*60}")
    print(f"  Done! {len(results)}/{len(FUTURES)} symbols ready")
    print(f"{'='*60}\n")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull futures OHLC from Schwab")
    parser.add_argument("--start", type=str,
                        default=(datetime.today() - timedelta(days=365*2)).strftime("%Y-%m-%d"),
                        help="Start date YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", type=str,
                        default=datetime.today().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--frequency_type", type=str, default="daily",
                        choices=["minute", "daily", "weekly", "monthly"],
                        help="Bar frequency type (default: daily)")
    parser.add_argument("--frequency", type=int, default=1,
                        help="Bar frequency size (default: 1)")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")

    data = pull_all(
        start=start_dt,
        end=end_dt,
        frequency_type=args.frequency_type,
        frequency=args.frequency
    )

    # Quick preview
    for ticker, df in data.items():
        if not df.empty:
            print(f"\n── {ticker} ({len(df)} bars) ──")
            print(df.tail(3).to_string(index=False))