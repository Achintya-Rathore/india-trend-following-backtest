"""
download_data.py  --  Stage 1/2 helper: download RAW daily data and save as CSV.

Run from the PROJECT ROOT:  pip install yfinance pandas requests
       python src/download_data.py

Output: data/raw/<name>.csv  (untouched raw data, no cleaning here)
Window: 2010-01-01 to today.
"""

import io
import os
from datetime import date

import pandas as pd
import requests
import yfinance as yf

START = "2010-01-01"
END = date.today().isoformat()
RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# name -> yfinance ticker (primary sources)
YAHOO_TICKERS = {
    "nifty50_yahoo": "^NSEI",       # Nifty 50 (price index, no dividends)
    "banknifty_yahoo": "^NSEBANK",  # Bank Nifty
    "usdinr_yahoo": "USDINR=X",     # USD/INR spot
    "gold_gcf_yahoo": "GC=F",       # COMEX gold futures (unadjusted roll jumps!)
}

status = {}  # name -> result string, printed at the end


def save(df: pd.DataFrame, name: str) -> None:
    """Save a dataframe to data/raw/<name>.csv and record a short status line."""
    path = os.path.join(RAW_DIR, f"{name}.csv")
    df.to_csv(path)
    status[name] = f"OK   rows={len(df):5d}  {df.index.min().date()} -> {df.index.max().date()}  ({path})"


def fail(name: str, err: Exception) -> None:
    status[name] = f"FAIL {type(err).__name__}: {err}"


# ---------- 1) Yahoo Finance (primary for all four assets) ----------
for name, ticker in YAHOO_TICKERS.items():
    try:
        df = yf.download(ticker, start=START, end=END, interval="1d",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns 2-level columns
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise ValueError("empty dataframe (ticker blocked or no data)")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        save(df, name)
    except Exception as e:
        fail(name, e)

# ---------- 2) FRED DEXINUS (cross-check for USD/INR) ----------
# Daily INR per USD, noon buying rate in New York. Missing days appear as '.'.
try:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["Date", "DEXINUS"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["DEXINUS"] = pd.to_numeric(df["DEXINUS"], errors="coerce")  # '.' -> NaN
    df = df.set_index("Date").loc[START:END]
    save(df, "usdinr_fred")
except Exception as e:
    fail("usdinr_fred", e)

# ---------- 3) GLD ETF (cross-check for gold: tracks spot, no futures roll jumps) ----------
# Stooq was tried first but blocks automated downloads, so GLD is the second gold source.
try:
    df = yf.download("GLD", start=START, end=END, interval="1d",
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError("empty dataframe")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    save(df, "gold_gld_yahoo")
except Exception as e:
    fail("gold_gld_yahoo", e)

# ---------- Summary ----------
print("\n=== DOWNLOAD SUMMARY ===")
for k, v in status.items():
    print(f"{k:20s} {v}")
print("\nNifty cross-check (niftyindices.com) needs a manual CSV download; not scripted.")
print("Paste this summary back to me before we start Stage 2.")
