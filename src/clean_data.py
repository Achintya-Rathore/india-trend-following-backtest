"""
clean_data.py -- Stage 2: load RAW CSVs, run data-quality checks, write CLEAN CSVs + a report.

How to run: put this file in the folder that contains `data/raw/`, then
    python clean_data.py
Reads : data/raw/*.csv           (made by download_data.py)
Writes: data/clean/<asset>.csv   (one column: close)
        outputs/cleaning_report.txt

PRINCIPLE: nothing is deleted silently. The only rows removed are
  (a) duplicate dates, (b) missing / non-positive prices, and
  (c) dates YOU list in CONFIRMED_BAD_TICKS after inspecting them.
Every removal is written to the report. Outliers are FLAGGED, never auto-removed.
"""

import argparse
import os

import numpy as np
import pandas as pd

# asset name -> raw file (without .csv). These are the series the backtest will use.
ASSETS = {  # asset -> (raw file, price column)
    "nifty50": ("nifty50_yahoo", "Close"),
    "banknifty": ("banknifty_yahoo", "Close"),
    "usdinr": ("usdinr_fred", "DEXINUS"),   # switched from Yahoo: Yahoo failed the cross-check
    "gold": ("gold_gcf_yahoo", "Close"),
}

# Dates you have CONFIRMED (by checking another source / the news) are bad ticks.
# Format: {"banknifty": ["2015-03-02"], ...}. Empty at first; fill AFTER reading the report.
CONFIRMED_BAD_TICKS = {}

OUTLIER_SIGMA = 5      # flag daily moves bigger than 5 standard deviations
GAP_DAYS = 5           # flag calendar gaps longer than 5 days
TOP_N = 10             # how many largest moves / disagreements to print

report_lines = []


def log(text=""):
    """Print and store a line, so the report file matches what you see on screen."""
    print(text)
    report_lines.append(text)


# ---------------------------------------------------------------- loading
def load_series(path, column="Close"):
    """Read one raw CSV and return the chosen price column as a float Series indexed by date."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]                         # drop junk header rows if any
    s = pd.to_numeric(df[column], errors="coerce")    # text like '.' becomes NaN
    s.name = "close"
    return s


# ---------------------------------------------------------------- per-asset checks
def check_and_clean(name, s):
    """Run all single-asset checks. Returns the cleaned Series."""
    log(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
    log(f"Raw rows: {len(s)}   first: {s.index.min().date()}   last: {s.index.max().date()}")

    # 1) order and duplicates
    if not s.index.is_monotonic_increasing:
        log("Index was NOT sorted -> sorting by date.")
        s = s.sort_index()
    dup = s.index.duplicated(keep="last")
    log(f"Duplicate dates: {dup.sum()}" + ("  -> kept the last row of each" if dup.sum() else ""))
    s = s[~dup]

    # 2) missing and impossible prices
    n_nan = s.isna().sum()
    n_bad = (s <= 0).sum()
    log(f"Missing prices: {n_nan}   Non-positive prices: {n_bad}  -> these rows removed")
    s = s[s.notna() & (s > 0)]

    # 3) weekend rows (an Indian index should have none; FX/gold may have a few)
    weekend = s[s.index.dayofweek >= 5]
    log(f"Rows on Sat/Sun: {len(weekend)}")

    # 4) calendar gaps longer than GAP_DAYS
    gaps = s.index.to_series().diff().dt.days
    big = gaps[gaps > GAP_DAYS]
    log(f"Gaps longer than {GAP_DAYS} calendar days: {len(big)}")
    for d, g in big.items():
        log(f"   gap of {int(g)} days ending {d.date()}")

    # 5) weekdays with no row (holidays, or missing data we cannot tell apart yet)
    all_weekdays = pd.bdate_range(s.index.min(), s.index.max())
    log(f"Weekdays with NO row (holidays or missing): {len(all_weekdays.difference(s.index))}")

    # 6) stale prices: same close as the previous row (typical of fake holiday rows in FX)
    same = s.diff() == 0
    run = (same != same.shift()).cumsum()
    longest = same.groupby(run).sum().max() if same.any() else 0
    log(f"Rows with the SAME close as previous row: {same.sum()}   (longest run: {int(longest)})")

    # 7) outliers: log return more than OUTLIER_SIGMA standard deviations from zero
    ret = np.log(s).diff().dropna()
    sigma = ret.std()
    flagged = ret[ret.abs() > OUTLIER_SIGMA * sigma]
    log(f"Daily vol (std of log return): {sigma:.3%}   Flagged moves > {OUTLIER_SIGMA} sigma: {len(flagged)}")
    for d, r in flagged.reindex(flagged.abs().sort_values(ascending=False).index)[:TOP_N].items():
        log(f"   {d.date()}  {r:+.2%}   <- CHECK: real event or bad tick?")

    # 8) apply the confirmed bad ticks (empty until you decide)
    for d in CONFIRMED_BAD_TICKS.get(name, []):
        if pd.Timestamp(d) in s.index:
            log(f"REMOVED confirmed bad tick on {d}")
            s = s.drop(pd.Timestamp(d))

    # 9) last rows: the newest data is the most likely to be revised
    log("Last 3 rows (provisional, may be revised):")
    for d, v in s.tail(3).items():
        log(f"   {d.date()}  {v:,.4f}")

    log(f"Clean rows: {len(s)}")
    return s


# ---------------------------------------------------------------- cross-source checks
def month_end_returns(s):
    """Last price of each calendar month, then month-on-month log return."""
    last = s.groupby(s.index.to_period("M")).last()
    return np.log(last).diff().dropna()


def compare_sources(title, primary, other, min_gap_pct=1.0):
    """Compare two series for the same asset on dates both have."""
    log(f"\n{'-' * 70}\nCROSS-CHECK: {title}\n{'-' * 70}")
    both = pd.concat([primary.rename("p"), other.rename("o")], axis=1, sort=True).dropna()
    log(f"Common dates: {len(both)}  ({both.index.min().date()} -> {both.index.max().date()})")

    # daily log returns computed on the COMMON dates so both span identical days
    r = np.log(both).diff().dropna()
    diff = (r["p"] - r["o"])
    log(f"Daily return correlation: {r['p'].corr(r['o']):.4f}")
    # Date-offset test: primary today vs the other source's return `lag` rows earlier.
    # If lag +1 beats lag 0 by a lot, the other source's dates are one day behind.
    for lag in (-1, 0, 1):
        log(f"   lag test  primary(t) vs other(t{-lag:+d}): corr {r['p'].corr(r['o'].shift(lag)):.4f}")
    mp, mo = month_end_returns(both["p"]), month_end_returns(both["o"])
    log(f"Month-end return correlation: {mp.corr(mo):.4f}")
    log(f"Mean absolute daily disagreement: {diff.abs().mean():.3%}")

    years = (both.index[-1] - both.index[0]).days / 365.25
    ann_p = (both["p"].iloc[-1] / both["p"].iloc[0]) ** (1 / years) - 1
    ann_o = (both["o"].iloc[-1] / both["o"].iloc[0]) ** (1 / years) - 1
    log(f"Annualised price return  primary: {ann_p:.2%}   other: {ann_o:.2%}   gap: {ann_p - ann_o:+.2%}")

    big = diff[diff.abs() > min_gap_pct / 100]
    log(f"Days where the two disagree by more than {min_gap_pct}%: {len(big)}")
    log("   by calendar month (clusters in roll months = futures roll artifacts):")
    log("   " + str(big.groupby(big.index.month).size().to_dict()))
    log(f"   largest {TOP_N} disagreements:")
    for d, v in diff.reindex(diff.abs().sort_values(ascending=False).index)[:TOP_N].items():
        log(f"   {d.date()}  primary-minus-other = {v:+.2%}")


def compare_nifty_banknifty(nifty, bank):
    """Both trade on the NSE calendar, so their dates should match almost exactly."""
    log(f"\n{'-' * 70}\nCROSS-CHECK: Nifty 50 vs Bank Nifty trading dates\n{'-' * 70}")
    only_n = nifty.index.difference(bank.index)
    only_b = bank.index.difference(nifty.index)
    log(f"Dates only in Nifty: {len(only_n)}   Dates only in Bank Nifty: {len(only_b)}")
    for label, dates in (("only Nifty", only_n), ("only BankNifty", only_b)):
        if len(dates):
            log(f"   {label}: " + ", ".join(str(d.date()) for d in dates[:15]) + (" ..." if len(dates) > 15 else ""))
    both = pd.concat([nifty.rename("n"), bank.rename("b")], axis=1, sort=True).dropna()
    r = np.log(both).diff().dropna()
    log(f"Daily return correlation Nifty vs Bank Nifty: {r['n'].corr(r['b']):.3f}  (diversification check)")


# ---------------------------------------------------------------- main
def main(root):
    raw_dir = os.path.join(root, "data", "raw")
    clean_dir = os.path.join(root, "data", "clean")
    out_dir = os.path.join(root, "outputs")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    clean = {}
    for name, (stem, column) in ASSETS.items():
        s = load_series(os.path.join(raw_dir, f"{stem}.csv"), column)
        clean[name] = check_and_clean(name, s)
        clean[name].to_frame().to_csv(os.path.join(clean_dir, f"{name}.csv"))

    compare_nifty_banknifty(clean["nifty50"], clean["banknifty"])

    # USD/INR: FRED (primary) vs Yahoo (kept only as a cross-check). The diagnostic showed
    # Yahoo's dates run about one day behind FRED and it has bad ticks (see the lag test below).
    yahoo_fx = load_series(os.path.join(raw_dir, "usdinr_yahoo.csv"), "Close")
    compare_sources("USD/INR  FRED (primary) vs Yahoo", clean["usdinr"], yahoo_fx.dropna(), min_gap_pct=0.5)

    # Gold: GC=F futures vs GLD ETF. Expect GLD to trail by roughly its ~0.4%/yr fee.
    gld = load_series(os.path.join(raw_dir, "gold_gld_yahoo.csv"), "Close")
    compare_sources("GOLD  GC=F (primary) vs GLD ETF", clean["gold"], gld.dropna(), min_gap_pct=1.0)

    with open(os.path.join(out_dir, "cleaning_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nClean files written to {clean_dir}\nReport saved to outputs/cleaning_report.txt")
    log("Paste the full report back to me before Stage 3.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder that contains data/raw")
    main(ap.parse_args().root)
