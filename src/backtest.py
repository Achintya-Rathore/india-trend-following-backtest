"""
backtest.py -- Stage 3: signal + backtest engine for ONE asset at a time.

IDEA IN ONE SENTENCE
    At every month-end, look at the last 12 months. If the asset is up, hold it next month.
    If it is down, hold cash (long/flat) or go short (long/short).

Run from the PROJECT ROOT:   python src/backtest.py
Needs: data/clean/<asset>.csv   (made by clean_data.py)

This file prints an IN-SAMPLE-ONLY quick look (up to 2018). The out-of-sample period
(2019 onwards) is deliberately NOT printed here: it is used once, in Stage 4.
"""

import argparse
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
# These are fixed BEFORE looking at any results. Do not change them after seeing numbers.
ASSETS = ["nifty50", "banknifty", "usdinr", "gold"]
LOOKBACK_MONTHS = 12        # main setting: textbook value, chosen in advance (9 and 15 = robustness only)
SIGNAL_LAG = 2              # trading days between the signal's close and the first return earned (explained below)
COST_BPS = {                # one-way cost per unit traded, in basis points (1 bp = 0.01%). My assumptions.
    "nifty50": 5.0,
    "banknifty": 5.0,
    "usdinr": 2.0,
    "gold": 3.0,
}
TRADING_DAYS = 252          # used only to turn an annual cash rate into a daily one
IN_SAMPLE_END = "2018-12-31"
OUT_OF_SAMPLE_START = "2019-01-01"   # not used in this stage


# ------------------------------------------------------------------ loading
def load_prices(name, root="."):
    """Read data/clean/<name>.csv and return the closing prices as a date-indexed Series."""
    path = os.path.join(root, "data", "clean", f"{name}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)   # first column (Date) becomes the index
    return df["close"].astype(float).sort_index()           # make sure dates are in order


# ------------------------------------------------------------------ step 1: month-end dates
def month_end_dates(prices):
    """Last trading day of each calendar month, using THIS asset's own calendar."""
    # group rows by calendar month and keep the last row of each group
    ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    # if the data stops in the middle of a month, that last "month-end" is not real -> drop it
    last_day = prices.index[-1]
    last_business_day_of_month = last_day + pd.offsets.BMonthEnd(0)   # rolls forward to the month's last weekday
    if last_day < last_business_day_of_month:
        ends = ends[:-1]
    return ends


# ------------------------------------------------------------------ step 2: the signal
def raw_signal(prices, lookback_months=LOOKBACK_MONTHS, variant="long_flat"):
    """
    The decision made at each month-end close, spread over every day until the next month-end.
      long_flat : +1 (hold) if the past-12-month return is positive, else 0 (cash)
      long_short: +1 (hold) if positive, else -1 (short)
    NaN until 12 months of history exist.
    """
    if variant not in ("long_flat", "long_short"):
        raise ValueError("variant must be 'long_flat' or 'long_short'")

    ends = month_end_dates(prices)                      # the dates on which we decide
    months = ends.to_period("M")                        # same dates, as calendar months
    if not (np.diff(months.asi8) == 1).all():           # safety: shift(12) below assumes no missing months
        raise ValueError("a calendar month is missing from the data; month-end shifting would be wrong")

    month_end_prices = prices.loc[ends]                 # ONLY month-end closes are used for the signal
    past_prices = month_end_prices.shift(lookback_months)   # the month-end close 12 month-ends earlier
    momentum = month_end_prices / past_prices - 1.0     # return over the last 12 months (known at this close)

    off_value = 0.0 if variant == "long_flat" else -1.0   # what "not positive" means for each variant
    decision = pd.Series(np.where(momentum > 0, 1.0, off_value), index=ends)
    decision[momentum.isna()] = np.nan                  # not enough history yet -> no decision

    # spread each month-end decision over all days (carry it forward until the next month-end)
    return decision.reindex(prices.index).ffill()


# ------------------------------------------------------------------ step 3: the backtest
def run_backtest(prices, lookback_months=LOOKBACK_MONTHS, variant="long_flat",
                 cost_bps=5.0, cost_multiplier=1.0, cash_rate=0.0, signal_lag=SIGNAL_LAG):
    """
    Day-by-day simulation. Returns a DataFrame, one row per trading day.

    TIMELINE for signal_lag = 2 (the default), using a real example:
      Fri 30 Oct  close : month-end. The signal is computed from prices up to this close.
      Mon  2 Nov  close : we TRADE at this close (the "next available price").
      Tue  3 Nov        : first day whose return the new position earns.
    So the position that earns day t's return was decided with data up to the close of day t-2.
    That is the look-ahead protection: it is the two shift() lines marked below.
    """
    if signal_lag < 1:
        raise ValueError("signal_lag must be >= 1 (0 would use today's close to earn today's return)")

    asset_ret = prices.pct_change()                     # simple daily return: today's close / yesterday's close - 1
    raw = raw_signal(prices, lookback_months, variant)  # decision known at each day's close

    # ---- LOOK-AHEAD PROTECTION (the only two lines that move the signal in time) ----
    target = raw.shift(signal_lag - 1)                  # position we hold AFTER trading at today's close
    held = target.shift(1)                              # position carried INTO today, so it earns today's return
    # ---------------------------------------------------------------------------------

    live = held.notna()                                 # True once the first signal has taken effect
    target0 = target.fillna(0.0)                        # before the first signal we hold nothing
    held0 = held.fillna(0.0)

    trade_size = target0.diff().abs().fillna(0.0)       # how much we trade at today's close (flat->long = 1, long->short = 2)
    cost = trade_size * (cost_bps * cost_multiplier) / 10_000.0   # trading cost as a fraction of capital, paid today

    cash_daily = (1.0 + cash_rate) ** (1.0 / TRADING_DAYS) - 1.0  # daily return on uninvested cash (0 in the base case)
    gross_ret = held0 * asset_ret + (1.0 - held0.abs()) * cash_daily   # return BEFORE costs (cash earns interest when not invested)
    net_ret = gross_ret - cost                          # return AFTER costs

    out = pd.DataFrame({
        "price": prices,
        "asset_ret": asset_ret,
        "raw_signal": raw,
        "target": target,
        "held": held,
        "trade_size": trade_size,
        "cost": cost,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "bh_ret": asset_ret,                            # buy-and-hold benchmark: always 100% invested
        "live": live,
    })
    return out.iloc[1:]                                 # first day has no return (no previous close)


# ------------------------------------------------------------------ helpers
def live_slice(bt, start=None, end=None):
    """Only the days after the first signal took effect, optionally cut to a date range."""
    return bt[bt["live"]].loc[start:end]


def total_return(daily_returns):
    """Compound daily returns into one total return."""
    return float((1.0 + daily_returns).prod() - 1.0)


# ------------------------------------------------------------------ quick look (in-sample only)
def quick_look(root="."):
    print("IN-SAMPLE ONLY (until %s). Out-of-sample stays untouched until Stage 4." % IN_SAMPLE_END)
    print("Lookback %d months | signal_lag %d | cash 0%% | costs (bps): %s\n" % (LOOKBACK_MONTHS, SIGNAL_LAG, COST_BPS))
    header = f"{'asset':10s}{'variant':12s}{'live from':12s}{'years':>6s}{'trades':>8s}{'invested':>10s}{'gross':>10s}{'net':>10s}{'buy&hold':>10s}"
    print(header)
    print("-" * len(header))
    for name in ASSETS:
        prices = load_prices(name, root)
        for variant in ("long_flat", "long_short"):
            bt = run_backtest(prices, variant=variant, cost_bps=COST_BPS[name])
            sub = live_slice(bt, end=IN_SAMPLE_END)
            years = len(sub) / TRADING_DAYS
            trades = int((sub["trade_size"] > 0).sum())            # days on which we traded
            invested = float((sub["held"].abs() > 0).mean())       # share of days with a non-zero position
            print(f"{name:10s}{variant:12s}{str(sub.index[0].date()):12s}{years:6.1f}{trades:8d}{invested:10.0%}"
                  f"{total_return(sub['gross_ret']):+10.1%}{total_return(sub['net_ret']):+10.1%}{total_return(sub['bh_ret']):+10.1%}")
    print("\n'gross' = before costs, 'net' = after costs, all are TOTAL returns over the window (not annualised).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder that contains data/clean")
    quick_look(ap.parse_args().root)
