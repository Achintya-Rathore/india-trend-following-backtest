"""
test_backtest.py -- unit checks for backtest.py. They use FAKE price data, so they test the
LOGIC of the engine, not the markets.

Run from the project root:   python src/test_backtest.py
(every line printed must say PASS; the script stops with an error if a check fails)
"""

import numpy as np
import pandas as pd

from backtest import month_end_dates, run_backtest, total_return, live_slice


def make_prices(years=9, seed=1, drift=0.0003, vol=0.01):
    """Fake daily prices: a random walk on weekdays, starting 2010-01-04."""
    idx = pd.bdate_range("2010-01-04", periods=years * 260)
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(drift, vol, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(daily_returns)), index=idx, name="close")


def test_no_future_leak():
    """Scramble every price AFTER a cutoff date. Nothing up to the cutoff may change."""
    prices = make_prices()
    cutoff = prices.index[1200]
    scrambled = prices.copy()
    after = scrambled.index > cutoff
    scrambled.loc[after] *= np.exp(np.random.default_rng(99).normal(0, 0.2, after.sum()))

    a = run_backtest(prices)
    b = run_backtest(scrambled)
    # positions and returns up to the cutoff must be identical
    pd.testing.assert_series_equal(a["held"].loc[:cutoff], b["held"].loc[:cutoff])
    pd.testing.assert_series_equal(a["net_ret"].loc[:cutoff], b["net_ret"].loc[:cutoff])
    # and the scramble really did change things afterwards (otherwise the test proves nothing)
    assert not np.allclose(a["net_ret"].loc[cutoff:].values, b["net_ret"].loc[cutoff:].values)
    print("PASS  no future leak: changing future prices does not change past positions or returns")


def test_shifting_signal_changes_results():
    """Shifting the signal by one day must change the result."""
    prices = make_prices()
    runs = {lag: run_backtest(prices, signal_lag=lag) for lag in (1, 2, 3)}
    common_start = max(r[r["live"]].index[0] for r in runs.values())   # compare over the same days
    totals = {lag: total_return(r["net_ret"].loc[common_start:]) for lag, r in runs.items()}
    values = list(totals.values())
    assert abs(values[0] - values[1]) > 1e-6 and abs(values[1] - values[2]) > 1e-6, totals
    print("PASS  shifting the signal one day changes the result:",
          {lag: f"{v:+.2%}" for lag, v in totals.items()})


def test_timing_by_hand():
    """A month-end signal must take effect exactly 2 trading days later, and trades happen 1 day after month-end."""
    prices = make_prices()
    bt = run_backtest(prices)
    ends = month_end_dates(prices)
    idx = bt.index
    checked = 0
    for s in ends:
        if s not in idx:
            continue
        i = idx.get_loc(s)
        if i < 1 or i + 2 >= len(idx) or pd.isna(bt["raw_signal"].iloc[i - 1]):
            continue
        assert bt["held"].iloc[i + 1] == bt["raw_signal"].iloc[i - 1]   # day after month-end: still the OLD position
        assert bt["held"].iloc[i + 2] == bt["raw_signal"].iloc[i]       # two days after: the NEW position
        checked += 1
    assert checked > 50, checked
    # every trade happens on the trading day right after a month-end
    first_days_after = {prices.index[prices.index.get_loc(s) + 1] for s in ends if prices.index.get_loc(s) + 1 < len(prices)}
    trade_days = set(bt.index[bt["trade_size"] > 0])
    assert trade_days <= first_days_after
    print(f"PASS  timing: checked {checked} month-ends; new position starts 2 days after, trades happen 1 day after")


def test_costs():
    """Costs can only lower returns; zero cost means net == gross; higher cost means lower result."""
    prices = make_prices()
    base = live_slice(run_backtest(prices, cost_bps=5))
    free = live_slice(run_backtest(prices, cost_bps=0))
    heavy = live_slice(run_backtest(prices, cost_bps=5, cost_multiplier=4))
    assert total_return(base["net_ret"]) < total_return(base["gross_ret"])
    assert np.allclose(free["net_ret"], free["gross_ret"])
    assert total_return(heavy["net_ret"]) < total_return(base["net_ret"])
    print("PASS  costs: net < gross, zero cost gives net == gross, 4x cost is worse than 1x")


def test_downtrend_behaviour():
    """In a steady downtrend: long/flat sits in cash (0 return); long/short makes money; buy-and-hold loses."""
    idx = pd.bdate_range("2010-01-04", periods=9 * 260)
    prices = pd.Series(100 * np.exp(-0.0005 * np.arange(len(idx))), index=idx)
    lf = live_slice(run_backtest(prices, variant="long_flat"))
    ls = live_slice(run_backtest(prices, variant="long_short"))
    assert abs(total_return(lf["net_ret"])) < 1e-12
    assert total_return(ls["net_ret"]) > 0
    assert total_return(lf["bh_ret"]) < 0
    print("PASS  downtrend: long/flat = cash, long/short profits, buy-and-hold loses")


def test_uptrend_equals_buy_and_hold():
    """In a steady uptrend the strategy is always long, so it must equal buy-and-hold."""
    idx = pd.bdate_range("2010-01-04", periods=9 * 260)
    prices = pd.Series(100 * np.exp(0.0005 * np.arange(len(idx))), index=idx)
    sub = live_slice(run_backtest(prices))
    assert np.isclose(total_return(sub["net_ret"]), total_return(sub["bh_ret"]))
    print("PASS  uptrend: strategy return equals buy-and-hold when the signal never turns off")


if __name__ == "__main__":
    test_no_future_leak()
    test_shifting_signal_changes_results()
    test_timing_by_hand()
    test_costs()
    test_downtrend_behaviour()
    test_uptrend_equals_buy_and_hold()
    print("\nAll checks passed.")
