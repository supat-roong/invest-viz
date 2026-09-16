"""Metrics derived from a simulated history.

Contributions arrive over time, so plain CAGR on portfolio value is
meaningless; the headline figure is the money-weighted return (XIRR).
Everything here is pure and dependency-light: no scipy, so the XIRR root is
found by a Newton step with a bracketed bisection fallback of our own.

These functions run once per ticker per frame (order 10^4 calls for one
video), so the internals stay in numpy and never iterate a DatetimeIndex in
Python.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .types import PortfolioHistory, TickerStats

DAYS_PER_YEAR = 365.0
_MAX_ITERATIONS = 100
_NEWTON_ITERATIONS = 40
_TOLERANCE = 1e-10

# Coarse scan points used to bracket a sign change before bisecting.
_SCAN_RATES = np.array((
    -0.999999, -0.9999, -0.999, -0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2,
    -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0,
    10.0, 100.0, 1000.0, 10000.0,
))

NAN = float('nan')


def xirr(cashflows, guess: float = 0.1) -> float:
    """Money-weighted annualised return, or nan when it cannot be solved."""
    try:
        dates = []
        amounts = []
        for date, amount in cashflows:
            amount = float(amount)
            if amount != amount or amount == 0.0:
                continue
            dates.append(pd.Timestamp(date))
            amounts.append(amount)
        if len(amounts) < 2:
            return NAN
        index = pd.DatetimeIndex(dates)
        years = _years_from(index, index.min())
        return _xirr_arrays(years, np.asarray(amounts, dtype=float), guess)
    except Exception:
        return NAN


def _xirr_arrays(years: np.ndarray, amounts: np.ndarray,
                 guess: float = 0.1) -> float:
    """XIRR from pre-built arrays, skipping all Timestamp boxing."""
    if amounts.size < 2:
        return NAN
    if not (amounts > 0).any() or not (amounts < 0).any():
        return NAN

    rate = _newton(years, amounts, float(guess))
    if rate is not None:
        return rate
    return _scan_and_bisect(years, amounts, float(guess))


def _newton(years: np.ndarray, amounts: np.ndarray, guess: float):
    """Newton iteration on NPV; None when it leaves the valid domain."""
    rate = guess
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for _ in range(_NEWTON_ITERATIONS):
            if not np.isfinite(rate) or rate <= -1.0:
                return None
            base = 1.0 + rate
            discount = base ** -years
            value = float(np.sum(amounts * discount))
            derivative = float(np.sum(-years * amounts * discount / base))
            if not np.isfinite(value) or not np.isfinite(derivative):
                return None
            if abs(value) < 1e-9:
                return float(rate)
            if derivative == 0.0:
                return None
            step = value / derivative
            rate -= step
            if abs(step) < _TOLERANCE:
                return float(rate) if rate > -1.0 else None
    return None


def _scan_and_bisect(years: np.ndarray, amounts: np.ndarray,
                     guess: float) -> float:
    rates = np.unique(np.append(_SCAN_RATES, guess))
    previous_rate = None
    previous_npv = None
    for rate in rates:
        value = _npv(float(rate), years, amounts)
        if not np.isfinite(value):
            previous_rate, previous_npv = None, None
            continue
        if value == 0.0:
            return float(rate)
        if previous_npv is not None and np.sign(value) != np.sign(previous_npv):
            return _bisect(previous_rate, float(rate), years, amounts)
        previous_rate, previous_npv = float(rate), value
    return NAN


def max_drawdown(values) -> float:
    """Most negative peak-to-trough fraction, e.g. -0.542; 0.0 if never down."""
    array = (values.to_numpy(dtype=float) if hasattr(values, 'to_numpy')
             else np.asarray(values, dtype=float))
    if array.ndim != 1 or array.size < 2:
        return 0.0
    array = array[~np.isnan(array)]
    if array.size < 2:
        return 0.0

    live = np.flatnonzero(array != 0.0)
    if live.size == 0:
        return 0.0
    array = array[live[0]:]
    if array.size < 2:
        return 0.0

    peak = np.maximum.accumulate(array)
    valid = peak > 0
    if not valid.any():
        return 0.0
    drawdown = np.empty_like(array)
    drawdown[:] = 0.0
    np.divide(array - peak, peak, out=drawdown, where=valid)
    return float(min(drawdown[valid].min(), 0.0))


def compute_stats(history: PortfolioHistory,
                  as_of: pd.Timestamp = None) -> TickerStats:
    """Metrics for `history` truncated at `as_of` (default: its last date)."""
    frame = history.frame
    if as_of is not None:
        frame = frame.loc[:pd.Timestamp(as_of)]
    if len(frame) == 0:
        return TickerStats(ticker=history.ticker, value=0.0, invested=0.0,
                           total_return=0.0, xirr=NAN,
                           max_drawdown=0.0, cum_dividends=0.0)

    values = frame['value'].to_numpy(dtype=float)
    invested_series = frame['invested'].to_numpy(dtype=float)

    value = float(values[-1])
    invested = float(invested_series[-1])
    cum_dividends = float(frame['cum_dividends'].to_numpy(dtype=float)[-1])
    total_return = value / invested - 1.0 if invested else 0.0

    contributions = np.diff(invested_series, prepend=0.0)
    paid = np.flatnonzero(contributions)
    if paid.size:
        flow_index = frame.index[paid].append(frame.index[[-1]])
        flow_amounts = np.append(-contributions[paid], value)
        years = _years_from(flow_index, flow_index[0])
        money_weighted = _xirr_arrays(years, flow_amounts)
    else:
        money_weighted = NAN

    return TickerStats(
        ticker=history.ticker,
        value=value,
        invested=invested,
        total_return=total_return,
        xirr=money_weighted,
        max_drawdown=max_drawdown(values),
        cum_dividends=cum_dividends,
    )


def rank_stats(stats: list) -> list:
    """New TickerStats ranked 1..N by descending value, ties by ticker name."""
    ordered = sorted(stats, key=lambda s: (-s.value, s.ticker))
    return [replace(item, rank=position) for position, item in enumerate(ordered, 1)]


def _years_from(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Year fractions since `origin`, independent of the index time unit.

    pandas 3 no longer guarantees nanosecond-backed timestamps, so dividing a
    raw int64 view by a nanosecond constant silently scales the result.
    """
    return np.asarray((index - origin) / pd.Timedelta(days=1),
                      dtype=float) / DAYS_PER_YEAR


def _npv(rate: float, years: np.ndarray, amounts: np.ndarray) -> float:
    if rate <= -1.0:
        return NAN
    with np.errstate(over='ignore', invalid='ignore'):
        return float(np.sum(amounts / np.power(1.0 + rate, years)))


def _bisect(low: float, high: float, years: np.ndarray,
            amounts: np.ndarray) -> float:
    low_npv = _npv(low, years, amounts)
    for _ in range(_MAX_ITERATIONS):
        middle = (low + high) / 2.0
        middle_npv = _npv(middle, years, amounts)
        if not np.isfinite(middle_npv):
            return NAN
        if middle_npv == 0.0 or (high - low) / 2.0 < _TOLERANCE:
            return float(middle)
        if np.sign(middle_npv) == np.sign(low_npv):
            low, low_npv = middle, middle_npv
        else:
            high = middle
    return float((low + high) / 2.0)
