"""Pure per-day portfolio simulation.

No I/O and no network: `simulate` turns a `PriceData` plus a `PortfolioSpec`
into a `PortfolioHistory`. Splits are annotation only — yfinance's
unadjusted close is already split-adjusted, so share counts are never
scaled by a split ratio.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .types import (
    CLOSE,
    DIVIDENDS,
    SPLITS,
    DcaFrequency,
    EventKind,
    MarketEvent,
    PortfolioHistory,
    PortfolioSpec,
    PriceData,
)

HISTORY_COLUMNS = ('price', 'shares', 'cash', 'invested', 'cum_dividends', 'value')

_STEPS = {
    DcaFrequency.DAILY: 'D',
    DcaFrequency.WEEKLY: '7D',
    DcaFrequency.BIWEEKLY: '14D',
    DcaFrequency.MONTHLY: pd.DateOffset(months=1),
    DcaFrequency.QUARTERLY: pd.DateOffset(months=3),
    DcaFrequency.YEARLY: pd.DateOffset(years=1),
}


def contribution_dates(index: pd.DatetimeIndex, frequency: DcaFrequency,
                       start: pd.Timestamp) -> pd.DatetimeIndex:
    """Trading days on which a recurring contribution lands.

    Targets are anchored on `start` and stepped by `frequency`, then snapped
    forward to the next date present in `index` so that a contribution is
    never dropped for landing on a weekend or market holiday. Targets past
    the end of `index` are discarded and each date is returned once.
    """
    freq = DcaFrequency(frequency)
    if freq is DcaFrequency.NONE or len(index) == 0:
        return index[:0]

    start = pd.Timestamp(start).normalize()
    last = index[-1]
    if start > last:
        return index[:0]

    targets = pd.date_range(start=start, end=last, freq=_STEPS[freq])
    if len(targets) == 0:
        return index[:0]

    positions = index.searchsorted(targets, side='left')
    positions = positions[positions < len(index)]
    return index[np.unique(positions)]


def simulate(prices: PriceData, spec: PortfolioSpec) -> PortfolioHistory:
    """Run `spec` against `prices` one trading day at a time."""
    index = prices.dates
    size = len(index)
    close = prices.close.to_numpy(dtype=float)
    dividends = np.nan_to_num(prices.dividends.to_numpy(dtype=float))
    splits = np.nan_to_num(prices.splits.to_numpy(dtype=float))

    first_day = prices.first_trading_day
    first_pos = index.get_loc(first_day) if first_day is not None else -1

    scheduled = np.zeros(size, dtype=bool)
    if first_day is not None and spec.dca_amount > 0:
        due = contribution_dates(index, spec.dca_frequency, first_day)
        scheduled[index.get_indexer(due)] = True

    shares = 0.0
    buy_cash = 0.0  # awaiting a purchase: whole-share remainder, reinvested dividends
    div_cash = 0.0  # dividends taken as cash, never spent
    invested = 0.0
    cum_dividends = 0.0

    out_shares = np.zeros(size)
    out_cash = np.zeros(size)
    out_invested = np.zeros(size)
    out_dividends = np.zeros(size)
    out_value = np.zeros(size)
    events: list[MarketEvent] = []

    for i in range(size):
        price = close[i]
        live = price == price and price > 0

        if splits[i] > 0 and splits[i] != 1.0:
            events.append(MarketEvent(prices.ticker, index[i], EventKind.SPLIT,
                                      float(splits[i])))

        if live:
            amount = 0.0
            if i == first_pos:
                amount += spec.initial_amount
            if scheduled[i]:
                amount += spec.dca_amount
            if amount > 0:
                invested += amount
                buy_cash += amount
                bought, buy_cash = _buy(buy_cash, price, spec.fractional_shares)
                shares += bought

            received = shares * dividends[i]
            if received > 0:
                cum_dividends += received
                events.append(MarketEvent(prices.ticker, index[i],
                                          EventKind.DIVIDEND, float(received)))
                if spec.reinvest_dividends:
                    buy_cash += received
                    bought, buy_cash = _buy(buy_cash, price, spec.fractional_shares)
                    shares += bought
                else:
                    div_cash += received

        cash = buy_cash + div_cash
        out_shares[i] = shares
        out_cash[i] = cash
        out_invested[i] = invested
        out_dividends[i] = cum_dividends
        out_value[i] = (shares * price if live else 0.0) + cash

    frame = pd.DataFrame(
        {
            'price': close,
            'shares': out_shares,
            'cash': out_cash,
            'invested': out_invested,
            'cum_dividends': out_dividends,
            'value': out_value,
        },
        index=index,
        columns=list(HISTORY_COLUMNS),
    ).astype(float)

    return PortfolioHistory(ticker=prices.ticker, frame=frame, spec=spec,
                            events=tuple(events))


def _buy(cash: float, price: float, fractional: bool) -> tuple[float, float]:
    """Spend `cash` at `price`, returning (shares bought, cash left over)."""
    if cash <= 0 or price <= 0:
        return 0.0, cash
    if fractional:
        return cash / price, 0.0
    quantity = math.floor(cash / price)
    return float(quantity), cash - quantity * price
