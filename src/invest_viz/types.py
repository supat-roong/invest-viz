"""Shared data contracts.

Frozen dataclasses only — no logic beyond trivial derived properties. Every
other module imports its interface from here, so this file is the single
source of truth for what flows between pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

# Column names used by every PriceData frame.
CLOSE = 'close'
DIVIDENDS = 'dividends'
SPLITS = 'splits'
PRICE_COLUMNS = (CLOSE, DIVIDENDS, SPLITS)


class DcaFrequency(str, Enum):
    NONE = 'none'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    BIWEEKLY = 'biweekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    YEARLY = 'yearly'


class EventKind(str, Enum):
    SPLIT = 'split'
    DIVIDEND = 'dividend'


@dataclass(frozen=True)
class PriceData:
    """Daily bars for one ticker on the shared calendar.

    `frame` is indexed by tz-naive normalized dates and has exactly the
    columns in PRICE_COLUMNS. `close` is yfinance's unadjusted close, which
    is already split-adjusted; `splits` is therefore annotation only and must
    never be applied to share counts.
    """
    ticker: str
    frame: pd.DataFrame
    currency: str = 'USD'

    @property
    def close(self) -> pd.Series:
        return self.frame[CLOSE]

    @property
    def dividends(self) -> pd.Series:
        return self.frame[DIVIDENDS]

    @property
    def splits(self) -> pd.Series:
        return self.frame[SPLITS]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def first_trading_day(self) -> pd.Timestamp | None:
        live = self.frame[CLOSE].dropna()
        return live.index[0] if len(live) else None


@dataclass(frozen=True)
class PortfolioSpec:
    """How one ticker is invested in. Pure parameters, no I/O."""
    initial_amount: float = 10_000.0
    dca_amount: float = 0.0
    dca_frequency: DcaFrequency = DcaFrequency.NONE
    reinvest_dividends: bool = True
    fractional_shares: bool = True


@dataclass(frozen=True)
class MarketEvent:
    ticker: str
    date: pd.Timestamp
    kind: EventKind
    value: float


@dataclass(frozen=True)
class PortfolioHistory:
    """Per-day simulation output for one ticker.

    `frame` is indexed by date with columns: price, shares, cash, invested,
    cum_dividends, value.
    """
    ticker: str
    frame: pd.DataFrame
    spec: PortfolioSpec
    events: tuple[MarketEvent, ...] = ()

    @property
    def value(self) -> pd.Series:
        return self.frame['value']

    @property
    def invested(self) -> pd.Series:
        return self.frame['invested']

    @property
    def cum_dividends(self) -> pd.Series:
        return self.frame['cum_dividends']

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.frame.index


@dataclass(frozen=True)
class TickerStats:
    """Metrics as of one point in time.

    With contributions arriving over time, CAGR on portfolio value is
    meaningless, so `xirr` (money-weighted return) is the headline figure.
    """
    ticker: str
    value: float
    invested: float
    total_return: float
    xirr: float
    max_drawdown: float
    cum_dividends: float
    rank: int = 0


@dataclass(frozen=True)
class SeriesSlice:
    """One ticker's data truncated at the playhead, ready to draw."""
    ticker: str
    dates: np.ndarray
    values: np.ndarray
    invested: np.ndarray
    cum_dividends: np.ndarray
    stats: TickerStats
    events: tuple[MarketEvent, ...] = ()


@dataclass(frozen=True)
class FrameState:
    """Everything renderer.render_frame needs for exactly one frame."""
    index: int
    date: pd.Timestamp
    progress: float
    series: tuple[SeriesSlice, ...]
    x_min: pd.Timestamp
    x_max: pd.Timestamp
    y_min: float
    y_max: float


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    surface: str
    grid: str
    text: str
    muted: str
    series: Sequence[str]
    positive: str
    negative: str
    font: str = 'DejaVu Sans'
    invested: str = '#8b949e'

    def color_for(self, index: int) -> str:
        return self.series[index % len(self.series)]
