"""Shared test helpers: offline fixture loading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from invest_viz.types import PRICE_COLUMNS, PriceData

FIXTURE_DIR = Path(__file__).parent / 'fixtures'


def load_prices(name: str, ticker: str = None) -> PriceData:
    """Read tests/fixtures/<name>.csv into a PriceData on a tz-naive index."""
    frame = pd.read_csv(FIXTURE_DIR / f'{name}.csv', parse_dates=['date'])
    frame = frame.set_index('date')
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    frame.index = index.normalize()
    frame.index.name = 'date'
    return PriceData(ticker=ticker or name.split('_')[0].upper(),
                     frame=frame[list(PRICE_COLUMNS)].astype(float))


@pytest.fixture
def prices():
    """The fixture loader itself, so a test can pick which CSV it needs."""
    return load_prices
