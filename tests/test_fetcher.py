"""fetcher: shape normalisation, retries, stale fallback and alignment.

Every unit test here is offline — `yfinance.download` is monkeypatched with
synthetic frames in the shapes yfinance really returns.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from invest_viz import fetcher
from invest_viz.cache import PriceCache
from invest_viz.fetcher import FetchError, align_calendar, fetch
from invest_viz.types import CLOSE, DIVIDENDS, PRICE_COLUMNS, SPLITS

YF_FIELDS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
             'Dividends', 'Stock Splits']

START = pd.Timestamp('2020-01-01')
END = pd.Timestamp('2020-03-01')


def block(index, close, dividends=None, splits=None, adj_close=None):
    """One ticker's flat yfinance sub-frame."""
    n = len(index)
    close = np.asarray(close, dtype=float)
    adj = np.asarray(adj_close if adj_close is not None else close, dtype=float)
    return pd.DataFrame(
        {
            'Open': close,
            'High': close,
            'Low': close,
            'Close': close,
            'Adj Close': adj,
            'Volume': np.full(n, 1_000.0),
            'Dividends': np.asarray(
                dividends if dividends is not None else np.zeros(n), dtype=float),
            'Stock Splits': np.asarray(
                splits if splits is not None else np.zeros(n), dtype=float),
        },
        index=index,
        columns=YF_FIELDS,
    )


def multi(blocks: dict) -> pd.DataFrame:
    """The MultiIndex (ticker, field) shape of group_by='ticker'."""
    return pd.concat(blocks, axis=1)


def bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods, name='Date')


def price_frame(index, close, dividends=None) -> pd.DataFrame:
    """A ready-made PRICE_COLUMNS frame, as fetcher hands to align_calendar."""
    n = len(index)
    return pd.DataFrame(
        {
            CLOSE: np.asarray(close, dtype=float),
            DIVIDENDS: np.asarray(
                dividends if dividends is not None else np.zeros(n), dtype=float),
            SPLITS: np.zeros(n),
        },
        index=pd.DatetimeIndex(index, name='date'),
    )


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Keep backoff out of the test runtime and record what was asked for."""
    slept: list = []
    monkeypatch.setattr(fetcher, '_sleep', slept.append)
    return slept


def patch_download(monkeypatch, fake):
    calls: list = []

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return fake(*args, **kwargs)

    monkeypatch.setattr(fetcher.yf, 'download', wrapper)
    return calls


# -- shape normalisation ---------------------------------------------------

def test_normalises_multiindex_shape(monkeypatch):
    index = bdays('2020-01-01', 5)
    raw = multi({
        'AAPL': block(index, [10, 11, 12, 13, 14], dividends=[0, 0, 0.5, 0, 0]),
        'MSFT': block(index, [20, 21, 22, 23, 24], splits=[0, 2.0, 0, 0, 0]),
    })
    patch_download(monkeypatch, lambda *a, **k: raw)

    data = fetch(['AAPL', 'MSFT'], START, END)

    assert set(data) == {'AAPL', 'MSFT'}
    for ticker, expected in (('AAPL', 10.0), ('MSFT', 20.0)):
        frame = data[ticker].frame
        assert list(frame.columns) == list(PRICE_COLUMNS)
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.tz is None
        assert (frame.index == frame.index.normalize()).all()
        assert frame[CLOSE].iloc[0] == expected
    assert data['AAPL'].frame[DIVIDENDS].iloc[2] == 0.5
    assert data['MSFT'].frame[SPLITS].iloc[1] == 2.0


def test_normalises_flat_single_ticker_shape(monkeypatch):
    index = bdays('2020-01-01', 4)
    raw = block(index, [10, 11, 12, 13], dividends=[0, 0.25, 0, 0])
    patch_download(monkeypatch, lambda *a, **k: raw)

    data = fetch(['AAPL'], START, END)

    frame = data['AAPL'].frame
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert len(frame) == 4
    assert frame[CLOSE].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert frame[DIVIDENDS].iloc[1] == 0.25


def test_normalises_tz_aware_index(monkeypatch):
    index = pd.date_range('2020-01-01 09:30', periods=3, freq='D',
                          tz='America/New_York', name='Date')
    patch_download(monkeypatch, lambda *a, **k: block(index, [10, 11, 12]))

    frame = fetch(['AAPL'], START, END)['AAPL'].frame

    assert frame.index.tz is None
    assert (frame.index == frame.index.normalize()).all()


def test_uses_close_not_adj_close(monkeypatch):
    index = bdays('2020-01-01', 3)
    raw = multi({'AAPL': block(index, close=[100, 101, 102],
                               adj_close=[1, 2, 3])})
    patch_download(monkeypatch, lambda *a, **k: raw)

    frame = fetch(['AAPL'], START, END)['AAPL'].frame

    assert frame[CLOSE].tolist() == [100.0, 101.0, 102.0]
    assert 'Adj Close' not in frame.columns


# -- cache integration -----------------------------------------------------

def test_cached_tickers_are_not_downloaded(monkeypatch, tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    cache.put('AAPL', price_frame(bdays('2020-01-01', 50), np.arange(50) + 1.0))

    index = bdays('2020-01-01', 50)
    raw = multi({'MSFT': block(index, np.arange(50) + 50.0)})
    calls = patch_download(monkeypatch, lambda *a, **k: raw)

    data = fetch(['AAPL', 'MSFT'], START, END, cache=cache)

    assert set(data) == {'AAPL', 'MSFT'}
    assert len(calls) == 1
    assert calls[0][0][0] == ['MSFT']  # AAPL served from disk


def test_successful_results_are_cached(monkeypatch, tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    index = bdays('2020-01-01', 50)
    patch_download(monkeypatch,
                   lambda *a, **k: multi({'AAPL': block(index, np.arange(50) + 1.0)}))

    fetch(['AAPL'], START, END, cache=cache)

    assert cache.path_for('AAPL').exists()
    assert cache.get('AAPL', START, END) is not None


# -- retries and fallback --------------------------------------------------

def test_retry_then_success(monkeypatch, no_sleeping):
    index = bdays('2020-01-01', 3)
    raw = multi({'AAPL': block(index, [10, 11, 12])})
    attempts = {'n': 0}

    def flaky(*args, **kwargs):
        attempts['n'] += 1
        if attempts['n'] < 3:
            raise RuntimeError('rate limited')
        return raw

    patch_download(monkeypatch, flaky)

    data = fetch(['AAPL'], START, END, max_retries=3)

    assert attempts['n'] == 3
    assert len(data['AAPL'].frame) == 3
    assert no_sleeping == [0.5, 1.0]  # exponential backoff between attempts


def test_retry_exhausted_falls_back_to_stale_with_warning(
        monkeypatch, tmp_path, caplog):
    # max_age_days=0 forces every entry to look stale, so get() always misses.
    cache = PriceCache(tmp_path, max_age_days=0)
    cache.put('AAPL', price_frame(bdays('2020-01-01', 40), np.arange(40) + 1.0))

    def always_fails(*args, **kwargs):
        raise RuntimeError('network down')

    calls = patch_download(monkeypatch, always_fails)

    with caplog.at_level(logging.WARNING, logger='invest_viz.fetcher'):
        data = fetch(['AAPL'], START, END, cache=cache, max_retries=3)

    assert len(calls) == 3
    assert len(data['AAPL'].frame) == 40
    assert data['AAPL'].frame[CLOSE].iloc[0] == 1.0
    assert any('stale' in record.message.lower() for record in caplog.records)


def test_no_stale_copy_raises_fetch_error_naming_every_failure(monkeypatch):
    index = bdays('2020-01-01', 3)
    # Only KO comes back; AAPL and MSFT produce nothing.
    raw = multi({'KO': block(index, [50, 51, 52])})
    patch_download(monkeypatch, lambda *a, **k: raw)

    with pytest.raises(FetchError) as excinfo:
        fetch(['AAPL', 'MSFT', 'KO'], START, END)

    message = str(excinfo.value)
    assert 'AAPL' in message
    assert 'MSFT' in message


def test_fetch_error_after_total_download_failure(monkeypatch, tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)

    def always_fails(*args, **kwargs):
        raise RuntimeError('network down')

    patch_download(monkeypatch, always_fails)

    with pytest.raises(FetchError) as excinfo:
        fetch(['AAPL', 'MSFT'], START, END, cache=cache)

    message = str(excinfo.value)
    assert 'AAPL' in message and 'MSFT' in message


# -- currency --------------------------------------------------------------

def test_mixed_currencies_warn_without_conversion(monkeypatch, caplog):
    index = bdays('2020-01-01', 3)
    raw = multi({
        'AAPL': block(index, [10, 11, 12]),
        'PTT.BK': block(index, [30, 31, 32]),
    })
    patch_download(monkeypatch, lambda *a, **k: raw)

    with caplog.at_level(logging.WARNING, logger='invest_viz.fetcher'):
        data = fetch(['AAPL', 'PTT.BK'], START, END)

    assert data['AAPL'].currency == 'USD'
    assert data['PTT.BK'].currency == 'THB'
    assert data['PTT.BK'].frame[CLOSE].iloc[0] == 30.0  # unconverted
    assert any('currencies' in record.message for record in caplog.records)


def test_single_currency_does_not_warn(monkeypatch, caplog):
    index = bdays('2020-01-01', 3)
    raw = multi({'AAPL': block(index, [10, 11, 12]),
                 'MSFT': block(index, [20, 21, 22])})
    patch_download(monkeypatch, lambda *a, **k: raw)

    with caplog.at_level(logging.WARNING, logger='invest_viz.fetcher'):
        fetch(['AAPL', 'MSFT'], START, END)

    assert not any('currencies' in record.message for record in caplog.records)


# -- align_calendar --------------------------------------------------------

def test_align_calendar_forward_fills_close_but_not_dividends():
    # AAPL trades Jan 1 and Jan 3; KO trades Jan 1, 2 and 3.
    aapl = price_frame([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-03')],
                       close=[10.0, 12.0], dividends=[0.75, 0.0])
    ko = price_frame([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'),
                      pd.Timestamp('2020-01-03')], close=[50.0, 51.0, 52.0])

    aligned = align_calendar({'AAPL': aapl, 'KO': ko})

    union = aligned['AAPL'].index
    assert list(union) == [pd.Timestamp('2020-01-01'),
                           pd.Timestamp('2020-01-02'),
                           pd.Timestamp('2020-01-03')]
    # close carried across AAPL's missing bar
    assert aligned['AAPL'][CLOSE].tolist() == [10.0, 10.0, 12.0]
    # the dividend appears on exactly one date, not forward-filled
    dividends = aligned['AAPL'][DIVIDENDS]
    assert (dividends != 0).sum() == 1
    assert dividends.loc[pd.Timestamp('2020-01-01')] == 0.75
    assert dividends.loc[pd.Timestamp('2020-01-02')] == 0.0
    assert dividends.loc[pd.Timestamp('2020-01-03')] == 0.0
    assert (aligned['AAPL'][SPLITS] == 0.0).all()


def test_align_calendar_leaves_pre_ipo_close_nan():
    old = price_frame(bdays('2020-01-01', 6), np.arange(6) + 10.0)
    new = price_frame(bdays('2020-01-07', 4), [99.0, 98.0, 97.0, 96.0])

    aligned = align_calendar({'OLD': old, 'NEW': new})

    frame = aligned['NEW']
    assert len(frame) == 8
    first = new.index[0]
    assert frame.loc[frame.index < first, CLOSE].isna().all()
    assert not frame.loc[frame.index >= first, CLOSE].isna().any()
    assert (frame[DIVIDENDS] == 0.0).all()
    assert aligned['OLD'][CLOSE].iloc[-1] == 15.0  # last close held to the end


def test_align_calendar_empty_input():
    assert align_calendar({}) == {}


def test_fetch_aligns_mixed_calendars(monkeypatch):
    raw = multi({
        'AAPL': block(bdays('2020-01-01', 5), np.arange(5) + 10.0),
        'PTT.BK': block(bdays('2020-01-03', 5), np.arange(5) + 30.0),
    })
    patch_download(monkeypatch, lambda *a, **k: raw)

    data = fetch(['AAPL', 'PTT.BK'], START, END)

    assert data['AAPL'].dates.equals(data['PTT.BK'].dates)
    assert data['PTT.BK'].first_trading_day == pd.Timestamp('2020-01-03')


# -- integration -----------------------------------------------------------

@pytest.mark.integration
def test_integration_downloads_aapl():
    data = fetch(['AAPL'], '2024-01-02', '2024-02-01')

    frame = data['AAPL'].frame
    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert len(frame) > 10
    assert frame[CLOSE].notna().all()
    assert (frame[CLOSE] > 0).all()
    assert data['AAPL'].currency == 'USD'
