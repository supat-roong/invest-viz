"""Bulk price fetching, cache integration and calendar alignment.

The only network-touching module in the pipeline. One `yf.download` call
covers every ticker the cache cannot serve; failures retry with exponential
backoff and then fall back to a stale cache entry with a warning.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf

from .cache import PriceCache
from .types import CLOSE, DIVIDENDS, PRICE_COLUMNS, SPLITS, PriceData

log = logging.getLogger(__name__)

DEFAULT_CURRENCY = 'USD'

# yfinance column names we source PRICE_COLUMNS from. `Close` is deliberate:
# with auto_adjust=False it is split-adjusted but NOT dividend-adjusted, which
# is what a real portfolio simulation needs. `Adj Close` is never used.
SOURCE_COLUMNS = {CLOSE: 'Close', DIVIDENDS: 'Dividends', SPLITS: 'Stock Splits'}

# Cheap offline currency inference from the Yahoo suffix. No network call is
# made for this; anything unrecognised is assumed to be USD.
SUFFIX_CURRENCIES = {
    '.BK': 'THB', '.T': 'JPY', '.KS': 'KRW', '.KQ': 'KRW', '.HK': 'HKD',
    '.SS': 'CNY', '.SZ': 'CNY', '.TW': 'TWD', '.L': 'GBP', '.PA': 'EUR',
    '.DE': 'EUR', '.F': 'EUR', '.AS': 'EUR', '.MI': 'EUR', '.MC': 'EUR',
    '.BR': 'EUR', '.LS': 'EUR', '.HE': 'EUR', '.VI': 'EUR', '.IR': 'EUR',
    '.SW': 'CHF', '.ST': 'SEK', '.OL': 'NOK', '.CO': 'DKK', '.TO': 'CAD',
    '.V': 'CAD', '.AX': 'AUD', '.NZ': 'NZD', '.NS': 'INR', '.BO': 'INR',
    '.SA': 'BRL', '.MX': 'MXN', '.JO': 'ZAR', '.TA': 'ILS', '.IS': 'TRY',
}


class FetchError(Exception):
    """One or more requested tickers produced no usable data."""


def _sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch the backoff away."""
    time.sleep(seconds)


def _backoff(attempt: int) -> float:
    """0.5s, 1s, 2s, ... for attempt 0, 1, 2."""
    return 0.5 * (2 ** attempt)


def currency_for(ticker: str) -> str:
    """Best-effort currency for a ticker, without touching the network."""
    name = ticker.upper()
    if name.endswith('=X'):
        # FX pairs are quoted in the second currency: THB=X is USD/THB.
        base = name[:-2]
        return base[3:] if len(base) == 6 else base or DEFAULT_CURRENCY
    if '-' in name:  # crypto pairs: BTC-USD, ETH-EUR
        return name.rsplit('-', 1)[1]
    for suffix, currency in SUFFIX_CURRENCIES.items():
        if name.endswith(suffix):
            return currency
    return DEFAULT_CURRENCY


def _series(sub: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in sub.columns:
        return None
    values = sub[name]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.Series(pd.to_numeric(values.to_numpy(), errors='coerce'))


def _ticker_columns(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker's sub-frame out of whatever shape yfinance returned."""
    columns = raw.columns
    if isinstance(columns, pd.MultiIndex):
        if ticker in columns.get_level_values(0):
            return raw[ticker]
        if ticker in columns.get_level_values(1):
            return raw.xs(ticker, axis=1, level=1)
        return None
    # Flat columns: yfinance drops the ticker level for a single ticker.
    return raw


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Return a PRICE_COLUMNS frame on a tz-naive normalised index."""
    sub = _ticker_columns(raw, ticker)
    if sub is None or sub.empty:
        return None

    index = pd.DatetimeIndex(sub.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    index = index.normalize()

    close = _series(sub, SOURCE_COLUMNS[CLOSE])
    if close is None:
        return None

    frame = pd.DataFrame(index=index)
    frame.index.name = 'date'
    frame[CLOSE] = close.to_numpy()
    for column in (DIVIDENDS, SPLITS):
        values = _series(sub, SOURCE_COLUMNS[column])
        frame[column] = 0.0 if values is None else values.to_numpy()
        frame[column] = frame[column].fillna(0.0)

    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    # Drop the non-trading days a bulk download pads in for other exchanges;
    # align_calendar rebuilds the union afterwards.
    frame = frame[frame[CLOSE].notna()]
    return frame[list(PRICE_COLUMNS)] if not frame.empty else None


def _download(tickers: Sequence[str], start: pd.Timestamp,
              end: pd.Timestamp, max_retries: int) -> pd.DataFrame | None:
    """One bulk download, retried with exponential backoff."""
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            return yf.download(
                list(tickers),
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                group_by='ticker',
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.warning('yfinance download failed (attempt %d/%d): %s',
                        attempt + 1, attempts, exc)
            if attempt + 1 < attempts:
                _sleep(_backoff(attempt))
    return None


def align_calendar(frames: dict) -> dict:
    """Reindex every frame onto the union calendar.

    `close` is forward-filled so a ticker closed for a local holiday keeps its
    last price, but it stays NaN before that ticker's own first trading day so
    portfolio.py can tell "not listed yet" from "market closed".
    `dividends` and `splits` fill with 0.0 — forward-filling a dividend would
    pay it out again on every subsequent bar.
    """
    if not frames:
        return {}

    union = None
    for frame in frames.values():
        index = pd.DatetimeIndex(frame.index)
        union = index if union is None else union.union(index)
    union = pd.DatetimeIndex(union).sort_values()

    aligned = {}
    for ticker, frame in frames.items():
        wide = frame.reindex(union)
        # ffill never reaches back before the first valid value, so pre-listing
        # bars stay NaN by construction.
        wide[CLOSE] = wide[CLOSE].ffill()
        wide[DIVIDENDS] = wide[DIVIDENDS].fillna(0.0)
        wide[SPLITS] = wide[SPLITS].fillna(0.0)
        wide.index.name = 'date'
        aligned[ticker] = wide[list(PRICE_COLUMNS)]
    return aligned


def fetch(tickers: Iterable[str], start, end, cache: PriceCache = None,
          max_retries: int = 3) -> dict:
    """Return `{ticker: PriceData}` on a shared calendar.

    Cache-served tickers are excluded from the network call. Tickers the
    network cannot supply fall back to a stale cache entry with a warning;
    those with no stale copy either are collected and reported together in a
    single FetchError.
    """
    wanted = list(dict.fromkeys(tickers))
    if not wanted:
        raise FetchError('no tickers requested')
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    frames: dict = {}
    missing: list = []
    for ticker in wanted:
        cached = cache.get(ticker, start, end) if cache is not None else None
        if cached is not None and not cached.empty:
            frames[ticker] = cached
        else:
            missing.append(ticker)

    raw = _download(missing, start, end, max_retries) if missing else None

    failed: list = []
    for ticker in missing:
        frame = None
        if raw is not None and not raw.empty:
            try:
                frame = _normalise(raw, ticker)
            except Exception as exc:
                log.warning('could not normalise %s: %s', ticker, exc)
                frame = None
        if frame is not None and not frame.empty:
            frames[ticker] = frame
            if cache is not None:
                cache.put(ticker, frame)
            continue

        stale = cache.get_stale(ticker) if cache is not None else None
        if stale is not None and not stale.empty:
            log.warning('%s: download failed, using stale cached data '
                        '(%s to %s)', ticker, stale.index[0].date(),
                        stale.index[-1].date())
            frames[ticker] = stale.loc[start:end].copy()
        else:
            failed.append(ticker)

    frames = {t: f for t, f in frames.items() if f is not None and not f.empty}
    failed.extend(t for t in wanted if t not in frames and t not in failed)
    if failed:
        raise FetchError('no usable data for: ' + ', '.join(sorted(failed)))

    aligned = align_calendar(frames)
    currencies = {t: currency_for(t) for t in aligned}
    distinct = sorted(set(currencies.values()))
    if len(distinct) > 1:
        log.warning('tickers span multiple currencies (%s); no FX conversion '
                    'is applied, so values are not directly comparable',
                    ', '.join(distinct))

    return {t: PriceData(ticker=t, frame=f, currency=currencies[t])
            for t, f in aligned.items()}
