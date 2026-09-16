"""On-disk parquet cache for daily price frames.

Layout
------
    <cache_dir>/<encoded-ticker>.parquet   one file per ticker
    <cache_dir>/index.json                 sidecar metadata

Each parquet holds the *widest* date range ever fetched for that ticker, so
re-rendering the same comparison with a different theme never re-fetches.
`index.json` maps the raw ticker to `{fetched_at, start, end, rows}`;
`fetched_at` is the ISO-8601 UTC time of the last `put`, and drives expiry.
Metadata lives in a sidecar rather than in parquet key-value metadata so the
whole cache can be inspected without opening every file.

Ticker names are percent-encoded for the filesystem, so `^GSPC`, `THB=X`,
`BTC-USD` and `PTT.BK` all round-trip. Note the encoding preserves case, so
on a case-insensitive filesystem `aapl` and `AAPL` share one entry; callers
are expected to normalise ticker case (config.py upper-cases them).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd

from .types import CLOSE, DIVIDENDS, PRICE_COLUMNS, SPLITS

log = logging.getLogger(__name__)

INDEX_NAME = 'index.json'

# Requested ranges rarely land on trading days: `2015-01-01` is a holiday and
# yfinance treats `end` as exclusive, so cached data legitimately starts a few
# days after `start` and ends a few days before `end`. Without this slack the
# cache would miss on every single request. Genuinely partial coverage is off
# by far more than this.
COVERAGE_SLACK = timedelta(days=5)


def encode_ticker(ticker: str) -> str:
    """Return a filesystem-safe stem for `ticker`."""
    return quote(ticker, safe='')


def decode_ticker(stem: str) -> str:
    """Inverse of `encode_ticker`."""
    return unquote(stem)


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce to a tz-naive, normalised, unique, sorted PRICE_COLUMNS frame."""
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(None)
    out = pd.DataFrame(index=index.normalize())
    out.index.name = 'date'
    for column in PRICE_COLUMNS:
        if column in frame.columns:
            values = frame[column]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            out[column] = pd.to_numeric(values.to_numpy(), errors='coerce')
        else:
            out[column] = float('nan') if column == CLOSE else 0.0
    out[DIVIDENDS] = out[DIVIDENDS].fillna(0.0)
    out[SPLITS] = out[SPLITS].fillna(0.0)
    out = out[~out.index.duplicated(keep='last')]
    return out.sort_index()


class PriceCache:
    """Widest-range parquet cache with age-based expiry and a stale fallback."""

    def __init__(self, cache_dir: Path, max_age_days: int = 1) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age_days = max_age_days

    # -- paths / metadata -------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.cache_dir / INDEX_NAME

    def path_for(self, ticker: str) -> Path:
        return self.cache_dir / f'{encode_ticker(ticker)}.parquet'

    def _read_index(self) -> dict:
        try:
            with open(self.index_path, 'rb') as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, index: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
        tmp.replace(self.index_path)

    # -- reads ------------------------------------------------------------

    def _load(self, ticker: str) -> pd.DataFrame | None:
        """Read a ticker's parquet, treating any unreadable file as a miss."""
        path = self.path_for(ticker)
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # corrupt/truncated file is a miss, not a crash
            log.warning('cache: unreadable parquet for %s (%s); treating as miss',
                        ticker, exc)
            return None
        if frame is None or frame.empty:
            return None
        try:
            return _normalise(frame)
        except Exception as exc:
            log.warning('cache: unusable frame for %s (%s); treating as miss',
                        ticker, exc)
            return None

    def _fetched_at(self, ticker: str) -> datetime | None:
        entry = self._read_index().get(ticker)
        if not isinstance(entry, dict):
            return None
        try:
            stamp = datetime.fromisoformat(entry['fetched_at'])
        except (KeyError, TypeError, ValueError):
            return None
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)

    def is_fresh(self, ticker: str) -> bool:
        """True when the entry exists and is younger than `max_age_days`."""
        if self.max_age_days <= 0:
            return False
        stamp = self._fetched_at(ticker)
        if stamp is None:
            return False
        age = datetime.now(timezone.utc) - stamp
        return age <= timedelta(days=self.max_age_days)

    def get(self, ticker: str, start: pd.Timestamp,
            end: pd.Timestamp) -> pd.DataFrame | None:
        """Return the `[start, end]` slice, or None on miss/partial/expired."""
        if not self.is_fresh(ticker):
            return None
        frame = self._load(ticker)
        if frame is None or frame.empty:
            return None
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        if frame.index[0] > start + COVERAGE_SLACK:
            return None
        if frame.index[-1] < end - COVERAGE_SLACK:
            return None
        return frame.loc[start:end].copy()

    def get_stale(self, ticker: str) -> pd.DataFrame | None:
        """Return the whole cached frame ignoring age; the network fallback."""
        return self._load(ticker)

    # -- writes -----------------------------------------------------------

    def put(self, ticker: str, frame: pd.DataFrame) -> None:
        """Merge `frame` into the cached entry and refresh `fetched_at`.

        Rows are merged by date with the new frame winning on overlap, so the
        stored range only ever widens.
        """
        incoming = _normalise(frame)
        existing = self._load(ticker)
        if existing is not None and not existing.empty:
            kept = existing.loc[~existing.index.isin(incoming.index)]
            merged = pd.concat([kept, incoming]).sort_index()
        else:
            merged = incoming
        if merged.empty:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path_for(ticker))

        index = self._read_index()
        index[ticker] = {
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'start': merged.index[0].isoformat(),
            'end': merged.index[-1].isoformat(),
            'rows': int(len(merged)),
        }
        self._write_index(index)
