"""PriceCache: round-trip, expiry, widening merges and corruption tolerance."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from invest_viz.cache import PriceCache, decode_ticker, encode_ticker
from invest_viz.types import CLOSE, DIVIDENDS, PRICE_COLUMNS, SPLITS

EXOTIC = ['^GSPC', 'BTC-USD', 'THB=X', 'PTT.BK', 'AAPL']


def make_frame(start: str, periods: int, price: float = 100.0) -> pd.DataFrame:
    """A small PRICE_COLUMNS frame of business days."""
    index = pd.bdate_range(start, periods=periods, name='date')
    return pd.DataFrame(
        {
            CLOSE: [price + i for i in range(periods)],
            DIVIDENDS: [0.0] * periods,
            SPLITS: [0.0] * periods,
        },
        index=index,
    )


def age_entry(cache: PriceCache, ticker: str, days: float) -> None:
    """Rewrite `fetched_at` so the entry looks `days` old."""
    index = json.loads(cache.index_path.read_text())
    stamp = datetime.now(timezone.utc) - timedelta(days=days)
    index[ticker]['fetched_at'] = stamp.isoformat()
    cache.index_path.write_text(json.dumps(index))


def test_round_trip(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    frame = make_frame('2020-01-01', 20)
    cache.put('AAPL', frame)

    got = cache.get('AAPL', frame.index[0], frame.index[-1])

    assert got is not None
    assert list(got.columns) == list(PRICE_COLUMNS)
    pd.testing.assert_frame_equal(got, frame, check_freq=False)


def test_miss_when_never_cached(tmp_path):
    cache = PriceCache(tmp_path)
    assert cache.get('AAPL', pd.Timestamp('2020-01-01'),
                     pd.Timestamp('2020-02-01')) is None
    assert cache.get_stale('AAPL') is None


@pytest.mark.parametrize('ticker', EXOTIC)
def test_exotic_ticker_names_round_trip(tmp_path, ticker):
    assert decode_ticker(encode_ticker(ticker)) == ticker

    cache = PriceCache(tmp_path, max_age_days=1)
    frame = make_frame('2021-03-01', 10)
    cache.put(ticker, frame)

    path = cache.path_for(ticker)
    assert path.exists()
    assert path.parent == tmp_path

    got = cache.get(ticker, frame.index[0], frame.index[-1])
    assert got is not None
    assert len(got) == 10


def test_exotic_tickers_do_not_collide(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    for i, ticker in enumerate(EXOTIC):
        cache.put(ticker, make_frame('2021-03-01', 10, price=100.0 + i))
    for i, ticker in enumerate(EXOTIC):
        got = cache.get(ticker, pd.Timestamp('2021-03-01'),
                        pd.Timestamp('2021-03-10'))
        assert got is not None
        assert got[CLOSE].iloc[0] == 100.0 + i


def test_miss_on_partial_range_coverage(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    cache.put('AAPL', make_frame('2020-01-01', 40))

    # Wants data starting a year before anything cached.
    assert cache.get('AAPL', pd.Timestamp('2019-01-01'),
                     pd.Timestamp('2020-02-01')) is None
    # Wants data running a year past the cached end.
    assert cache.get('AAPL', pd.Timestamp('2020-01-02'),
                     pd.Timestamp('2021-01-01')) is None
    # Fully covered: a hit.
    assert cache.get('AAPL', pd.Timestamp('2020-01-06'),
                     pd.Timestamp('2020-01-10')) is not None


def test_slice_is_limited_to_requested_range(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    cache.put('AAPL', make_frame('2020-01-01', 40))

    got = cache.get('AAPL', pd.Timestamp('2020-01-06'),
                    pd.Timestamp('2020-01-10'))

    assert got.index[0] == pd.Timestamp('2020-01-06')
    assert got.index[-1] == pd.Timestamp('2020-01-10')


def test_miss_when_expired(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    frame = make_frame('2020-01-01', 20)
    cache.put('AAPL', frame)
    assert cache.get('AAPL', frame.index[0], frame.index[-1]) is not None

    age_entry(cache, 'AAPL', days=3)

    assert cache.get('AAPL', frame.index[0], frame.index[-1]) is None


def test_zero_max_age_is_always_stale(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=0)
    frame = make_frame('2020-01-01', 20)
    cache.put('AAPL', frame)

    assert cache.get('AAPL', frame.index[0], frame.index[-1]) is None
    assert cache.get_stale('AAPL') is not None


def test_get_stale_returns_expired_entry(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    frame = make_frame('2020-01-01', 20)
    cache.put('AAPL', frame)
    age_entry(cache, 'AAPL', days=99)

    assert cache.get('AAPL', frame.index[0], frame.index[-1]) is None

    stale = cache.get_stale('AAPL')
    assert stale is not None
    pd.testing.assert_frame_equal(stale, frame, check_freq=False)


def test_put_widens_range_and_never_narrows(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    cache.put('AAPL', make_frame('2020-06-01', 20))
    wide = cache.get_stale('AAPL')

    # An earlier window and a later window both widen the stored range.
    cache.put('AAPL', make_frame('2020-01-01', 20))
    cache.put('AAPL', make_frame('2020-12-01', 20))
    widened = cache.get_stale('AAPL')

    assert widened.index[0] == pd.Timestamp('2020-01-01')
    assert widened.index[-1] > wide.index[-1]
    assert len(widened) == 60

    # A narrow put in the middle must not shrink what is stored.
    cache.put('AAPL', make_frame('2020-06-01', 3))
    after = cache.get_stale('AAPL')

    assert after.index[0] == widened.index[0]
    assert after.index[-1] == widened.index[-1]
    assert len(after) == len(widened)


def test_put_lets_new_data_win_on_overlap(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    cache.put('AAPL', make_frame('2020-01-01', 10, price=100.0))
    cache.put('AAPL', make_frame('2020-01-01', 5, price=500.0))

    stored = cache.get_stale('AAPL')

    assert len(stored) == 10
    assert stored[CLOSE].iloc[0] == 500.0
    assert stored[CLOSE].iloc[4] == 504.0
    assert stored[CLOSE].iloc[5] == 105.0  # untouched older tail


def test_put_refreshes_fetched_at(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    cache.put('AAPL', make_frame('2020-01-01', 10))
    age_entry(cache, 'AAPL', days=99)
    assert cache.get('AAPL', pd.Timestamp('2020-01-01'),
                     pd.Timestamp('2020-01-10')) is None

    cache.put('AAPL', make_frame('2020-01-01', 10))

    assert cache.get('AAPL', pd.Timestamp('2020-01-01'),
                     pd.Timestamp('2020-01-10')) is not None


def test_corrupt_parquet_is_a_miss_not_an_exception(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    frame = make_frame('2020-01-01', 20)
    cache.put('AAPL', frame)

    cache.path_for('AAPL').write_bytes(b'not a parquet file at all')

    assert cache.get('AAPL', frame.index[0], frame.index[-1]) is None
    assert cache.get_stale('AAPL') is None


def test_corrupt_index_json_is_a_miss(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    frame = make_frame('2020-01-01', 20)
    cache.put('AAPL', frame)

    cache.index_path.write_text('{ this is not json')

    assert cache.get('AAPL', frame.index[0], frame.index[-1]) is None
    # The parquet itself is still readable, so the stale fallback survives.
    assert cache.get_stale('AAPL') is not None


def test_put_normalises_tz_aware_index(tmp_path):
    cache = PriceCache(tmp_path, max_age_days=1)
    index = pd.date_range('2020-01-01 14:30', periods=5, freq='D',
                          tz='America/New_York')
    frame = pd.DataFrame(
        {CLOSE: [1.0, 2.0, 3.0, 4.0, 5.0], DIVIDENDS: 0.0, SPLITS: 0.0},
        index=index,
    )

    cache.put('AAPL', frame)
    stored = cache.get_stale('AAPL')

    assert stored.index.tz is None
    assert (stored.index == stored.index.normalize()).all()
