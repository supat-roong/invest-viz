"""Tests for mapping a long daily series onto a fixed frame budget.

Everything here is offline: PortfolioHistory objects are built inline or
from the committed CSV fixtures.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invest_viz import timeline
from invest_viz.types import (
    EventKind,
    MarketEvent,
    PortfolioHistory,
    PortfolioSpec,
)

FIXTURES = Path(__file__).parent / 'fixtures'


# --- helpers ----------------------------------------------------------------

def make_history(
    ticker: str,
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    invested: np.ndarray | None = None,
    cum_dividends: np.ndarray | None = None,
    events: tuple = (),
) -> PortfolioHistory:
    """Build a PortfolioHistory directly, without invoking the simulator."""
    n = len(dates)
    values = np.asarray(values, dtype=float)
    if invested is None:
        invested = np.full(n, 10_000.0)
    if cum_dividends is None:
        cum_dividends = np.zeros(n)
    frame = pd.DataFrame(
        {
            'price': values / 100.0,
            'shares': np.full(n, 100.0),
            'cash': np.zeros(n),
            'invested': np.asarray(invested, dtype=float),
            'cum_dividends': np.asarray(cum_dividends, dtype=float),
            'value': values,
        },
        index=dates,
    )
    return PortfolioHistory(
        ticker=ticker, frame=frame, spec=PortfolioSpec(), events=events
    )


def load_fixture(name: str) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / name, parse_dates=['date'])
    return frame.set_index('date')


def history_from_fixture(name: str, ticker: str) -> PortfolioHistory:
    """Turn fixture bars into a plausible buy-and-hold history."""
    bars = load_fixture(name)
    close = bars['close'].astype(float)
    shares = 10_000.0 / float(close.iloc[0])
    values = (close * shares).to_numpy()
    cum_div = (bars['dividends'].astype(float) * shares).cumsum().to_numpy()
    events = tuple(
        MarketEvent(ticker, date, EventKind.SPLIT, float(ratio))
        for date, ratio in bars['splits'].items()
        if float(ratio) != 0.0
    )
    return make_history(
        ticker, bars.index, values, cum_dividends=cum_div, events=events
    )


def synthetic_dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range('2015-01-02', periods=n)


def synthetic_histories(count: int, n: int) -> list:
    dates = synthetic_dates(n)
    rng = np.random.default_rng(7)
    # Monthly DCA, so stats sees a realistic number of cashflows per frame.
    invested = 10_000.0 + 500.0 * (np.arange(n) // 21)
    out = []
    for i in range(count):
        walk = np.cumprod(1.0 + rng.normal(0.0004, 0.012, n))
        out.append(make_history(
            f'T{i}', dates, invested * walk, invested=invested
        ))
    return out


# --- frame_count ------------------------------------------------------------

def test_frame_count_basic():
    assert timeline.frame_count(30, 45) == 1350
    assert timeline.frame_count(60, 10) == 600


def test_frame_count_rounds():
    assert timeline.frame_count(30, 1.5) == 45
    assert timeline.frame_count(30, 0.51) == 15
    assert timeline.frame_count(24, 2.02) == 48


def test_frame_count_minimum_is_one():
    assert timeline.frame_count(30, 0.0) == 1
    assert timeline.frame_count(1, 0.01) == 1
    assert timeline.frame_count(30, -5) == 1


# --- date_for_frame ---------------------------------------------------------

def test_date_for_frame_spans_endpoints():
    dates = synthetic_dates(2516)
    total = timeline.frame_count(30, 45)
    assert timeline.date_for_frame(0, total, dates) == dates[0]
    assert timeline.date_for_frame(total - 1, total, dates) == dates[-1]


def test_date_for_frame_is_monotonic():
    dates = synthetic_dates(2516)
    total = 1350
    seen = [timeline.date_for_frame(i, total, dates) for i in range(total)]
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_date_for_frame_clamps_out_of_range_indices():
    dates = synthetic_dates(100)
    assert timeline.date_for_frame(-4, 50, dates) == dates[0]
    assert timeline.date_for_frame(999, 50, dates) == dates[-1]


def test_ease_slows_the_opening_frames():
    dates = synthetic_dates(2516)
    total = 1350
    positions = np.arange(len(dates))

    def bar_index(i, ease):
        date = timeline.date_for_frame(i, total, dates, ease=ease)
        return positions[dates.get_loc(date)]

    early = [30, 60, 120, 200]
    for i in early:
        assert bar_index(i, 0.25) < bar_index(i, 0.0), f'frame {i}'

    # The ease only borrows time; the endpoints still line up.
    assert timeline.date_for_frame(0, total, dates, ease=0.25) == dates[0]
    assert timeline.date_for_frame(total - 1, total, dates, ease=0.25) == dates[-1]


def test_ease_zero_is_linear():
    dates = synthetic_dates(1000)
    total = 300
    for i in (0, 37, 150, 299):
        expected = dates[int(round(i / (total - 1) * (len(dates) - 1)))]
        assert timeline.date_for_frame(i, total, dates, ease=0.0) == expected


# --- build_states -----------------------------------------------------------

@pytest.fixture(scope='module')
def long_states():
    histories = synthetic_histories(3, 2516)
    return histories, timeline.build_states(histories, fps=30, duration_seconds=45)


def test_build_states_produces_the_frame_budget(long_states):
    histories, states = long_states
    dates = histories[0].dates
    assert len(states) == 1350
    assert states[0].date == dates[0]
    assert states[-1].date == dates[-1]
    assert [s.index for s in states] == list(range(1350))


def test_playhead_advances_less_than_two_bars_per_frame(long_states):
    histories, states = long_states
    dates = histories[0].dates
    steps = [len(s.series[0].dates) for s in states]
    assert steps[0] == 1
    assert steps[-1] == len(dates)
    advance = np.diff(steps)
    assert advance.max() <= 3
    assert abs(advance.mean() - (len(dates) - 1) / 1349) < 0.05


def test_progress_runs_zero_to_one(long_states):
    _, states = long_states
    assert states[0].progress == pytest.approx(0.0)
    assert states[-1].progress == pytest.approx(1.0)
    assert all(0.0 <= s.progress <= 1.0 for s in states)
    assert all(b.progress >= a.progress for a, b in zip(states, states[1:]))


def test_end_hold_appends_exact_frame_count():
    histories = synthetic_histories(2, 400)
    base = timeline.build_states(histories, fps=30, duration_seconds=10)
    held = timeline.build_states(
        histories, fps=30, duration_seconds=10, end_hold_seconds=3
    )
    assert len(held) - len(base) == 90
    final_date = histories[0].dates[-1]
    for state in held[len(base):]:
        assert state.date == final_date
        assert state.progress == 1.0
        assert len(state.series[0].dates) == len(histories[0].dates)
    assert [s.index for s in held] == list(range(len(held)))


def test_end_hold_zero_adds_nothing():
    histories = synthetic_histories(1, 200)
    states = timeline.build_states(
        histories, fps=30, duration_seconds=5, end_hold_seconds=0
    )
    assert len(states) == 150


def test_y_max_is_monotonically_non_decreasing(long_states):
    _, states = long_states
    y_max = [s.y_max for s in states]
    assert all(b >= a for a, b in zip(y_max, y_max[1:]))
    y_min = [s.y_min for s in states]
    assert all(b <= a for a, b in zip(y_min, y_min[1:]))
    assert min(y_min) >= 0.0
    assert all(s.y_max > s.y_min for s in states)


def test_y_range_covers_all_drawn_data(long_states):
    histories, states = long_states
    for state in states[::97]:
        for sl in state.series:
            assert state.y_max >= sl.values.max()
            assert state.y_max >= sl.invested.max()
            assert state.y_max >= sl.cum_dividends.max()
            assert state.y_min <= sl.values.min()


def test_expanding_window_keeps_x_axis_fixed(long_states):
    histories, states = long_states
    dates = histories[0].dates
    assert all(s.x_min == dates[0] for s in states)
    assert all(s.x_max == dates[-1] for s in states)


def test_trailing_window_tracks_the_playhead():
    histories = synthetic_histories(2, 800)
    states = timeline.build_states(
        histories, fps=30, duration_seconds=8, window='trailing'
    )
    spans = {s.x_max - s.x_min for s in states}
    assert len(spans) == 1, 'trailing window span must stay fixed'
    assert states[-1].x_max > states[0].x_max
    assert all(s.x_max >= s.x_min for s in states)


def test_unknown_window_is_rejected():
    histories = synthetic_histories(1, 100)
    with pytest.raises(ValueError, match='window'):
        timeline.build_states(
            histories, fps=30, duration_seconds=2, window='sliding'
        )


def test_empty_histories_is_rejected():
    with pytest.raises(ValueError):
        timeline.build_states([], fps=30, duration_seconds=2)


# --- SeriesSlice ------------------------------------------------------------

def test_slices_are_truncated_at_the_playhead_and_equal_length(long_states):
    histories, states = long_states
    for state in states[::53]:
        assert len(state.series) == len(histories)
        lengths = {len(sl.dates) for sl in state.series}
        assert len(lengths) == 1
        n = lengths.pop()
        assert n >= 1
        for sl in state.series:
            assert len(sl.values) == n
            assert len(sl.invested) == n
            assert len(sl.cum_dividends) == n
            assert pd.Timestamp(sl.dates[-1]) == state.date
            assert pd.Timestamp(sl.dates[0]) == histories[0].dates[0]


def test_slice_values_match_the_source_history(long_states):
    histories, states = long_states
    state = states[600]
    n = len(state.series[0].dates)
    for history, sl in zip(histories, state.series):
        assert sl.ticker == history.ticker
        np.testing.assert_allclose(sl.values, history.value.to_numpy()[:n])
        np.testing.assert_allclose(sl.invested, history.invested.to_numpy()[:n])


def test_events_are_truncated_at_the_playhead():
    history = history_from_fixture('aapl_2020_split.csv', 'AAPL')
    assert history.events, 'fixture should contain at least one split'
    states = timeline.build_states([history], fps=30, duration_seconds=3)
    for state in states:
        for event in state.series[0].events:
            assert event.date <= state.date
    assert len(states[-1].series[0].events) == len(history.events)
    assert len(states[0].series[0].events) == 0


def test_ranks_are_a_permutation_of_one_to_n():
    histories = synthetic_histories(5, 600)
    states = timeline.build_states(histories, fps=30, duration_seconds=6)
    for state in states[::17]:
        ranks = sorted(sl.stats.rank for sl in state.series)
        assert ranks == list(range(1, len(histories) + 1))


def test_stats_are_computed_at_the_playhead():
    histories = [history_from_fixture('ko_3y.csv', 'KO')]
    states = timeline.build_states(histories, fps=30, duration_seconds=5)
    for state in states[::29]:
        sl = state.series[0]
        assert sl.stats.ticker == 'KO'
        assert sl.stats.value == pytest.approx(float(sl.values[-1]))
        assert sl.stats.cum_dividends == pytest.approx(
            float(sl.cum_dividends[-1])
        )


def test_multiple_real_fixtures_build_together():
    histories = [
        history_from_fixture('aapl_3y.csv', 'AAPL'),
        history_from_fixture('msft_3y.csv', 'MSFT'),
        history_from_fixture('ko_3y.csv', 'KO'),
    ]
    states = timeline.build_states(
        histories, fps=30, duration_seconds=15, end_hold_seconds=2, ease_seconds=1
    )
    assert len(states) == 450 + 60
    assert states[0].date == histories[0].dates[0]
    assert states[449].date == histories[0].dates[-1]
    y_max = [s.y_max for s in states]
    assert all(b >= a for a, b in zip(y_max, y_max[1:]))


# --- performance ------------------------------------------------------------

def test_builds_1350_frames_for_5_tickers_quickly():
    histories = synthetic_histories(5, 2516)
    start = time.perf_counter()
    states = timeline.build_states(
        histories, fps=30, duration_seconds=45, end_hold_seconds=3
    )
    elapsed = time.perf_counter() - start
    assert len(states) == 1350 + 90
    assert elapsed < 30.0, f'build_states took {elapsed:.2f}s'
