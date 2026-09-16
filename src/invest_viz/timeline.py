"""Map a long daily series onto a fixed frame budget.

A ten-year daily history is ~2516 bars, but a 45s clip at 30fps has only
1350 frames. This module decides which date each frame's playhead sits on,
slices every history at that playhead, and precomputes the axis ranges the
renderer needs. The x-axis is fixed and the y-axis only ever grows, so the
chart never visibly snaps backwards mid-render.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import stats as stats_module
from .types import FrameState, PortfolioHistory, SeriesSlice

#: Fraction of the data range added above and below the drawn values.
Y_PADDING = 0.08

#: Fraction of the full date range shown by the 'trailing' window.
TRAILING_FRACTION = 0.25

WINDOWS = ('expanding', 'trailing')


def frame_count(fps: int, duration_seconds: float) -> int:
    """Number of animated frames for a clip, never fewer than one."""
    return max(1, int(round(float(fps) * float(duration_seconds))))


def date_for_frame(index: int, total: int, dates: pd.DatetimeIndex,
                   ease: float = 0.0) -> pd.Timestamp:
    """Date under the playhead on frame `index` of `total`.

    Frame 0 lands on `dates[0]` and frame `total - 1` on `dates[-1]`.
    `ease` in [0, 1] spends that leading fraction of the animation easing
    in from rest, so the lines do not jerk into motion; 0 is pure linear.
    """
    count = len(dates)
    if count == 0:
        raise ValueError('dates must not be empty')
    if total <= 1:
        return dates[-1]
    fraction = min(1.0, max(0.0, index / (total - 1)))
    position = _eased(fraction, ease) * (count - 1)
    return dates[int(round(position))]


def build_states(histories: list[PortfolioHistory], fps: int,
                 duration_seconds: float, end_hold_seconds: float = 0.0,
                 ease_seconds: float = 0.0,
                 window: str = 'expanding') -> list[FrameState]:
    """One FrameState per rendered frame, plus any end-hold frames.

    Every history is truncated at the playhead, its stats are computed at
    that date and ranked against its peers. With `window='expanding'` the
    x-axis is pinned to the full date range and the y-axis is monotonically
    non-decreasing.
    """
    if not histories:
        raise ValueError('build_states needs at least one history')
    if window not in WINDOWS:
        raise ValueError(f'window must be one of {WINDOWS}, got {window!r}')

    master = _master_dates(histories)
    if len(master) == 0:
        raise ValueError('histories contain no dates')

    total = frame_count(fps, duration_seconds)
    ease = 0.0
    if ease_seconds > 0 and duration_seconds > 0:
        ease = min(1.0, ease_seconds / float(duration_seconds))

    prepared = [_Prepared(history) for history in histories]

    playhead = np.array(
        [date_for_frame(i, total, master, ease).to_datetime64()
         for i in range(total)],
        dtype='datetime64[ns]',
    )

    # Bar counts per history per frame, and the growing y-extent, vectorised.
    counts = [
        np.searchsorted(p.date_values, playhead, side='right') for p in prepared
    ]
    event_counts = [
        np.searchsorted(p.event_dates, playhead, side='right') for p in prepared
    ]
    raw_max, raw_min = _extent(prepared, counts, total)
    y_min, y_max = _pad_extent(raw_min, raw_max)

    x_min_values, x_max_values = _x_axis(window, master, playhead)

    states: list[FrameState] = []
    for i in range(total):
        series = []
        frame_stats = []
        for p, count_array, event_array in zip(prepared, counts, event_counts):
            n = int(count_array[i])
            events = p.events[:int(event_array[i])]
            ticker_stats = stats_module.compute_stats(p.truncate(n, events))
            frame_stats.append(ticker_stats)
            series.append(
                SeriesSlice(
                    ticker=p.ticker,
                    dates=p.date_values[:n],
                    values=p.values[:n],
                    invested=p.invested[:n],
                    cum_dividends=p.cum_dividends[:n],
                    stats=ticker_stats,
                    events=events,
                )
            )
        ranked = {s.ticker: s for s in stats_module.rank_stats(frame_stats)}
        series = tuple(replace(s, stats=ranked[s.ticker]) for s in series)

        states.append(
            FrameState(
                index=i,
                date=pd.Timestamp(playhead[i]),
                progress=(i / (total - 1)) if total > 1 else 1.0,
                series=series,
                x_min=x_min_values[i],
                x_max=x_max_values[i],
                y_min=float(y_min[i]),
                y_max=float(y_max[i]),
            )
        )

    hold = max(0, int(round(float(fps) * float(end_hold_seconds))))
    if hold and states:
        final = states[-1]
        for offset in range(1, hold + 1):
            states.append(
                replace(final, index=final.index + offset, progress=1.0)
            )
    return states


# --- internals --------------------------------------------------------------

class _Prepared:
    """Numpy views of one history, sliced per frame without copying."""

    __slots__ = ('ticker', 'frame', 'spec', 'date_values', 'values',
                 'invested', 'cum_dividends', 'events', 'event_dates',
                 'running_max', 'running_min')

    def __init__(self, history: PortfolioHistory) -> None:
        frame = history.frame
        self.ticker = history.ticker
        self.frame = frame
        self.spec = history.spec
        self.date_values = frame.index.values.astype('datetime64[ns]')
        self.values = frame['value'].to_numpy(dtype=float)
        self.invested = frame['invested'].to_numpy(dtype=float)
        self.cum_dividends = frame['cum_dividends'].to_numpy(dtype=float)

        self.events = tuple(sorted(history.events, key=lambda e: e.date))
        self.event_dates = np.array(
            [pd.Timestamp(e.date).to_datetime64() for e in self.events],
            dtype='datetime64[ns]',
        )

        drawn = np.vstack([self.values, self.invested, self.cum_dividends])
        self.running_max = np.maximum.accumulate(np.nanmax(drawn, axis=0))
        self.running_min = np.minimum.accumulate(np.nanmin(drawn, axis=0))

    def truncate(self, n: int, events: tuple) -> PortfolioHistory:
        """A history backed by a *view* of the first `n` bars — no copy."""
        return PortfolioHistory(
            ticker=self.ticker,
            frame=self.frame.iloc[:n],
            spec=self.spec,
            events=events,
        )


def _master_dates(histories: list[PortfolioHistory]) -> pd.DatetimeIndex:
    index = histories[0].frame.index
    for history in histories[1:]:
        if not index.equals(history.frame.index):
            index = index.union(history.frame.index)
    return pd.DatetimeIndex(index).sort_values()


def _eased(fraction: float, ease: float) -> float:
    """Remap [0, 1] so the leading `ease` fraction accelerates from rest.

    Speed follows a smoothstep ramp over the ease window and is flat after
    it, then the whole curve is renormalised so f(0) = 0 and f(1) = 1.
    """
    ease = min(1.0, max(0.0, float(ease)))
    if ease <= 0.0:
        return fraction
    scale = 1.0 / (1.0 - ease / 2.0)
    if fraction >= ease:
        return min(1.0, scale * (fraction - ease / 2.0))
    u = fraction / ease
    return scale * ease * (u ** 3 - 0.5 * u ** 4)


def _extent(prepared: list, counts: list, total: int) -> tuple:
    raw_max = np.full(total, -np.inf)
    raw_min = np.full(total, np.inf)
    for p, count_array in zip(prepared, counts):
        live = count_array > 0
        if not live.any():
            continue
        last = np.clip(count_array - 1, 0, len(p.running_max) - 1)
        raw_max = np.where(
            live, np.maximum(raw_max, p.running_max[last]), raw_max
        )
        raw_min = np.where(
            live, np.minimum(raw_min, p.running_min[last]), raw_min
        )
    raw_max = np.where(np.isfinite(raw_max), raw_max, 1.0)
    raw_min = np.where(np.isfinite(raw_min), raw_min, 0.0)
    return raw_max, raw_min


def _pad_extent(raw_min: np.ndarray, raw_max: np.ndarray) -> tuple:
    span = raw_max - raw_min
    pad = np.where(
        span > 0,
        span * Y_PADDING,
        np.maximum(np.abs(raw_max) * Y_PADDING, 1.0),
    )
    y_max = np.maximum.accumulate(raw_max + pad)
    y_min = np.minimum.accumulate(np.maximum(0.0, raw_min - pad))
    y_max = np.maximum.accumulate(np.maximum(y_max, y_min + 1.0))
    return y_min, y_max


def _x_axis(window: str, master: pd.DatetimeIndex,
            playhead: np.ndarray) -> tuple:
    first = master[0]
    last = master[-1]
    if window == 'expanding':
        return [first] * len(playhead), [last] * len(playhead)

    span = (last - first) * TRAILING_FRACTION
    if span <= pd.Timedelta(0):
        span = pd.Timedelta(days=1)
    x_max = [max(pd.Timestamp(d), first + span) for d in playhead]
    x_min = [d - span for d in x_max]
    return x_min, x_max
