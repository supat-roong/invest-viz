"""Renderer tests: offline, no ffmpeg, no portfolio/timeline imports.

Every FrameState here is built by hand from numpy arrays so these tests stay
independent of the simulation modules.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from investment_video.config import ASPECT_RESOLUTIONS, PanelsConfig
from investment_video.renderer import (ChartRenderer, format_money,
                                       format_percent, resolve_label_positions)
from investment_video.themes import get_theme, list_themes
from investment_video.types import (EventKind, FrameState, MarketEvent,
                                    SeriesSlice, TickerStats)

LANDSCAPE = ASPECT_RESOLUTIONS['16:9']
PORTRAIT = ASPECT_RESOLUTIONS['9:16']


def make_series(ticker: str, bars: int = 400, growth: float = 1.4,
                final_value: float = None, rank: int = 1,
                xirr: float = 0.12, events: bool = False,
                seed: int = 0) -> SeriesSlice:
    """One plausible ticker slice: rising value, staircase contributions."""
    dates = pd.date_range('2015-01-02', periods=bars, freq='B')
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.006, bars))
    curve = np.linspace(1.0, growth, bars) * np.exp(walk)
    invested = 10_000.0 + 500.0 * np.floor(np.arange(bars) / 21.0)
    values = invested * curve
    if final_value is not None and values[-1] > 0:
        values = values * (final_value / values[-1])
    dividends = np.cumsum(np.where(np.arange(bars) % 63 == 62, 90.0, 0.0))

    marks: tuple = ()
    if events:
        marks = tuple(
            MarketEvent(ticker=ticker, date=dates[index],
                        kind=EventKind.DIVIDEND if step % 2 else EventKind.SPLIT,
                        value=0.22)
            for step, index in enumerate(range(60, bars, 70)))

    stats = TickerStats(ticker=ticker, value=float(values[-1]),
                        invested=float(invested[-1]),
                        total_return=float(values[-1] / invested[-1] - 1.0),
                        xirr=xirr, max_drawdown=-0.184,
                        cum_dividends=float(dividends[-1]), rank=rank)
    return SeriesSlice(ticker=ticker,
                       dates=dates.values.astype('datetime64[ns]'),
                       values=values, invested=invested,
                       cum_dividends=dividends, stats=stats, events=marks)


def make_state(series: list, index: int = 0) -> FrameState:
    values = np.concatenate([s.values for s in series if len(s.values)])
    dates = series[0].dates
    return FrameState(
        index=index,
        date=pd.Timestamp(dates[-1]),
        progress=1.0,
        series=tuple(series),
        x_min=pd.Timestamp(dates[0]),
        x_max=pd.Timestamp(dates[-1]),
        y_min=0.0,
        y_max=float(values.max()) * 1.08,
    )


def default_state(tickers=('AAPL', 'MSFT', 'KO'), **kwargs) -> FrameState:
    series = [make_series(ticker, growth=1.2 + 0.25 * i, rank=i + 1,
                          xirr=0.15 - 0.03 * i, seed=i, **kwargs)
              for i, ticker in enumerate(tickers)]
    return make_state(series)


@pytest.fixture
def panels():
    return PanelsConfig()


def png_size(path) -> tuple:
    with Image.open(path) as image:
        return image.size


# --- dimensions -------------------------------------------------------------

@pytest.mark.parametrize('resolution', [LANDSCAPE, PORTRAIT])
def test_render_writes_png_of_exact_resolution(tmp_path, panels, resolution):
    renderer = ChartRenderer(get_theme('midnight'), panels, resolution)
    try:
        out = renderer.render(default_state(), tmp_path / 'frame.png')
    finally:
        renderer.close()
    assert out.exists()
    assert png_size(out) == resolution


def test_render_returns_the_path_and_creates_parents(tmp_path, panels):
    renderer = ChartRenderer(get_theme('paper'), panels, LANDSCAPE)
    try:
        target = tmp_path / 'nested' / 'deeper' / 'frame.png'
        assert renderer.render(default_state(), target) == target
        assert target.exists()
    finally:
        renderer.close()


# --- performance ------------------------------------------------------------

def test_reuses_the_figure_across_frames(tmp_path, panels):
    """40 frames must average well under a quarter second each.

    A per-frame figure rebuild blows this budget by an order of magnitude.
    On a saturated host wall clock measures the host rather than the
    renderer, so the same budget is then applied to CPU time, which a
    rebuild inflates just as surely.
    """
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    state = default_state()
    frames = 40
    try:
        renderer.render(state, tmp_path / 'warmup.png')  # first-draw font cache
        figure_id = id(renderer.fig)
        axes_before = len(renderer.fig.axes)
        lines_before = sum(len(ax.lines) for ax in renderer.fig.axes)

        start, cpu_start = time.perf_counter(), time.process_time()
        for index in range(frames):
            renderer.render(state, tmp_path / f'f{index:03d}.png')
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start

        assert id(renderer.fig) == figure_id
        assert len(renderer.fig.axes) == axes_before
        assert sum(len(ax.lines) for ax in renderer.fig.axes) == lines_before
    finally:
        renderer.close()

    mean_ms = elapsed / frames * 1000.0
    cpu_ms = cpu_elapsed / frames * 1000.0
    load = _load_average()
    if load is not None and load > 1.5 * (os.cpu_count() or 1):
        assert cpu_ms < 250.0, (f'{cpu_ms:.1f}ms of cpu per frame at load '
                                f'{load:.1f} (wall {mean_ms:.1f}ms)')
    else:
        assert mean_ms < 250.0, f'{mean_ms:.1f}ms per frame'


def _load_average():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):  # pragma: no cover - platform specific
        return None


# --- panels -----------------------------------------------------------------

def test_stats_panel_on_and_off_both_render_and_differ(tmp_path):
    state = default_state()
    outputs = {}
    for flag in (False, True):
        renderer = ChartRenderer(get_theme('midnight'),
                                 PanelsConfig(stats=flag), LANDSCAPE)
        try:
            path = renderer.render(state, tmp_path / f'stats_{flag}.png')
        finally:
            renderer.close()
        assert png_size(path) == LANDSCAPE
        outputs[flag] = path.read_bytes()
    assert outputs[False] != outputs[True]


def test_all_panels_disabled_still_renders(tmp_path):
    panels = PanelsConfig(stats=False, invested_line=False,
                          dividend_line=False, event_markers=False)
    renderer = ChartRenderer(get_theme('neon'), panels, LANDSCAPE)
    try:
        path = renderer.render(default_state(), tmp_path / 'bare.png')
    finally:
        renderer.close()
    assert png_size(path) == LANDSCAPE


def test_event_markers_render(tmp_path, panels):
    state = default_state(events=True)
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    try:
        path = renderer.render(state, tmp_path / 'events.png')
    finally:
        renderer.close()
    assert png_size(path) == LANDSCAPE


def test_eight_tickers_render(tmp_path, panels):
    tickers = ('AAPL', 'MSFT', 'KO', 'VOO', 'TSLA', 'AMZN', 'NVDA', 'JNJ')
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    try:
        path = renderer.render(default_state(tickers), tmp_path / 'eight.png')
    finally:
        renderer.close()
    assert png_size(path) == LANDSCAPE


def test_empty_slice_does_not_raise(tmp_path, panels):
    """A ticker that has not IPO'd yet contributes no points."""
    live = make_series('AAPL')
    empty = SeriesSlice(
        ticker='LATE', dates=np.empty(0, dtype='datetime64[ns]'),
        values=np.empty(0), invested=np.empty(0),
        cum_dividends=np.empty(0),
        stats=TickerStats(ticker='LATE', value=0.0, invested=0.0,
                          total_return=0.0, xirr=float('nan'),
                          max_drawdown=0.0, cum_dividends=0.0, rank=2))
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    try:
        path = renderer.render(make_state([live, empty]), tmp_path / 'late.png')
    finally:
        renderer.close()
    assert png_size(path) == LANDSCAPE


# --- head labels ------------------------------------------------------------

def test_head_labels_do_not_overlap(tmp_path, panels):
    """Two near-identical finishes must still get two readable labels."""
    a = make_series('AAA', final_value=100_000.0, rank=1, seed=1)
    b = make_series('BBB', final_value=100_050.0, rank=2, seed=2)
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    try:
        renderer.render(make_state([a, b]), tmp_path / 'labels.png')
        positions = renderer.label_positions
        gap = abs(positions['AAA'] - positions['BBB'])
        assert gap >= renderer.min_label_gap - 1e-6
    finally:
        renderer.close()


def test_resolve_label_positions_preserves_order_and_gap():
    ys = [100.0, 101.0, 102.5, 40.0, 300.0]
    resolved = resolve_label_positions(ys, 20.0)
    assert len(resolved) == len(ys)
    ordered = sorted(resolved)
    assert all(b - a >= 20.0 - 1e-9 for a, b in zip(ordered, ordered[1:]))
    for i in range(len(ys)):
        for j in range(len(ys)):
            if ys[i] < ys[j]:
                assert resolved[i] <= resolved[j]


def test_resolve_label_positions_edge_cases():
    assert resolve_label_positions([], 10.0) == []
    assert resolve_label_positions([5.0], 10.0) == [5.0]
    assert resolve_label_positions([1.0, 90.0], 10.0) == [1.0, 90.0]


def test_resolve_label_positions_keeps_the_block_centred():
    resolved = resolve_label_positions([100.0, 100.0], 20.0)
    assert sorted(resolved) == [90.0, 110.0]


# --- colour identity --------------------------------------------------------

def test_colour_follows_the_ticker_not_the_rank(tmp_path, panels):
    theme = get_theme('midnight')
    renderer = ChartRenderer(theme, panels, LANDSCAPE)
    try:
        first = default_state(('AAPL', 'MSFT', 'KO'))
        renderer.render(first, tmp_path / 'a.png')
        before = {t: renderer.color_for_ticker(t)
                  for t in ('AAPL', 'MSFT', 'KO')}

        # Same tickers, ranks completely reversed.
        reranked = [make_series(t, rank=rank, seed=i) for i, (t, rank)
                    in enumerate(zip(('AAPL', 'MSFT', 'KO'), (3, 2, 1)))]
        renderer.render(make_state(reranked), tmp_path / 'b.png')
        after = {t: renderer.color_for_ticker(t)
                 for t in ('AAPL', 'MSFT', 'KO')}
    finally:
        renderer.close()

    assert before == after
    assert before['AAPL'] == theme.color_for(0)
    assert before['MSFT'] == theme.color_for(1)
    assert before['KO'] == theme.color_for(2)
    assert len(set(before.values())) == 3


# --- formatting -------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    (0, '$0'),
    (1234, '$1,234'),
    (1234.4, '$1,234'),
    (999_999, '$999,999'),
    (1_234_567, '$1.23M'),
    (2_500_000_000, '$2.50B'),
    (-1234, '-$1,234'),
    (-1_234_567, '-$1.23M'),
])
def test_format_money(value, expected):
    assert format_money(value) == expected


def test_format_money_currencies():
    assert format_money(1234, 'EUR') == '€1,234'
    assert format_money(1234, 'GBP') == '£1,234'
    assert format_money(1234, 'JPY') == '¥1,234'
    assert format_money(1234, 'THB') == '฿1,234'
    assert format_money(1234, 'SEK') == 'SEK 1,234'


def test_format_money_non_finite():
    assert format_money(float('nan')) == '—'
    assert format_money(float('inf')) == '—'
    assert format_money(None) == '—'
    assert 'nan' not in format_money(float('nan'))


def test_format_percent():
    assert format_percent(0.1234) == '+12.3%'
    assert format_percent(-0.042) == '-4.2%'
    assert format_percent(float('nan')) == '—'


def test_nan_xirr_renders_as_a_dash_not_nan(tmp_path, panels):
    series = make_series('AAPL', xirr=float('nan'))
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    try:
        renderer.render(make_state([series]), tmp_path / 'nan.png')
        texts = [row['mwr'].get_text() for row in renderer._panel_rows]
    finally:
        renderer.close()
    assert texts == ['MWR —']
    assert all('nan' not in text for text in texts)


# --- themes -----------------------------------------------------------------

@pytest.mark.parametrize('name', list_themes())
def test_every_theme_renders_a_frame(tmp_path, panels, name):
    renderer = ChartRenderer(get_theme(name), panels, LANDSCAPE)
    try:
        path = renderer.render(default_state(), tmp_path / f'{name}.png')
    finally:
        renderer.close()
    assert png_size(path) == LANDSCAPE


def test_head_labels_stay_inside_the_chart(tmp_path, panels):
    """The right-hand gutter must fit the widest label this frame can make."""
    series = [make_series('GOOGL', final_value=9_850_000.0, rank=1, seed=4),
              make_series('BRK-B', final_value=4_100_000.0, rank=2, seed=5)]
    renderer = ChartRenderer(get_theme('midnight'), panels, LANDSCAPE)
    try:
        renderer.render(make_state(series), tmp_path / 'wide.png')
        canvas = renderer.fig.canvas.get_renderer()
        bounds = renderer.ax.get_window_extent()
        for artists in renderer._series.values():
            label = artists.head_label
            if not label.get_text():
                continue
            box = label.get_window_extent(canvas)
            assert box.x0 >= bounds.x0
            assert box.x1 <= bounds.x1 + 1.0
    finally:
        renderer.close()
