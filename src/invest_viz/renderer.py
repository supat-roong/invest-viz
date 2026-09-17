"""Single-frame chart rendering.

One `ChartRenderer` is built per video and reused for every frame: the Figure,
the Axes and every artist are created once and only their data is updated, so
a 1400-frame clip costs one figure build rather than 1400. Updating the
artists costs about 2ms a frame against ~35ms to build the figure plus the
cost of recreating every artist, so this is the main thing the module is
shaped around.

Charting rules this module obeys: one y-scale per axes (dividends get their
own short sub-axis rather than a forbidden second y-axis), colour follows the
ticker and never its rank, text wears text colours while a coloured marker
beside it carries identity, and every line is direct-labelled at its head.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')  # noqa: E402 — must precede the pyplot import

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

from .config import PanelsConfig  # noqa: E402
from .types import EventKind, FrameState, SeriesSlice, Theme  # noqa: E402

#: Every figure is built at this dpi, so one point is exactly 100/72 pixels
#: and a 1920x1080 frame is figsize (19.2, 10.8).
DPI = 100

#: Frames are an intermediate handed straight to ffmpeg and deleted after, so
#: they use the cheapest PNG compression. The default level 6 costs several
#: extra milliseconds a frame and buys nothing downstream.
PNG_OPTIONS = {'compress_level': 1}

#: Font sizes and gaps are quoted for a 1080px short edge and scaled from it.
BASE_SHORT_EDGE = 1080

LINE_WIDTH = 2.0
DIVIDEND_LINE_WIDTH = 1.0
INVESTED_LINE_WIDTH = 1.8
HEAD_MARKER_POINTS = 7.0          # 7pt ~ 9.7px at dpi 100, above the 8px floor
EVENT_MARKER_POINTS = {'dividend': 3.4, 'split': 6.0}
MIN_LABEL_GAP_PX = 33.0
LABEL_OFFSET_PX = 13.0

#: The dividend strip is a fifth of the chart's height, so its head labels run
#: smaller and pack tighter than the ones on the main chart.
DIVIDEND_LABEL_FONT_POINTS = 11.0
DIVIDEND_LABEL_GAP_PX = 17.0
DIVIDEND_HEAD_MARKER_POINTS = 4.5

#: Head labels live inside the axes, in a gutter reserved on the right so
#: they never flip sides part-way through the animation. The gutter is sized
#: from the widest label the frame can produce, clamped to this fraction of
#: the plot width.
MAX_LABEL_GUTTER_FRACTION = 0.34
LABEL_FONT_POINTS = 15.0

#: Rough width of one character as a fraction of the font size. Deliberately
#: generous: over-reserving costs a little chart width, under-reserving puts a
#: label on top of the lines.
CHAR_WIDTH = 0.64

#: Main chart to dividend sub-axis height ratio.
DIVIDEND_HEIGHT_RATIO = 0.25

CURRENCY_SYMBOLS = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'THB': '฿',
    'CAD': 'C$',
    'AUD': 'A$',
    'CHF': 'CHF ',
    'CNY': '¥',
    'INR': '₹',
}

#: What a non-finite number renders as. Never the string 'nan'.
MISSING = '—'

_EVENT_MARKERS = {EventKind.DIVIDEND: 'o', EventKind.SPLIT: '^'}


def currency_symbol(currency: str) -> str:
    """Symbol for a currency code, falling back to the code plus a space."""
    if not currency:
        return CURRENCY_SYMBOLS['USD']
    code = str(currency).upper()
    return CURRENCY_SYMBOLS.get(code, f'{code} ')


def format_money(value: float, currency: str = 'USD') -> str:
    """Compact money: 1234 -> '$1,234', 1234567 -> '$1.23M', nan -> em dash."""
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not np.isfinite(number):
        return MISSING

    symbol = currency_symbol(currency)
    sign = '-' if number < 0 else ''
    magnitude = abs(number)
    for limit, suffix in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M')):
        if magnitude >= limit:
            return f'{sign}{symbol}{magnitude / limit:.2f}{suffix}'
    return f'{sign}{symbol}{magnitude:,.0f}'


def format_percent(value: float, decimals: int = 1) -> str:
    """A rate as a signed percent, or an em dash when it is not finite."""
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    if not np.isfinite(number):
        return MISSING
    return f'{number * 100:+.{decimals}f}%'


def resolve_label_positions(ys: Sequence[float],
                            min_gap: float) -> list[float]:
    """Push colliding label positions apart, preserving their order.

    Returns one position per input in input order. No two results are closer
    than `min_gap`, relative order is preserved, and the block keeps the
    centre of the original positions so labels do not drift off one end.
    """
    values = [float(y) for y in ys]
    count = len(values)
    if count < 2:
        return values
    gap = max(0.0, float(min_gap))
    if gap == 0.0:
        return values

    order = sorted(range(count), key=lambda i: (values[i], i))
    resolved = [0.0] * count
    previous: float | None = None
    for index in order:
        position = values[index]
        if previous is not None and position < previous + gap:
            position = previous + gap
        resolved[index] = position
        previous = position

    # A uniform shift preserves both the ordering and every gap.
    shift = ((max(values) + min(values)) - (max(resolved) + min(resolved))) / 2.0
    if shift:
        resolved = [position + shift for position in resolved]
    return resolved


class _SeriesArtists:
    """The reusable artists belonging to one ticker."""

    __slots__ = ('value_line', 'head_marker', 'head_label', 'dividend_line',
                 'dividend_marker', 'dividend_label', 'event_markers')

    def __init__(self, value_line, head_marker, head_label, dividend_line,
                 dividend_marker, dividend_label, event_markers):
        self.value_line = value_line
        self.head_marker = head_marker
        self.head_label = head_label
        self.dividend_line = dividend_line
        self.dividend_marker = dividend_marker
        self.dividend_label = dividend_label
        self.event_markers = event_markers


class ChartRenderer:
    """Draws FrameStates to PNG, reusing one Figure for the whole video."""

    def __init__(self, theme: Theme, panels: PanelsConfig,
                 resolution: tuple[int, int], currency: str = 'USD') -> None:
        self.theme = theme
        self.panels = panels
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.currency = currency
        self.dpi = DPI

        width, height = self.resolution
        self.scale = min(width, height) / BASE_SHORT_EDGE
        self.min_label_gap = MIN_LABEL_GAP_PX * self.scale
        self.dividend_label_gap = DIVIDEND_LABEL_GAP_PX * self.scale
        self.portrait = height > width

        #: ticker -> resolved head-label y in pixels, refreshed by render().
        self.label_positions: dict[str, float] = {}

        self.fig = plt.figure(figsize=(width / self.dpi, height / self.dpi),
                              dpi=self.dpi, facecolor=theme.background)
        self.fig.patch.set_facecolor(theme.background)

        self._color_index: dict[str, int] = {}
        self._series: dict[str, _SeriesArtists] = {}
        self._tickers: tuple[str, ...] = ()
        self._panel_rows: list[dict] = []
        self._x_axis_style: str | None = None
        self._x_data_max: float | None = None
        self._x_date_format = '%Y'
        self._layout_cache = self._layout()

        self._build_axes()
        self._build_static_artists()

    # --- geometry -----------------------------------------------------------

    def _layout(self) -> dict:
        """Figure-fraction rectangles for this aspect and panel set."""
        stats = bool(self.panels.stats)
        if self.portrait:
            chart_left, chart_right = 0.105, 0.945
            top = 0.895
            bottom = 0.385 if stats else 0.075
            panel = (0.045, 0.045, 0.910, 0.285) if stats else None
        else:
            chart_left = 0.055
            chart_right = 0.655 if stats else 0.965
            top, bottom = 0.860, 0.085
            panel = (0.690, 0.085, 0.285, 0.775) if stats else None
        return {
            'chart_left': chart_left,
            'chart_right': chart_right,
            'top': top,
            'bottom': bottom,
            'panel': panel,
        }

    def _build_axes(self) -> None:
        theme = self.theme
        layout = self._layout_cache
        left = layout['chart_left']
        width = layout['chart_right'] - layout['chart_left']
        top, bottom = layout['top'], layout['bottom']
        span = top - bottom

        if self.panels.dividend_line:
            self.ax = self.fig.add_axes((left, bottom, width, span))
            self.ax_dividends = self.fig.add_axes(
                (left, bottom, width, span * 0.2), sharex=self.ax)
        else:
            self.ax = self.fig.add_axes((left, bottom, width, span))
            self.ax_dividends = None
        self._place_chart(bottom)

        self._style_axes(self.ax, show_x_labels=self.ax_dividends is None)
        if self.ax_dividends is not None:
            self._style_axes(self.ax_dividends, show_x_labels=True)
            self.ax_dividends.yaxis.set_major_locator(MaxNLocator(3))
            self.ax_dividends.text(
                0.008, 0.80, 'dividends received',
                transform=self.ax_dividends.transAxes, color=theme.muted,
                fontsize=11.5 * self.scale, family=theme.font, va='bottom')

        self.panel_ax = None
        if layout['panel'] is not None:
            self.panel_ax = self.fig.add_axes(layout['panel'])
            self.panel_ax.set_facecolor(theme.surface)
            self.panel_ax.set_xticks(())
            self.panel_ax.set_yticks(())
            for spine in self.panel_ax.spines.values():
                spine.set_visible(False)

    def _place_chart(self, bottom: float) -> None:
        """Position the chart axes between `bottom` and the header band."""
        layout = self._layout_cache
        left = layout['chart_left']
        width = layout['chart_right'] - layout['chart_left']
        span = max(0.05, layout['top'] - bottom)
        if self.ax_dividends is None:
            self.ax.set_position((left, bottom, width, span))
            return
        gap = 0.045 * span
        dividend_height = (span - gap) * (
            DIVIDEND_HEIGHT_RATIO / (1.0 + DIVIDEND_HEIGHT_RATIO))
        self.ax.set_position((left, bottom + dividend_height + gap, width,
                              span - gap - dividend_height))
        self.ax_dividends.set_position((left, bottom, width, dividend_height))

    def _style_axes(self, ax, show_x_labels: bool) -> None:
        theme = self.theme
        ax.set_facecolor(theme.surface)
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color=theme.grid,
                linewidth=0.9 * self.scale, alpha=1.0)
        ax.grid(False, axis='x')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(theme.grid)
            ax.spines[side].set_linewidth(1.0 * self.scale)
        ax.tick_params(colors=theme.muted, labelsize=12.5 * self.scale,
                       length=0, pad=6 * self.scale,
                       labelfontfamily=theme.font)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: format_money(v, self.currency)))
        if not show_x_labels:
            ax.tick_params(axis='x', labelbottom=False)

    def _build_static_artists(self) -> None:
        theme = self.theme
        layout = self._layout_cache
        height = self.resolution[1]
        top_px = layout['top'] * height

        self.date_text = self.fig.text(
            layout['chart_left'], (top_px + 20 * self.scale) / height, '',
            color=theme.text, family=theme.font, fontweight='bold',
            fontsize=30 * self.scale, va='bottom', ha='left')
        self.caption_text = self.fig.text(
            layout['chart_left'],
            (top_px + 78 * self.scale) / height, 'PORTFOLIO VALUE',
            color=theme.muted, family=theme.font,
            fontsize=13 * self.scale, va='bottom', ha='left')

        self._invested_line = None
        self._invested_label = None
        if self.panels.invested_line:
            self._invested_line = Line2D(
                [], [], color=theme.invested, linewidth=INVESTED_LINE_WIDTH * self.scale,
                linestyle=(0, (6, 4)), drawstyle='steps-post', zorder=2)
            self.ax.add_line(self._invested_line)
            self._invested_label = self.ax.text(
                0, 0, '', color=theme.muted, family=theme.font,
                fontsize=14 * self.scale, va='center', ha='left',
                transform=IdentityTransform(), clip_on=False, zorder=6)

        if self.panel_ax is not None:
            self.panel_title = self._panel_text(
                0.055, 0.0, 'STANDINGS', color=theme.muted,
                fontsize=12.5 * self.scale, ha='left', va='top')
            self.panel_title.set_text('STANDINGS')

    # --- colours ------------------------------------------------------------

    def color_for_ticker(self, ticker: str) -> str:
        """The series colour of a ticker, fixed by its index, never its rank."""
        index = self._color_index.get(ticker)
        if index is None:
            index = len(self._color_index)
            self._color_index[ticker] = index
        return self.theme.color_for(index)

    # --- artist lifecycle ---------------------------------------------------

    def _ensure_series(self, state: FrameState) -> None:
        tickers = tuple(s.ticker for s in state.series)
        if tickers == self._tickers:
            return

        for artists in self._series.values():
            artists.value_line.remove()
            artists.head_marker.remove()
            artists.head_label.remove()
            if artists.dividend_line is not None:
                artists.dividend_line.remove()
                artists.dividend_marker.remove()
                artists.dividend_label.remove()
            for marker in artists.event_markers.values():
                marker.remove()
        self._series.clear()

        for position, slice_ in enumerate(state.series):
            self._color_index.setdefault(slice_.ticker, position)
        for slice_ in state.series:
            self._series[slice_.ticker] = self._make_series_artists(slice_.ticker)

        self._tickers = tickers
        self._rebuild_panel_rows(len(tickers))

    def _make_series_artists(self, ticker: str) -> _SeriesArtists:
        theme = self.theme
        color = self.color_for_ticker(ticker)

        value_line = Line2D([], [], color=color,
                            linewidth=LINE_WIDTH * self.scale,
                            solid_capstyle='round', zorder=4)
        self.ax.add_line(value_line)

        head_marker = Line2D([], [], color=color, marker='o', linestyle='none',
                             markersize=HEAD_MARKER_POINTS * self.scale,
                             markeredgecolor=theme.surface,
                             markeredgewidth=1.6 * self.scale, zorder=5)
        self.ax.add_line(head_marker)

        head_label = self.ax.text(
            0, 0, '', color=theme.text, family=theme.font,
            fontsize=LABEL_FONT_POINTS * self.scale, fontweight='bold',
            va='center',
            ha='left', transform=IdentityTransform(), clip_on=False, zorder=6)

        dividend_line = None
        dividend_marker = None
        dividend_label = None
        if self.ax_dividends is not None:
            dividend_line = Line2D(
                [], [], color=color, linewidth=DIVIDEND_LINE_WIDTH * self.scale,
                linestyle=':', zorder=4)
            self.ax_dividends.add_line(dividend_line)
            dividend_marker = Line2D(
                [], [], color=color, marker='o', linestyle='none',
                markersize=DIVIDEND_HEAD_MARKER_POINTS * self.scale,
                markeredgecolor=theme.surface,
                markeredgewidth=1.2 * self.scale, zorder=5)
            self.ax_dividends.add_line(dividend_marker)
            dividend_label = self.ax_dividends.text(
                0, 0, '', color=theme.text, family=theme.font,
                fontsize=DIVIDEND_LABEL_FONT_POINTS * self.scale,
                va='center', ha='left', transform=IdentityTransform(),
                clip_on=False, zorder=6)

        event_markers = {}
        if self.panels.event_markers:
            for kind, marker in _EVENT_MARKERS.items():
                size = EVENT_MARKER_POINTS[kind.value] * self.scale
                artist = Line2D([], [], color=color, marker=marker,
                                linestyle='none', markersize=size,
                                markeredgecolor=theme.surface,
                                markeredgewidth=1.2 * self.scale, zorder=4.5)
                self.ax.add_line(artist)
                event_markers[kind] = artist

        return _SeriesArtists(value_line, head_marker, head_label,
                              dividend_line, dividend_marker, dividend_label,
                              event_markers)

    def _panel_text(self, x: float, y: float, text: str, color: str,
                    fontsize: float, ha: str = 'left', va: str = 'center',
                    weight: str = 'normal'):
        artist = self.panel_ax.text(
            x, y, text, color=color, family=self.theme.font, fontsize=fontsize,
            ha=ha, va=va, fontweight=weight, transform=self.panel_ax.transAxes,
            clip_on=False)
        return artist

    def _rebuild_panel_rows(self, count: int) -> None:
        if self.panel_ax is None:
            return
        for row in self._panel_rows:
            for artist in row.values():
                artist.remove()
        self._panel_rows = []

        theme = self.theme
        scale = self.scale
        x0, y0, width, height = self._layout_cache['panel']
        figure_height = self.resolution[1]
        header_px = 44 * scale
        foot_px = 20 * scale
        available = height * figure_height - header_px - foot_px
        row_px = min(94 * scale, available / max(1, count))

        # Shrink the surface to its content: a tall half-empty box reads as a
        # mistake. Landscape hangs it from the chart top, portrait sits it on
        # the bottom edge of its band.
        panel_px = min(height * figure_height,
                       header_px + row_px * count + foot_px)
        panel_height = panel_px / figure_height
        top = y0 + height if not self.portrait else y0 + panel_height
        self.panel_ax.set_position((x0, top - panel_height, width, panel_height))
        if self.portrait:
            # Let the chart claim whatever the standings did not need, rather
            # than leaving a band of empty background between the two.
            self._place_chart(top + 0.030)

        panel_width_px = width * self.resolution[0]
        column = 0.115
        column_2 = column + 172 * scale / panel_width_px
        self.panel_title.set_position(
            (0.055, (panel_px - 24 * scale) / panel_px))

        for index in range(count):
            top_px = panel_px - header_px - index * row_px
            line1 = (top_px - 22 * scale) / panel_px
            line2 = (top_px - 50 * scale) / panel_px
            marker = Line2D([0.055], [line1], marker='o', linestyle='none',
                            markersize=8.5 * scale, color=theme.muted,
                            transform=self.panel_ax.transAxes, clip_on=False)
            self.panel_ax.add_line(marker)
            self._panel_rows.append({
                'marker': marker,
                'ticker': self._panel_text(column, line1, '', theme.text,
                                           16 * scale, weight='bold'),
                'value': self._panel_text(0.945, line1, '', theme.text,
                                          15.5 * scale, ha='right'),
                'mwr': self._panel_text(column, line2, '', theme.muted,
                                        12.5 * scale),
                'detail': self._panel_text(column_2, line2, '', theme.muted,
                                           12.5 * scale),
            })

    # --- rendering ----------------------------------------------------------

    def render(self, state: FrameState, path: Path) -> Path:
        """Draw one frame and write it as a PNG, returning the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_series(state)
        self._update_axes(state)
        self._update_series(state)
        self._update_labels(state)
        self._update_dividend_labels(state)
        self._update_panel(state)
        self.date_text.set_text(state.date.strftime('%Y-%m-%d'))

        self.fig.savefig(path, dpi=self.dpi, facecolor=self.theme.background,
                         pil_kwargs=PNG_OPTIONS)
        return path

    def _update_axes(self, state: FrameState) -> None:
        x_min = mdates.date2num(state.x_min)
        x_max = mdates.date2num(state.x_max)
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_max = x_min + 1.0
        span = x_max - x_min
        self.ax.set_xlim(x_min, x_max + span * self._gutter_fraction(state))

        y_min, y_max = float(state.y_min), float(state.y_max)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_max = (y_min if np.isfinite(y_min) else 0.0) + 1.0
            y_min = y_max - 1.0
        self.ax.set_ylim(y_min, y_max)
        self.ax.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
        self._x_data_max = x_max
        self._update_x_axis(span)

        if self.ax_dividends is not None:
            top = 0.0
            for slice_ in state.series:
                if len(slice_.cum_dividends):
                    last = float(np.nanmax(slice_.cum_dividends[-1:]))
                    if np.isfinite(last):
                        top = max(top, last)
            self.ax_dividends.set_ylim(0.0, top * 1.3 if top > 0 else 1.0)

    def _gutter_fraction(self, state: FrameState) -> float:
        """Share of the x span reserved for head labels, as a fraction.

        Sized from the widest label this frame can produce so a label never
        has to flip onto the lines, and quantised so the axis does not
        breathe by a pixel every time a value gains a digit.
        """
        longest = max((len(s.ticker) for s in state.series), default=4)
        if self.panels.invested_line:
            longest = max(longest, len('invested'))
        characters = longest + 2 + len(format_money(state.y_max, self.currency))
        needed = (characters * CHAR_WIDTH * LABEL_FONT_POINTS * self.scale
                  * DPI / 72.0) + (LABEL_OFFSET_PX + 24) * self.scale
        plot_width = max(1.0, self.ax.get_window_extent().width)
        fraction = math.ceil(needed / plot_width * 40.0) / 40.0
        fraction = min(MAX_LABEL_GUTTER_FRACTION, fraction)
        return fraction / (1.0 - fraction)

    def _update_x_axis(self, span_days: float) -> None:
        years = span_days / 365.0
        if years < 2:
            style = 'months'
        elif years <= 9:
            style = 'years:1'
        elif years <= 18:
            style = 'years:2'
        else:
            style = 'years:5'
        if style == self._x_axis_style:
            return
        axis = (self.ax_dividends or self.ax).xaxis
        if style.startswith('years'):
            axis.set_major_locator(mdates.YearLocator(int(style.split(':')[1])))
            self._x_date_format = '%Y'
        else:
            axis.set_major_locator(mdates.AutoDateLocator(maxticks=7))
            self._x_date_format = '%b %Y'
        axis.set_major_formatter(FuncFormatter(self._format_date_tick))
        self._x_axis_style = style

    def _format_date_tick(self, value: float, _position=None) -> str:
        """Blank any tick past the end of the data.

        The head-label gutter widens xlim beyond the final bar, and the year
        locator happily places ticks out there. Labelling them would put a
        year on the axis that the comparison never reaches: a run ending
        2025-01-01 was drawing a 2026 tick.
        """
        if self._x_data_max is not None and value > self._x_data_max:
            return ''
        try:
            return mdates.num2date(value).strftime(self._x_date_format)
        except (ValueError, OverflowError):
            return ''

    def _update_series(self, state: FrameState) -> None:
        for slice_ in state.series:
            artists = self._series[slice_.ticker]
            xs = _to_numbers(slice_.dates)
            values = np.asarray(slice_.values, dtype=float)
            artists.value_line.set_data(xs, values)

            if len(xs):
                artists.head_marker.set_data(xs[-1:], values[-1:])
            else:
                artists.head_marker.set_data((), ())

            if artists.dividend_line is not None:
                artists.dividend_line.set_data(
                    xs, np.asarray(slice_.cum_dividends, dtype=float))

            self._update_events(slice_, artists, xs, values)

        if self._invested_line is not None:
            first = state.series[0] if state.series else None
            if first is not None and len(first.invested):
                self._invested_line.set_data(
                    _to_numbers(first.dates),
                    np.asarray(first.invested, dtype=float))
            else:
                self._invested_line.set_data((), ())

    def _update_events(self, slice_: SeriesSlice, artists: _SeriesArtists,
                       xs: np.ndarray, values: np.ndarray) -> None:
        if not artists.event_markers:
            return
        points: dict = {kind: ([], []) for kind in _EVENT_MARKERS}
        if len(xs) and slice_.events:
            dates = np.asarray(slice_.dates, dtype='datetime64[ns]')
            for event in slice_.events:
                stamp = np.datetime64(event.date.to_datetime64()
                                      if hasattr(event.date, 'to_datetime64')
                                      else event.date, 'ns')
                index = int(np.searchsorted(dates, stamp, side='right')) - 1
                if index < 0 or index >= len(xs):
                    continue
                bucket = points.get(EventKind(event.kind))
                if bucket is None:
                    continue
                bucket[0].append(xs[index])
                bucket[1].append(values[index])
        for kind, artist in artists.event_markers.items():
            artist.set_data(points[kind][0], points[kind][1])

    def _update_labels(self, state: FrameState) -> None:
        """Place head labels, pushing colliding ones apart in pixel space."""
        transform = self.ax.transData
        entries = []
        for slice_ in state.series:
            artists = self._series[slice_.ticker]
            if not len(slice_.values):
                artists.head_label.set_text('')
                continue
            x_data = _to_numbers(slice_.dates)[-1]
            y_data = float(slice_.values[-1])
            if not np.isfinite(y_data):
                artists.head_label.set_text('')
                continue
            px, py = transform.transform((x_data, y_data))
            artists.head_label.set_text(
                f'{slice_.ticker}  {format_money(y_data, self.currency)}')
            entries.append((artists.head_label, px, py))

        if self._invested_label is not None:
            first = state.series[0] if state.series else None
            if first is not None and len(first.invested):
                x_data = _to_numbers(first.dates)[-1]
                y_data = float(first.invested[-1])
                px, py = transform.transform((x_data, y_data))
                self._invested_label.set_text(
                    f'invested  {format_money(y_data, self.currency)}')
                entries.append((self._invested_label, px, py))
            else:
                self._invested_label.set_text('')

        resolved = resolve_label_positions([entry[2] for entry in entries],
                                           self.min_label_gap)
        bounds = self.ax.get_window_extent()
        self.label_positions = {}
        for (artist, px, _py), y in zip(entries, resolved):
            artist.set_position((px + LABEL_OFFSET_PX * self.scale, y))
            artist.set_ha('left')
            text = artist.get_text()
            estimated = len(text) * 0.60 * artist.get_fontsize() * DPI / 72.0
            if px + LABEL_OFFSET_PX * self.scale + estimated > bounds.x1:
                artist.set_position((px - LABEL_OFFSET_PX * self.scale, y))
                artist.set_ha('right')
            if artist is not self._invested_label:
                self.label_positions[text.split('  ')[0]] = y

    def _update_dividend_labels(self, state: FrameState) -> None:
        """Direct-label each dividend line with the cash it has paid out.

        The strip is short, so positions are packed with a smaller gap than
        the main chart and the whole block is clamped inside the axes rather
        than allowed to spill onto the chart or the date ticks.
        """
        if self.ax_dividends is None:
            return
        transform = self.ax_dividends.transData
        entries = []
        for slice_ in state.series:
            artists = self._series[slice_.ticker]
            total = (float(slice_.cum_dividends[-1])
                     if len(slice_.cum_dividends) else float('nan'))
            if not len(slice_.dates) or not np.isfinite(total):
                artists.dividend_label.set_text('')
                artists.dividend_marker.set_data((), ())
                continue
            x_data = _to_numbers(slice_.dates)[-1]
            artists.dividend_marker.set_data([x_data], [total])
            artists.dividend_label.set_text(format_money(total, self.currency))
            entries.append((artists.dividend_label,
                            *transform.transform((x_data, total))))

        if not entries:
            return
        resolved = resolve_label_positions([entry[2] for entry in entries],
                                           self.dividend_label_gap)
        bounds = self.ax_dividends.get_window_extent()
        half = DIVIDEND_LABEL_FONT_POINTS * self.scale * DPI / 72.0 / 2.0
        low, high = bounds.y0 + half, bounds.y1 - half
        overflow = max(0.0, (max(resolved) - min(resolved)) - (high - low))
        if overflow:  # taller than the strip: keep the block centred on it
            low, high = low - overflow / 2.0, high + overflow / 2.0
        shift = max(low - min(resolved), min(0.0, high - max(resolved)))
        for (artist, px, _py), y in zip(entries, resolved):
            artist.set_position((px + LABEL_OFFSET_PX * self.scale, y + shift))

    def _update_panel(self, state: FrameState) -> None:
        if self.panel_ax is None:
            return
        theme = self.theme
        ordered = sorted(state.series,
                         key=lambda s: (s.stats.rank or 10 ** 6, s.ticker))
        for row, slice_ in zip(self._panel_rows, ordered):
            stats = slice_.stats
            row['marker'].set_color(self.color_for_ticker(slice_.ticker))
            row['marker'].set_markeredgecolor(theme.surface)
            row['ticker'].set_text(slice_.ticker)
            row['value'].set_text(format_money(stats.value, self.currency))

            xirr = float(stats.xirr)
            row['mwr'].set_text(f'MWR {format_percent(xirr)}')
            if not np.isfinite(xirr):
                row['mwr'].set_color(theme.muted)
            else:
                row['mwr'].set_color(theme.positive if xirr >= 0
                                     else theme.negative)

            row['detail'].set_text(
                f'DD {format_percent(stats.max_drawdown)}     '
                f'Div {format_money(stats.cum_dividends, self.currency)}')

        for row in self._panel_rows[len(ordered):]:
            for key in ('ticker', 'value', 'mwr', 'detail'):
                row[key].set_text('')

    def close(self) -> None:
        """Release the figure. The renderer must not be used afterwards."""
        plt.close(self.fig)


def _to_numbers(dates) -> np.ndarray:
    """Dates as matplotlib float days, for any datetime-ish array."""
    array = np.asarray(dates)
    if array.size == 0:
        return np.empty(0, dtype=float)
    return mdates.date2num(array)
