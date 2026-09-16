"""Intro and outro title cards.

These are rendered once per video rather than once per frame, so unlike the
chart renderer they build a throwaway figure per call. They share the chart's
theme, surface and text colours so the cards do not look like a different
product from the frames they bracket.

Both cards lay their content out top-down in pixels, then size the surface
panel to what the content actually needed and centre it. That keeps a
three-ticker card from looking like a mostly empty box in 9:16.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')  # noqa: E402 — must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .renderer import (BASE_SHORT_EDGE, DPI, PNG_OPTIONS,  # noqa: E402
                       format_money, format_percent)
from .types import Theme  # noqa: E402

#: Rough width of one character as a fraction of the font size, used only to
#: shrink over-long titles so they never run off the card.
_CHAR_WIDTH = 0.58

_FREQUENCY_WORDS = {
    'none': 'no recurring contributions',
    'daily': 'daily',
    'weekly': 'weekly',
    'biweekly': 'every two weeks',
    'monthly': 'monthly',
    'quarterly': 'quarterly',
    'yearly': 'yearly',
}


def render_intro(config_summary: dict, theme: Theme, resolution,
                 path: Path) -> Path:
    """Title, tickers, date range and investment parameters.

    `config_summary` is read leniently: `title`, `tickers`, `start`, `end`,
    `initial_amount`, `dca_amount`, `dca_frequency`, `reinvest_dividends`,
    `fractional_shares` and `currency` are all optional. The ticker order is
    the series order, so it fixes each ticker's colour exactly as the chart
    does.
    """
    summary = dict(config_summary or {})
    tickers = [str(t) for t in summary.get('tickers', ())]
    currency = summary.get('currency', 'USD')
    parameters = _parameter_lines(summary, currency)

    fig, ax, scale = _card_figure(theme, resolution)
    width_px, height_px = int(resolution[0]), int(resolution[1])

    def sz(pixels):
        return pixels * scale

    # Lay the content out top-down in pixels, then fit the card to it.
    cursor = sz(70)
    caption_y = cursor + sz(12)
    cursor += sz(38)
    title_y = cursor + sz(36)
    cursor += sz(84)
    range_y = cursor + sz(16)
    cursor += sz(46)
    rule_y = cursor + sz(12)
    cursor += sz(58)
    ticker_top = cursor + sz(22)
    cursor += sz(58) * max(1, len(tickers)) + sz(52)
    parameter_top = cursor + sz(16)
    cursor += sz(46) * len(parameters) + sz(64)

    card = _card_panel(ax, theme, cursor, height_px, portrait=height_px > width_px)
    inner_px = (card['x1'] - card['x0']) * width_px * 0.88

    def at(pixels):
        return card['y1'] - pixels / height_px

    title = summary.get('title') or ' vs '.join(tickers) or 'Investment comparison'
    _text(ax, 0.5, at(caption_y), 'INVESTMENT COMPARISON', theme.muted,
          sz(15), theme)
    _text(ax, 0.5, at(title_y), title, theme.text,
          _fit(title, inner_px, sz(52)), theme, weight='bold')

    date_range = f"{summary.get('start', '')}  →  {summary.get('end', '')}"
    _text(ax, 0.5, at(range_y), date_range.strip(), theme.muted, sz(22), theme)
    ax.add_line(Line2D([0.44, 0.56], [at(rule_y)] * 2, color=theme.grid,
                       linewidth=1.5 * scale))

    # Ticker roll-call: each row centred so a two-letter ticker does not look
    # adrift under a centred title.
    for index, ticker in enumerate(tickers):
        y = at(ticker_top + index * sz(58))
        label_px = len(ticker) * 0.62 * sz(27) * DPI / 72.0
        marker_x = 0.5 - (label_px / 2.0 + sz(28)) / width_px
        ax.add_line(Line2D([marker_x], [y], marker='o', linestyle='none',
                           markersize=sz(11), color=theme.color_for(index),
                           markeredgecolor=theme.surface,
                           markeredgewidth=1.6 * scale))
        _text(ax, marker_x + sz(28) / width_px, y, ticker, theme.text,
              sz(27), theme, ha='left', weight='bold')

    for index, line in enumerate(parameters):
        _text(ax, 0.5, at(parameter_top + index * sz(46)), line, theme.muted,
              sz(21), theme)

    return _save(fig, path, theme)


def render_outro(final_stats: list, theme: Theme, resolution, path: Path,
                 currency: str = 'USD', order: Sequence = None) -> Path:
    """Final leaderboard in rank order with value and money-weighted return.

    Rows are ordered by rank, but colour follows the ticker, so the palette
    index comes from something rank-independent: `order` (the chart's series
    order) when given, otherwise the order of `final_stats` itself.
    """
    stats = list(final_stats or ())
    sequence = [str(t) for t in order] if order else [s.ticker for s in stats]
    colors = {ticker: theme.color_for(index)
              for index, ticker in enumerate(sequence)}
    ordered = sorted(stats, key=lambda s: (s.rank or 10 ** 6, -s.value, s.ticker))

    fig, ax, scale = _card_figure(theme, resolution)
    width_px, height_px = int(resolution[0]), int(resolution[1])

    def sz(pixels):
        return pixels * scale

    row_px = sz(104)
    cursor = sz(74)
    caption_y = cursor + sz(12)
    cursor += sz(40)
    title_y = cursor + sz(30)
    cursor += sz(108)
    row_top = cursor + sz(30)
    cursor += row_px * max(1, len(ordered)) + sz(52)

    card = _card_panel(ax, theme, cursor, height_px,
                       portrait=height_px > width_px)

    def at(pixels):
        return card['y1'] - pixels / height_px

    _text(ax, 0.5, at(caption_y), 'FINAL STANDINGS', theme.muted, sz(15), theme)
    _text(ax, 0.5, at(title_y), 'Who came out ahead', theme.text, sz(40),
          theme, weight='bold')

    left = card['x0'] + sz(70) / width_px
    right = card['x1'] - sz(70) / width_px

    for index, item in enumerate(ordered):
        y = at(row_top + index * row_px)
        _text(ax, left, y, str(item.rank or index + 1), theme.muted, sz(22),
              theme, ha='left')
        ax.add_line(Line2D([left + sz(64) / width_px], [y], marker='o',
                           linestyle='none', markersize=sz(11),
                           color=colors.get(item.ticker, theme.color_for(index)),
                           markeredgecolor=theme.surface,
                           markeredgewidth=1.6 * scale))
        _text(ax, left + sz(96) / width_px, y, item.ticker, theme.text,
              sz(26), theme, ha='left', weight='bold')
        _text(ax, right, y, format_money(item.value, currency), theme.text,
              sz(26), theme, ha='right')

        xirr = float(item.xirr)
        color = theme.muted if xirr != xirr else (
            theme.positive if xirr >= 0 else theme.negative)
        _text(ax, right, at(row_top + index * row_px + sz(34)),
              f'MWR {format_percent(item.xirr)}', color, sz(17), theme,
              ha='right')

        if index < len(ordered) - 1:
            ax.add_line(Line2D([left, right],
                               [at(row_top + index * row_px + row_px / 2.0)] * 2,
                               color=theme.grid, linewidth=1.0 * scale,
                               zorder=1))

    return _save(fig, path, theme)


# --- internals --------------------------------------------------------------

def _card_figure(theme: Theme, resolution):
    width, height = int(resolution[0]), int(resolution[1])
    scale = min(width, height) / BASE_SHORT_EDGE
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI,
                     facecolor=theme.background)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor(theme.background)
    ax.set_xticks(())
    ax.set_yticks(())
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax, scale


def _card_panel(ax, theme: Theme, content_px: float, height_px: int,
                portrait: bool) -> dict:
    """Draw the surface panel, sized to its content and vertically centred."""
    x0, x1 = (0.055, 0.945) if portrait else (0.09, 0.91)
    height = min(0.88, max(0.26, content_px / height_px))
    y0 = (1.0 - height) / 2.0
    y1 = y0 + height
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, height,
        boxstyle='round,pad=0,rounding_size=0.012',
        facecolor=theme.surface, edgecolor=theme.grid, linewidth=1.5,
        transform=ax.transAxes, zorder=0))
    return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}


def _text(ax, x: float, y: float, text: str, color: str, fontsize: float,
          theme: Theme, ha: str = 'center', weight: str = 'normal'):
    return ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha,
                   va='center', family=theme.font, fontweight=weight,
                   transform=ax.transAxes, zorder=3)


def _fit(text: str, max_px: float, fontsize: float) -> float:
    """Shrink a font size until the string fits the given pixel width."""
    if not text:
        return fontsize
    width = len(text) * _CHAR_WIDTH * fontsize * DPI / 72.0
    if width <= max_px:
        return fontsize
    return max(fontsize * 0.45, fontsize * max_px / width)


def _parameter_lines(summary: dict, currency: str) -> list:
    initial = summary.get('initial_amount', 0.0) or 0.0
    dca = summary.get('dca_amount', 0.0) or 0.0
    frequency = str(summary.get('dca_frequency', 'none')).lower()
    lines = [f'Initial investment {format_money(initial, currency)}']
    if dca and frequency != 'none':
        lines.append(f'plus {format_money(dca, currency)} '
                     f'{_FREQUENCY_WORDS.get(frequency, frequency)}')
    else:
        lines.append('no recurring contributions')
    lines.append('Dividends reinvested' if summary.get('reinvest_dividends', True)
                 else 'Dividends taken as cash')
    if summary.get('fractional_shares', True) is False:
        lines.append('Whole shares only')
    return lines


def _save(fig, path: Path, theme: Theme) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=theme.background,
                pil_kwargs=PNG_OPTIONS)
    plt.close(fig)
    return path
