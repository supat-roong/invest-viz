"""Colour themes for the chart.

Every `series` list below was produced by searching hue orderings against the
dataviz palette validator and keeping only orders that clear all of its hard
gates on that theme's own surface. Reproduce with:

    node validate_palette.js "<series joined by commas>" --mode <mode> --surface <background>

Recorded results (validator exit 0 for all six):

    midnight   dark  #0d1117  CVD 8.4  normal 19.3  contrast all >= 3:1
    paper      light #faf9f6  CVD 9.1  normal 19.6  contrast WARN on 3 slots
    terminal   dark  #07100a  CVD 9.5  normal 15.6  contrast all >= 3:1
    bloomberg  dark  #000000  CVD 11.3 normal 18.9  contrast all >= 3:1
    solarized  dark  #002b36  CVD 13.4 normal 25.0  contrast all >= 3:1
    neon       dark  #0a0a12  CVD 12.2 normal 21.4  contrast all >= 3:1

`paper` is the one theme with slots below 3:1 against its surface, which the
validator flags as needing relief. The renderer always direct-labels each line
at its head, which supplies that relief; do not remove head labels from the
light theme.

Series order is the colourblind-safety mechanism, not decoration. Reordering
these lists invalidates the results above — re-run the validator if you do.
"""
from __future__ import annotations

from .types import Theme

MIDNIGHT = Theme(
    name='midnight',
    background='#0d1117',
    surface='#161b22',
    grid='#21262d',
    text='#e6edf3',
    muted='#8b949e',
    series=('#3987e5', '#d95926', '#199e70', '#c98500',
            '#d55181', '#008300', '#9085e9', '#e66767'),
    positive='#3fb950',
    negative='#f85149',
    invested='#8b949e',
    font='DejaVu Sans',
)

PAPER = Theme(
    name='paper',
    background='#faf9f6',
    surface='#ffffff',
    grid='#e5e3dd',
    text='#1a1a19',
    muted='#6b6a65',
    series=('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
            '#e87ba4', '#008300', '#4a3aa7', '#e34948'),
    positive='#1a7f37',
    negative='#cf222e',
    invested='#6b6a65',
    font='DejaVu Sans',
)

TERMINAL = Theme(
    name='terminal',
    background='#07100a',
    surface='#0c1a12',
    grid='#163020',
    text='#c8f5d8',
    muted='#5f9e78',
    series=('#0fb059', '#cf5fae', '#84a20a', '#6f8cf0',
            '#d96a4a', '#1ba5be', '#bf8900', '#c9524f'),
    positive='#0fb059',
    negative='#c9524f',
    invested='#5f9e78',
    font='DejaVu Sans Mono',
)

BLOOMBERG = Theme(
    name='bloomberg',
    background='#000000',
    surface='#0a0a0a',
    grid='#1f1a12',
    text='#f5c877',
    muted='#8a7346',
    series=('#c78409', '#34ab81', '#aa7dd0', '#cf7a3a',
            '#489cd2', '#a89500', '#5f8fd6', '#d96a6a'),
    positive='#34ab81',
    negative='#d96a6a',
    invested='#8a7346',
    font='DejaVu Sans Mono',
)

SOLARIZED = Theme(
    name='solarized',
    background='#002b36',
    surface='#073642',
    grid='#0b4a5a',
    text='#93a1a1',
    muted='#586e75',
    series=('#268bd2', '#cb4b16', '#22a298', '#dc322f',
            '#6c71c4', '#b58900', '#d33682', '#859900'),
    positive='#859900',
    negative='#dc322f',
    invested='#586e75',
    font='DejaVu Sans',
)

NEON = Theme(
    name='neon',
    background='#0a0a12',
    surface='#12121f',
    grid='#1e1e35',
    text='#e8e8ff',
    muted='#7a7a9e',
    series=('#00a7c1', '#f2564a', '#8a7cf5', '#cd8100',
            '#00ad86', '#d94fd9', '#4cae00', '#f0489b'),
    positive='#00ad86',
    negative='#f2564a',
    invested='#7a7a9e',
    font='DejaVu Sans',
)

_THEMES = {t.name: t for t in (MIDNIGHT, PAPER, TERMINAL, BLOOMBERG, SOLARIZED, NEON)}

MAX_SERIES = min(len(t.series) for t in _THEMES.values())


def list_themes() -> list:
    """Theme names in display order."""
    return list(_THEMES)


def get_theme(name: str) -> Theme:
    try:
        return _THEMES[name]
    except KeyError:
        raise KeyError(f'unknown theme {name!r}; available: '
                       f'{", ".join(_THEMES)}') from None


def is_dark(theme: Theme) -> bool:
    """True when the background is dark enough to need light-on-dark text."""
    hexcode = theme.background.lstrip('#')
    r, g, b = (int(hexcode[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5
