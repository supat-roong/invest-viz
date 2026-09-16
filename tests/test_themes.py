"""Themes are a safety mechanism, not decoration — these tests guard that."""
from __future__ import annotations

import re

import pytest

from investment_video.themes import (MAX_SERIES, get_theme, is_dark,
                                     list_themes)

HEX = re.compile(r'^#[0-9a-f]{6}$')


def test_six_themes_are_registered():
    assert list_themes() == ['midnight', 'paper', 'terminal', 'bloomberg',
                             'solarized', 'neon']


@pytest.mark.parametrize('name', list_themes())
def test_every_theme_is_fully_specified(name):
    theme = get_theme(name)
    for field in ('background', 'surface', 'grid', 'text', 'muted',
                  'positive', 'negative', 'invested'):
        value = getattr(theme, field)
        assert HEX.match(value), f'{name}.{field} is not lowercase #rrggbb: {value}'
    assert theme.name == name
    assert theme.font


@pytest.mark.parametrize('name', list_themes())
def test_every_theme_carries_the_full_series_palette(name):
    theme = get_theme(name)
    assert len(theme.series) == MAX_SERIES
    assert all(HEX.match(c) for c in theme.series)
    assert len(set(theme.series)) == len(theme.series), 'duplicate series colour'


@pytest.mark.parametrize('name', list_themes())
def test_series_colours_differ_from_the_background(name):
    theme = get_theme(name)
    assert theme.background not in theme.series


def test_color_for_wraps_around():
    theme = get_theme('midnight')
    assert theme.color_for(0) == theme.series[0]
    assert theme.color_for(len(theme.series)) == theme.series[0]
    assert theme.color_for(len(theme.series) + 3) == theme.series[3]


def test_unknown_theme_names_the_alternatives():
    with pytest.raises(KeyError) as excinfo:
        get_theme('vaporwave')
    message = str(excinfo.value)
    assert 'vaporwave' in message
    for name in list_themes():
        assert name in message


def test_is_dark_matches_the_declared_palettes():
    assert is_dark(get_theme('midnight'))
    assert is_dark(get_theme('bloomberg'))
    assert not is_dark(get_theme('paper'))
