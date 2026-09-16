"""Intro and outro card tests: offline, no ffmpeg."""
from __future__ import annotations

import pytest
from PIL import Image

from invest_viz.cards import render_intro, render_outro
from invest_viz.config import ASPECT_RESOLUTIONS
from invest_viz.themes import get_theme, list_themes
from invest_viz.types import TickerStats

LANDSCAPE = ASPECT_RESOLUTIONS['16:9']
PORTRAIT = ASPECT_RESOLUTIONS['9:16']
RESOLUTIONS = [LANDSCAPE, PORTRAIT]

SUMMARY = {
    'title': 'AAPL vs MSFT vs KO',
    'tickers': ['AAPL', 'MSFT', 'KO'],
    'start': '2015-01-01',
    'end': '2025-01-01',
    'initial_amount': 10_000.0,
    'dca_amount': 500.0,
    'dca_frequency': 'monthly',
    'reinvest_dividends': True,
    'fractional_shares': True,
    'currency': 'USD',
}

FINAL_STATS = [
    TickerStats(ticker='AAPL', value=124_500.0, invested=65_000.0,
                total_return=0.915, xirr=0.163, max_drawdown=-0.312,
                cum_dividends=4_120.0, rank=1),
    TickerStats(ticker='MSFT', value=118_900.0, invested=65_000.0,
                total_return=0.829, xirr=0.151, max_drawdown=-0.284,
                cum_dividends=3_480.0, rank=2),
    TickerStats(ticker='KO', value=81_200.0, invested=65_000.0,
                total_return=0.249, xirr=0.058, max_drawdown=-0.221,
                cum_dividends=9_640.0, rank=3),
]


def png_size(path) -> tuple:
    with Image.open(path) as image:
        return image.size


@pytest.mark.parametrize('resolution', RESOLUTIONS)
def test_intro_writes_png_of_exact_resolution(tmp_path, resolution):
    path = render_intro(SUMMARY, get_theme('midnight'), resolution,
                        tmp_path / 'intro.png')
    assert path.exists()
    assert png_size(path) == resolution


@pytest.mark.parametrize('resolution', RESOLUTIONS)
def test_outro_writes_png_of_exact_resolution(tmp_path, resolution):
    path = render_outro(FINAL_STATS, get_theme('midnight'), resolution,
                        tmp_path / 'outro.png')
    assert path.exists()
    assert png_size(path) == resolution


@pytest.mark.parametrize('name', list_themes())
def test_every_theme_renders_both_cards(tmp_path, name):
    theme = get_theme(name)
    intro = render_intro(SUMMARY, theme, LANDSCAPE, tmp_path / f'{name}_i.png')
    outro = render_outro(FINAL_STATS, theme, LANDSCAPE, tmp_path / f'{name}_o.png')
    assert png_size(intro) == LANDSCAPE
    assert png_size(outro) == LANDSCAPE


@pytest.mark.parametrize('name', list_themes())
def test_every_theme_renders_both_cards_portrait(tmp_path, name):
    theme = get_theme(name)
    intro = render_intro(SUMMARY, theme, PORTRAIT, tmp_path / f'{name}_i.png')
    outro = render_outro(FINAL_STATS, theme, PORTRAIT, tmp_path / f'{name}_o.png')
    assert png_size(intro) == PORTRAIT
    assert png_size(outro) == PORTRAIT


def test_intro_creates_missing_parent_directories(tmp_path):
    target = tmp_path / 'deep' / 'nested' / 'intro.png'
    assert render_intro(SUMMARY, get_theme('paper'), LANDSCAPE, target) == target
    assert target.exists()


def test_intro_without_a_title_falls_back_to_the_tickers(tmp_path):
    summary = {k: v for k, v in SUMMARY.items() if k != 'title'}
    path = render_intro(summary, get_theme('paper'), LANDSCAPE,
                        tmp_path / 'intro.png')
    assert png_size(path) == LANDSCAPE


def test_intro_handles_no_dca_and_cash_dividends(tmp_path):
    summary = dict(SUMMARY, dca_amount=0.0, dca_frequency='none',
                   reinvest_dividends=False, fractional_shares=False)
    path = render_intro(summary, get_theme('terminal'), LANDSCAPE,
                        tmp_path / 'intro.png')
    assert png_size(path) == LANDSCAPE


def test_intro_with_eight_tickers(tmp_path):
    summary = dict(SUMMARY, title=None,
                   tickers=['AAPL', 'MSFT', 'KO', 'VOO', 'TSLA', 'AMZN',
                            'NVDA', 'JNJ'])
    path = render_intro(summary, get_theme('midnight'), PORTRAIT,
                        tmp_path / 'intro.png')
    assert png_size(path) == PORTRAIT


def test_outro_with_a_nan_return_shows_a_dash(tmp_path):
    stats = [TickerStats(ticker='AAPL', value=1000.0, invested=1000.0,
                         total_return=0.0, xirr=float('nan'),
                         max_drawdown=0.0, cum_dividends=0.0, rank=1)]
    path = render_outro(stats, get_theme('midnight'), LANDSCAPE,
                        tmp_path / 'outro.png')
    assert png_size(path) == LANDSCAPE


def test_outro_uses_a_currency_symbol(tmp_path):
    path = render_outro(FINAL_STATS, get_theme('solarized'), LANDSCAPE,
                        tmp_path / 'outro.png', currency='THB')
    assert png_size(path) == LANDSCAPE


def pixels(path) -> bytes:
    with Image.open(path) as image:
        return image.convert('RGB').tobytes()


def test_outro_colour_follows_the_series_order_not_the_rank(tmp_path):
    """Reordering the input must not repaint a ticker when `order` is given."""
    theme = get_theme('midnight')
    order = [item.ticker for item in FINAL_STATS]
    shuffled = [FINAL_STATS[2], FINAL_STATS[0], FINAL_STATS[1]]
    first = render_outro(FINAL_STATS, theme, LANDSCAPE, tmp_path / 'a.png',
                         order=order)
    second = render_outro(shuffled, theme, LANDSCAPE, tmp_path / 'b.png',
                          order=order)
    assert pixels(first) == pixels(second)

    # Without an explicit order the input order is the palette order, so the
    # shuffled list genuinely paints the tickers differently.
    third = render_outro(shuffled, theme, LANDSCAPE, tmp_path / 'c.png')
    assert pixels(third) != pixels(first)
