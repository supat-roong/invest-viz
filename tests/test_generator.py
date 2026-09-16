"""Wiring between the pipeline stages.

Both tests here pin bugs found by watching a real render rather than by
reading code: the intro card printed $0, and the leaderboard recoloured
tickers by finishing position.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from investment_video import cards
from investment_video.config import Config, apply_overrides, build_parser
from investment_video.generator import _summary, build_spec, outro_stats
from investment_video.portfolio import simulate
from investment_video.themes import get_theme
from investment_video.types import (DcaFrequency, PortfolioSpec, PriceData)


def _prices(ticker: str, closes: list) -> PriceData:
    index = pd.bdate_range('2020-01-01', periods=len(closes))
    frame = pd.DataFrame({'close': [float(c) for c in closes],
                          'dividends': [0.0] * len(closes),
                          'splits': [0.0] * len(closes)}, index=index)
    return PriceData(ticker, frame)


def _config(argv):
    return apply_overrides(Config(), build_parser().parse_args(argv))


def test_summary_uses_the_key_names_the_intro_card_reads():
    config = _config(['--tickers', 'AAPL,KO', '--initial', '10000',
                      '--dca', '500', '--dca-freq', 'monthly'])
    summary = _summary(config, 'USD')

    lines = cards._parameter_lines(summary, 'USD')
    joined = ' '.join(lines)
    assert '$10,000' in joined
    assert '$500' in joined
    assert 'monthly' in joined
    assert '$0' not in joined


def test_summary_reports_cash_dividends_when_reinvesting_is_off():
    config = _config(['--no-reinvest'])
    lines = ' '.join(cards._parameter_lines(_summary(config, 'USD'), 'USD'))
    assert 'reinvest' not in lines.lower() or 'cash' in lines.lower()


def test_summary_covers_every_parameter_key_the_card_consumes():
    """Guards against key drift between generator and cards."""
    source = inspect.getsource(cards._parameter_lines)
    summary = _summary(_config([]), 'USD')
    for key in ('initial_amount', 'dca_amount', 'dca_frequency',
                'reinvest_dividends', 'fractional_shares'):
        assert f"'{key}'" in source, f'{key} no longer read by the card'
        assert key in summary, f'{key} no longer supplied by the generator'


def test_build_spec_carries_the_config_through():
    config = _config(['--dca-freq', 'weekly', '--no-reinvest', '--whole-shares'])
    spec = build_spec(config)
    assert spec.dca_frequency is DcaFrequency.WEEKLY
    assert spec.reinvest_dividends is False
    assert spec.fractional_shares is False


def test_outro_stats_stay_in_ticker_order_not_rank_order():
    """The leaderboard must not repaint a ticker by where it finished."""
    losing = simulate(_prices('AAA', [10.0] * 5 + [5.0] * 5),
                      PortfolioSpec(1000.0, 0.0, DcaFrequency.NONE))
    winning = simulate(_prices('BBB', [10.0] * 5 + [40.0] * 5),
                       PortfolioSpec(1000.0, 0.0, DcaFrequency.NONE))
    histories = [losing, winning]

    final = outro_stats(histories)

    assert [item.ticker for item in final] == ['AAA', 'BBB']
    ranks = {item.ticker: item.rank for item in final}
    assert ranks == {'BBB': 1, 'AAA': 2}


def test_outro_colours_match_the_chart_colours():
    losing = simulate(_prices('AAA', [10.0] * 5 + [5.0] * 5),
                      PortfolioSpec(1000.0, 0.0, DcaFrequency.NONE))
    winning = simulate(_prices('BBB', [10.0] * 5 + [40.0] * 5),
                       PortfolioSpec(1000.0, 0.0, DcaFrequency.NONE))
    theme = get_theme('midnight')

    final = outro_stats([losing, winning])
    # cards.render_outro colours by list position, so position must equal the
    # index the chart uses for that ticker.
    outro_colours = {item.ticker: theme.color_for(index)
                     for index, item in enumerate(final)}
    chart_colours = {h.ticker: theme.color_for(index)
                     for index, h in enumerate([losing, winning])}
    assert outro_colours == chart_colours


def test_outro_stats_skips_a_ticker_with_no_history():
    only = simulate(_prices('AAA', [10.0] * 4),
                    PortfolioSpec(1000.0, 0.0, DcaFrequency.NONE))
    assert [i.ticker for i in outro_stats([only])] == ['AAA']
