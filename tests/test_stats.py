"""Metric tests. All offline."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from invest_viz.portfolio import simulate
from invest_viz.stats import compute_stats, max_drawdown, rank_stats, xirr
from invest_viz.types import (
    PRICE_COLUMNS,
    DcaFrequency,
    PortfolioSpec,
    PriceData,
    TickerStats,
)

START = pd.Timestamp('2020-01-01')


def make_stats(ticker: str, value: float) -> TickerStats:
    return TickerStats(ticker=ticker, value=value, invested=100.0,
                       total_return=0.0, xirr=0.0, max_drawdown=0.0,
                       cum_dividends=0.0)


def test_xirr_of_a_ten_percent_year():
    rate = xirr([(START, -1000.0), (START + pd.Timedelta(days=365), 1100.0)])

    assert rate == pytest.approx(0.10, abs=1e-3)


def test_xirr_of_a_flat_investment_is_zero():
    rate = xirr([(START, -1000.0), (START + pd.Timedelta(days=730), 1000.0)])

    assert rate == pytest.approx(0.0, abs=1e-6)


def test_xirr_of_a_loss_is_negative():
    rate = xirr([(START, -1000.0), (START + pd.Timedelta(days=365), 500.0)])

    assert rate == pytest.approx(-0.5, abs=1e-3)


def test_xirr_handles_several_contributions():
    flows = [(START, -1000.0),
             (START + pd.Timedelta(days=365), -1000.0),
             (START + pd.Timedelta(days=730), 2100.0)]
    rate = xirr(flows)

    assert math.isfinite(rate)
    assert 0.0 < rate < 0.1


@pytest.mark.parametrize('flows', [
    [],
    [(START, -1000.0)],
    [(START, -1000.0), (START + pd.Timedelta(days=365), -500.0)],
    [(START, 1000.0), (START + pd.Timedelta(days=365), 500.0)],
    [(START, 0.0), (START, 0.0)],
    [(START, float('nan')), (START, 1.0)],
])
def test_xirr_returns_nan_for_degenerate_input(flows):
    assert math.isnan(xirr(flows))


def test_xirr_never_raises_on_junk():
    assert math.isnan(xirr([('not-a-date', -1.0), (START, 1.0)]))
    assert math.isnan(xirr(None))


def test_max_drawdown_of_a_known_fifty_percent_fall():
    series = pd.Series([100.0, 200.0, 100.0, 150.0])

    assert max_drawdown(series) == pytest.approx(-0.5)


def test_max_drawdown_of_a_rising_series_is_zero():
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_max_drawdown_ignores_leading_zeros_and_nans():
    padded = pd.Series([0.0, 0.0, np.nan, 100.0, 200.0, 100.0, 150.0])

    assert max_drawdown(padded) == pytest.approx(-0.5)


def test_max_drawdown_of_a_degenerate_series_is_zero():
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0
    assert max_drawdown(pd.Series([0.0, 0.0])) == 0.0


def test_compute_stats_on_a_hand_built_history():
    index = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04'])
    frame = pd.DataFrame(
        {'close': [10.0, 20.0, 25.0], 'dividends': [0.0, 1.0, 0.0],
         'splits': [0.0, 0.0, 0.0]},
        index=index, columns=list(PRICE_COLUMNS))
    history = simulate(PriceData(ticker='TOY', frame=frame),
                       PortfolioSpec(initial_amount=100.0, dca_amount=0.0,
                                     reinvest_dividends=False))
    stats = compute_stats(history)

    assert stats.ticker == 'TOY'
    assert stats.value == pytest.approx(260.0)
    assert stats.invested == pytest.approx(100.0)
    assert stats.total_return == pytest.approx(1.6)
    assert stats.cum_dividends == pytest.approx(10.0)
    assert stats.max_drawdown == 0.0
    assert stats.rank == 0


def test_compute_stats_truncates_at_as_of(prices):
    history = simulate(prices('ko_dividends'),
                       PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0))
    as_of = pd.Timestamp('2022-06-30')
    stats = compute_stats(history, as_of=as_of)

    assert stats.value == pytest.approx(history.frame['value'].loc[:as_of].iloc[-1])
    assert stats.cum_dividends == pytest.approx(
        history.frame['cum_dividends'].loc[as_of])
    assert stats.value != history.frame['value'].iloc[-1]


def test_compute_stats_total_return_is_zero_when_nothing_is_invested():
    index = pd.DatetimeIndex(['2024-01-02', '2024-01-03'])
    frame = pd.DataFrame(
        {'close': [10.0, 20.0], 'dividends': [0.0, 0.0], 'splits': [0.0, 0.0]},
        index=index, columns=list(PRICE_COLUMNS))
    history = simulate(PriceData(ticker='TOY', frame=frame),
                       PortfolioSpec(initial_amount=0.0, dca_amount=0.0,
                                     dca_frequency=DcaFrequency.NONE))
    stats = compute_stats(history)

    assert stats.invested == 0.0
    assert stats.total_return == 0.0
    assert math.isnan(stats.xirr)


def test_compute_stats_xirr_is_plausible_for_a_real_run(prices):
    history = simulate(prices('aapl_10y'),
                       PortfolioSpec(initial_amount=10_000.0, dca_amount=500.0,
                                     dca_frequency=DcaFrequency.MONTHLY))
    stats = compute_stats(history)

    assert math.isfinite(stats.xirr)
    assert 0.0 < stats.xirr < 1.0
    assert stats.max_drawdown < 0.0
    assert stats.value > stats.invested


def test_rank_stats_orders_by_value_descending():
    ranked = rank_stats([make_stats('KO', 100.0), make_stats('AAPL', 300.0),
                         make_stats('MSFT', 200.0)])

    assert [s.ticker for s in ranked] == ['AAPL', 'MSFT', 'KO']
    assert [s.rank for s in ranked] == [1, 2, 3]


def test_rank_stats_breaks_ties_by_ticker_name():
    ranked = rank_stats([make_stats('MSFT', 100.0), make_stats('AAPL', 100.0)])

    assert [s.ticker for s in ranked] == ['AAPL', 'MSFT']
    assert [s.rank for s in ranked] == [1, 2]


def test_rank_stats_does_not_mutate_its_input():
    original = [make_stats('KO', 100.0), make_stats('AAPL', 300.0)]
    ranked = rank_stats(original)

    assert [s.rank for s in original] == [0, 0]
    assert all(new is not old for new in ranked for old in original)


def test_rank_stats_of_an_empty_list():
    assert rank_stats([]) == []
