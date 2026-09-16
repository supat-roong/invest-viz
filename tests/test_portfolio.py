"""Simulation tests. All offline, against committed CSV fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from invest_viz.portfolio import contribution_dates, simulate
from invest_viz.types import (
    PRICE_COLUMNS,
    DcaFrequency,
    EventKind,
    PortfolioSpec,
    PriceData,
)

SPLIT_DATE = pd.Timestamp('2020-08-31')
LUMP_SUM = PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                         dca_frequency=DcaFrequency.NONE)


def toy_prices(closes, dividends=None, splits=None, ticker='TOY') -> PriceData:
    """A hand-sized PriceData on consecutive January 2024 trading days."""
    index = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04'][:len(closes)])
    frame = pd.DataFrame(
        {
            'close': closes,
            'dividends': dividends or [0.0] * len(closes),
            'splits': splits or [0.0] * len(closes),
        },
        index=index,
        columns=list(PRICE_COLUMNS),
    ).astype(float)
    frame.index.name = 'date'
    return PriceData(ticker=ticker, frame=frame)


def test_split_does_not_multiply_share_count(prices):
    """Regression: close is already split-adjusted, so shares must not 4x."""
    history = simulate(prices('aapl_2020_split'), LUMP_SUM)
    shares = history.frame['shares']

    before = shares.loc[pd.Timestamp('2020-08-28')]
    after = shares.loc[SPLIT_DATE]

    assert before > 0
    assert after / before == pytest.approx(1.0, abs=1e-12)
    assert shares.max() / shares.min() < 1.5


def test_split_is_recorded_as_a_single_event(prices):
    history = simulate(prices('aapl_2020_split'), LUMP_SUM)
    splits = [e for e in history.events if e.kind is EventKind.SPLIT]

    assert len(splits) == 1
    assert splits[0].date == SPLIT_DATE
    assert splits[0].value == 4.0
    assert splits[0].ticker == 'AAPL'


def test_value_is_continuous_across_the_split(prices):
    history = simulate(prices('aapl_2020_split'), LUMP_SUM)
    value = history.frame['value']

    ratio = value.loc[SPLIT_DATE] / value.loc[pd.Timestamp('2020-08-28')]
    assert 0.9 < ratio < 1.1


def test_events_are_chronological(prices):
    history = simulate(prices('aapl_2020_split'), LUMP_SUM)
    dates = [event.date for event in history.events]

    assert dates == sorted(dates)
    assert any(event.kind is EventKind.DIVIDEND for event in history.events)


def test_reinvesting_dividends_beats_taking_them_as_cash(prices):
    data = prices('ko_dividends')
    drip = simulate(data, PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                                        reinvest_dividends=True))
    cash = simulate(data, PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                                        reinvest_dividends=False))

    assert drip.frame['shares'].iloc[-1] > cash.frame['shares'].iloc[-1]
    assert drip.frame['value'].iloc[-1] > cash.frame['value'].iloc[-1]
    assert drip.frame['cash'].iloc[-1] == 0.0


def test_cash_dividend_mode_holds_every_dividend_as_cash(prices):
    history = simulate(prices('ko_dividends'),
                       PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                                     reinvest_dividends=False))
    frame = history.frame

    assert frame['cash'].iloc[-1] == frame['cum_dividends'].iloc[-1]
    assert frame['cum_dividends'].iloc[-1] > 0
    assert frame['shares'].nunique() == 1


def test_cum_dividends_matches_across_modes_at_the_first_dividend(prices):
    data = prices('ko_dividends')
    first_dividend = data.dividends[data.dividends > 0].index[0]
    drip = simulate(data, PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                                        reinvest_dividends=True))
    cash = simulate(data, PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                                        reinvest_dividends=False))

    assert first_dividend == pd.Timestamp('2022-03-14')
    assert (drip.frame['cum_dividends'].loc[first_dividend]
            == cash.frame['cum_dividends'].loc[first_dividend])


def test_dividend_events_carry_the_amount_received(prices):
    data = prices('ko_dividends')
    history = simulate(data, PortfolioSpec(initial_amount=10_000.0, dca_amount=0.0,
                                           reinvest_dividends=False))
    dividends = [e for e in history.events if e.kind is EventKind.DIVIDEND]

    assert len(dividends) == int((data.dividends > 0).sum())
    total = sum(event.value for event in dividends)
    assert total == pytest.approx(history.frame['cum_dividends'].iloc[-1])


def test_monthly_dca_snaps_forward_off_a_closed_market(prices):
    index = prices('ko_dividends').dates
    due = contribution_dates(index, DcaFrequency.MONTHLY, pd.Timestamp('2022-01-01'))

    assert pd.Timestamp('2022-01-01').day_name() == 'Saturday'
    assert due[0] == pd.Timestamp('2022-01-03')
    assert len(due) == 24  # every month from 2022-01 through 2023-12
    assert due.is_unique
    assert due.equals(due.sort_values())
    assert index.isin(due).sum() == len(due)


def test_contribution_dates_drops_targets_past_the_index():
    index = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04'])
    due = contribution_dates(index, DcaFrequency.YEARLY, pd.Timestamp('2024-01-02'))

    assert list(due) == [pd.Timestamp('2024-01-02')]


def test_contribution_dates_is_empty_for_frequency_none():
    index = pd.DatetimeIndex(['2024-01-02', '2024-01-03'])
    due = contribution_dates(index, DcaFrequency.NONE, pd.Timestamp('2024-01-02'))

    assert len(due) == 0


def test_daily_dca_never_returns_a_duplicate_date(prices):
    index = prices('ko_dividends').dates
    due = contribution_dates(index, DcaFrequency.DAILY, index[0])

    assert due.is_unique
    assert due.equals(index)


def test_dca_invests_once_per_scheduled_day(prices):
    data = prices('ko_dividends')
    spec = PortfolioSpec(initial_amount=0.0, dca_amount=500.0,
                         dca_frequency=DcaFrequency.MONTHLY)
    history = simulate(data, spec)

    due = contribution_dates(data.dates, DcaFrequency.MONTHLY, data.first_trading_day)
    assert history.frame['invested'].iloc[-1] == pytest.approx(500.0 * len(due))


def test_zero_dividend_window_leaves_value_as_shares_times_price(prices):
    data = prices('msft_10y')
    window = data.frame.loc['2015-01-02':'2015-02-13']
    assert (window['dividends'] == 0).all()

    history = simulate(PriceData(ticker='MSFT', frame=window), LUMP_SUM)
    frame = history.frame

    assert (frame['cum_dividends'] == 0).all()
    assert frame['cash'].eq(0.0).all()
    np.testing.assert_allclose(frame['value'], frame['shares'] * frame['price'])
    assert not [e for e in history.events if e.kind is EventKind.DIVIDEND]


def test_whole_share_mode_carries_a_cash_remainder(prices):
    data = prices('aapl_10y')
    spec = PortfolioSpec(initial_amount=10_000.0, dca_amount=500.0,
                         dca_frequency=DcaFrequency.MONTHLY,
                         reinvest_dividends=False, fractional_shares=False)
    whole = simulate(data, spec)
    fractional = simulate(data, PortfolioSpec(
        initial_amount=10_000.0, dca_amount=500.0,
        dca_frequency=DcaFrequency.MONTHLY, reinvest_dividends=False,
        fractional_shares=True))

    shares = whole.frame['shares']
    assert np.all(shares == np.floor(shares))
    assert whole.frame['cash'].iloc[-1] > whole.frame['cum_dividends'].iloc[-1]
    assert (whole.frame['invested'].iloc[-1]
            == pytest.approx(fractional.frame['invested'].iloc[-1]))
    assert whole.frame['value'].iloc[-1] != fractional.frame['value'].iloc[-1]


def test_whole_share_remainder_accumulates_into_a_later_purchase():
    """15 then 15 of cash at a price of 10 must buy a share on the second day."""
    data = toy_prices([10.0, 10.0, 10.0])
    spec = PortfolioSpec(initial_amount=0.0, dca_amount=15.0,
                         dca_frequency=DcaFrequency.DAILY,
                         reinvest_dividends=False, fractional_shares=False)
    frame = simulate(data, spec).frame

    assert list(frame['shares']) == [1.0, 3.0, 4.0]
    assert list(frame['cash']) == [5.0, 0.0, 5.0]
    assert list(frame['invested']) == [15.0, 30.0, 45.0]
    assert list(frame['value']) == [15.0, 30.0, 45.0]


def test_toy_three_days_with_dividends_reinvested():
    """By hand: 100 buys 10 shares at 10; a 1.00 dividend buys 0.5 more at 20."""
    data = toy_prices([10.0, 20.0, 25.0], dividends=[0.0, 1.0, 0.0])
    frame = simulate(data, PortfolioSpec(initial_amount=100.0, dca_amount=0.0,
                                         reinvest_dividends=True)).frame

    assert list(frame['shares']) == [10.0, 10.5, 10.5]
    assert list(frame['cash']) == [0.0, 0.0, 0.0]
    assert list(frame['invested']) == [100.0, 100.0, 100.0]
    assert list(frame['cum_dividends']) == [0.0, 10.0, 10.0]
    assert list(frame['value']) == [100.0, 210.0, 262.5]


def test_toy_three_days_with_dividends_as_cash():
    """Same bars, but the 10.00 of dividend sits in cash instead of buying."""
    data = toy_prices([10.0, 20.0, 25.0], dividends=[0.0, 1.0, 0.0])
    frame = simulate(data, PortfolioSpec(initial_amount=100.0, dca_amount=0.0,
                                         reinvest_dividends=False)).frame

    assert list(frame['shares']) == [10.0, 10.0, 10.0]
    assert list(frame['cash']) == [0.0, 10.0, 10.0]
    assert list(frame['invested']) == [100.0, 100.0, 100.0]
    assert list(frame['cum_dividends']) == [0.0, 10.0, 10.0]
    assert list(frame['value']) == [100.0, 210.0, 260.0]


def test_contribution_is_bought_before_the_same_day_dividend():
    """Ordering check: day one's dividend is paid on shares bought that day."""
    data = toy_prices([10.0, 10.0], dividends=[2.0, 0.0])
    frame = simulate(data, PortfolioSpec(initial_amount=100.0, dca_amount=0.0,
                                         reinvest_dividends=False)).frame

    assert frame['cum_dividends'].iloc[0] == 20.0


def test_frame_shape_and_dtypes(prices):
    history = simulate(prices('ko_dividends'), LUMP_SUM)

    assert list(history.frame.columns) == [
        'price', 'shares', 'cash', 'invested', 'cum_dividends', 'value']
    assert history.frame.index.equals(prices('ko_dividends').dates)
    assert all(dtype == np.float64 for dtype in history.frame.dtypes)
    assert history.spec is LUMP_SUM


def test_ticker_that_ipos_mid_range_is_dormant_until_its_first_close():
    index = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04'])
    frame = pd.DataFrame(
        {'close': [np.nan, 20.0, 25.0], 'dividends': [0.0, 0.0, 0.0],
         'splits': [0.0, 0.0, 0.0]},
        index=index, columns=list(PRICE_COLUMNS))
    data = PriceData(ticker='IPO', frame=frame)

    assert data.first_trading_day == pd.Timestamp('2024-01-03')
    result = simulate(data, PortfolioSpec(initial_amount=100.0, dca_amount=0.0)).frame

    assert result['shares'].iloc[0] == 0.0
    assert result['invested'].iloc[0] == 0.0
    assert result['value'].iloc[0] == 0.0
    assert result['shares'].iloc[1] == 5.0
    assert result['value'].iloc[2] == 125.0
