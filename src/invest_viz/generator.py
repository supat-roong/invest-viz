"""Wire the pipeline together: fetch, simulate, render, encode."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import stats as stats_mod
from . import timeline
from .assembler import check_ffmpeg, frames_to_video
from .cache import PriceCache
from .cards import render_intro, render_outro
from .config import Config
from .fetcher import fetch
from .portfolio import simulate
from .renderer import ChartRenderer
from .themes import get_theme
from .types import PortfolioSpec, DcaFrequency

log = logging.getLogger(__name__)

FRAME_PATTERN = 'frame_%05d.png'


def build_spec(config: Config) -> PortfolioSpec:
    pf = config.portfolio
    return PortfolioSpec(
        initial_amount=pf.initial_amount,
        dca_amount=pf.dca_amount,
        dca_frequency=DcaFrequency(pf.dca_frequency),
        reinvest_dividends=pf.reinvest_dividends,
        fractional_shares=pf.fractional_shares,
    )


def _summary(config: Config, currency: str) -> dict:
    """Key names here are the contract cards.render_intro reads."""
    pf = config.portfolio
    return {
        'title': 'Investment Comparison',
        'tickers': list(config.data.tickers),
        'start': config.data.start,
        'end': config.data.end,
        'initial_amount': pf.initial_amount,
        'dca_amount': pf.dca_amount,
        'dca_frequency': pf.dca_frequency,
        'reinvest_dividends': pf.reinvest_dividends,
        'fractional_shares': pf.fractional_shares,
        'currency': currency,
    }


def outro_stats(histories: list) -> list:
    """Ranked stats, returned in ticker order rather than rank order.

    cards.render_outro takes each series colour from the position of the item in
    the list it is handed. Handing it rank order would colour tickers by their
    finishing position, so a ticker would change colour between the chart and
    the leaderboard.
    """
    ranked = stats_mod.rank_stats([stats_mod.compute_stats(h) for h in histories])
    by_ticker = {item.ticker: item for item in ranked}
    return [by_ticker[h.ticker] for h in histories if h.ticker in by_ticker]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink a card frame rather than copying it hundreds of times."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def run(config: Config, keep_frames: bool = False) -> Path:
    check_ffmpeg()

    theme = get_theme(config.theme.name)
    spec = build_spec(config)
    start = pd.Timestamp(config.data.start)
    end = pd.Timestamp(config.data.end)

    cache = PriceCache(Path(config.data.cache_dir), config.data.cache_max_age_days)
    log.info('fetching %d tickers %s..%s', len(config.data.tickers),
             config.data.start, config.data.end)
    prices = fetch(config.data.tickers, start, end, cache=cache)

    currencies = {p.currency for p in prices.values()}
    currency = currencies.pop() if len(currencies) == 1 else 'USD'

    histories = []
    for ticker in config.data.tickers:
        if ticker not in prices:
            continue
        histories.append(simulate(prices[ticker], spec))
    if not histories:
        raise RuntimeError('no ticker produced a simulation')
    log.info('simulated %d tickers over %d bars', len(histories),
             len(histories[0].frame))

    states = timeline.build_states(
        histories,
        fps=config.video.fps,
        duration_seconds=config.video.duration_seconds,
        end_hold_seconds=config.video.end_hold_seconds,
    )
    log.info('built %d frame states', len(states))

    resolution = config.video.resolution
    fps = config.video.fps
    out_dir = Path(config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix='icvg_', dir='tmp' if Path('tmp').is_dir() else None))
    frames = work / 'frames'
    frames.mkdir()
    n = 0
    t0 = time.time()
    try:
        intro_frames = int(round(fps * config.video.intro_seconds))
        if config.panels.intro_card and intro_frames:
            card = work / 'intro.png'
            render_intro(_summary(config, currency), theme, resolution, card)
            for _ in range(intro_frames):
                _link_or_copy(card, frames / (FRAME_PATTERN % n))
                n += 1
            log.info('intro card: %d frames', intro_frames)

        renderer = ChartRenderer(theme, config.panels, resolution, currency=currency)
        try:
            for i, state in enumerate(states):
                renderer.render(state, frames / (FRAME_PATTERN % n))
                n += 1
                if i and i % 200 == 0:
                    log.info('rendered %d/%d chart frames', i, len(states))
        finally:
            renderer.close()

        outro_frames = int(round(fps * config.video.outro_seconds))
        if config.panels.outro_card and outro_frames:
            final = outro_stats(histories)
            card = work / 'outro.png'
            render_outro(final, theme, resolution, card, currency=currency,
                         order=[h.ticker for h in histories])
            for _ in range(outro_frames):
                _link_or_copy(card, frames / (FRAME_PATTERN % n))
                n += 1
            log.info('outro card: %d frames', outro_frames)

        log.info('rendered %d frames in %.1fs (%.0f ms/frame)',
                 n, time.time() - t0, 1000 * (time.time() - t0) / max(n, 1))

        output = out_dir / f'{config.output.name}.mp4'
        frames_to_video(frames, fps, output)
        log.info('encoded %s', output)
        return output
    finally:
        if keep_frames:
            log.info('frames kept at %s', frames)
        else:
            shutil.rmtree(work, ignore_errors=True)
