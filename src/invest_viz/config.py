"""TOML defaults plus CLI overrides, validated into a Config."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

from .types import DcaFrequency

REPO_ROOT = Path(__file__).resolve().parents[2]

ASPECT_RESOLUTIONS = {
    '16:9': (1920, 1080),
    '9:16': (1080, 1920),
}


@dataclass
class DataConfig:
    tickers: list = field(default_factory=lambda: ['AAPL', 'MSFT', 'KO'])
    start: str = '2015-01-01'
    end: str = '2025-01-01'
    cache_dir: str = '.cache'
    cache_max_age_days: int = 1


@dataclass
class PortfolioConfig:
    initial_amount: float = 10_000.0
    dca_amount: float = 500.0
    dca_frequency: str = 'monthly'
    reinvest_dividends: bool = True
    fractional_shares: bool = True


@dataclass
class VideoConfig:
    fps: int = 30
    duration_seconds: float = 45.0
    aspect: str = '16:9'
    end_hold_seconds: float = 3.0
    intro_seconds: float = 2.5
    outro_seconds: float = 4.0

    @property
    def resolution(self) -> tuple:
        return ASPECT_RESOLUTIONS[self.aspect]


@dataclass
class ThemeConfig:
    name: str = 'midnight'


@dataclass
class PanelsConfig:
    stats: bool = True
    invested_line: bool = True
    dividend_line: bool = True
    event_markers: bool = True
    intro_card: bool = True
    outro_card: bool = True


@dataclass
class OutputConfig:
    name: str = 'comparison'
    output_dir: str = 'output'


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    panels: PanelsConfig = field(default_factory=PanelsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output.output_dir) / f'{self.output.name}.mp4'


def load_config(path: Path = None) -> Config:
    if path is None:
        path = REPO_ROOT / 'config.toml'
    if not Path(path).exists():
        return Config()
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    return Config(
        data=DataConfig(**data.get('data', {})),
        portfolio=PortfolioConfig(**data.get('portfolio', {})),
        video=VideoConfig(**data.get('video', {})),
        theme=ThemeConfig(**data.get('theme', {})),
        panels=PanelsConfig(**data.get('panels', {})),
        output=OutputConfig(**data.get('output', {})),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='invest-viz',
        description='Animate an N-way investment comparison as a silent MP4.',
    )
    p.add_argument('--config', type=Path, help='path to config.toml')
    p.add_argument('--tickers', help='comma-separated, e.g. AAPL,MSFT,KO')
    p.add_argument('--start', help='YYYY-MM-DD')
    p.add_argument('--end', help='YYYY-MM-DD')
    p.add_argument('--initial', type=float, help='lump sum at the start date')
    p.add_argument('--dca', type=float, help='recurring contribution amount')
    p.add_argument('--dca-freq', choices=[f.value for f in DcaFrequency])
    p.add_argument('--no-reinvest', action='store_true',
                   help='take dividends as cash instead of reinvesting')
    p.add_argument('--whole-shares', action='store_true',
                   help='disallow fractional shares')
    p.add_argument('--theme')
    p.add_argument('--list-themes', action='store_true')
    p.add_argument('--aspect', choices=sorted(ASPECT_RESOLUTIONS))
    p.add_argument('--fps', type=int)
    p.add_argument('--duration', type=float, help='seconds of animation')
    p.add_argument('--output', help='output file name without extension')
    p.add_argument('--output-dir')
    p.add_argument('--no-cache', action='store_true')
    return p


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    data, pf = config.data, config.portfolio
    video, theme, output = config.video, config.theme, config.output

    if args.tickers:
        data = replace(data, tickers=[t.strip().upper()
                                      for t in args.tickers.split(',') if t.strip()])
    if args.start:
        data = replace(data, start=args.start)
    if args.end:
        data = replace(data, end=args.end)
    if args.no_cache:
        data = replace(data, cache_max_age_days=0)

    if args.initial is not None:
        pf = replace(pf, initial_amount=args.initial)
    if args.dca is not None:
        pf = replace(pf, dca_amount=args.dca)
    if args.dca_freq:
        pf = replace(pf, dca_frequency=args.dca_freq)
    if args.no_reinvest:
        pf = replace(pf, reinvest_dividends=False)
    if args.whole_shares:
        pf = replace(pf, fractional_shares=False)

    if args.aspect:
        video = replace(video, aspect=args.aspect)
    if args.fps is not None:
        video = replace(video, fps=args.fps)
    if args.duration is not None:
        video = replace(video, duration_seconds=args.duration)

    if args.theme:
        theme = replace(theme, name=args.theme)

    if args.output:
        output = replace(output, name=args.output)
    if args.output_dir:
        output = replace(output, output_dir=args.output_dir)

    return Config(data=data, portfolio=pf, video=video, theme=theme,
                  panels=config.panels, output=output)


def validate(config: Config) -> None:
    """Raise ValueError describing the first problem found."""
    from .themes import list_themes  # local import keeps types/config cycle-free

    d, pf, v = config.data, config.portfolio, config.video

    if not d.tickers:
        raise ValueError('no tickers given')
    if len(set(d.tickers)) != len(d.tickers):
        raise ValueError(f'duplicate tickers: {d.tickers}')

    import pandas as pd
    try:
        start, end = pd.Timestamp(d.start), pd.Timestamp(d.end)
    except Exception as e:
        raise ValueError(f'unparseable date: {e}') from e
    if start >= end:
        raise ValueError(f'start {d.start} is not before end {d.end}')

    try:
        DcaFrequency(pf.dca_frequency)
    except ValueError:
        raise ValueError(
            f'unknown dca_frequency {pf.dca_frequency!r}; expected one of '
            f'{[f.value for f in DcaFrequency]}') from None

    if pf.initial_amount < 0 or pf.dca_amount < 0:
        raise ValueError('investment amounts must not be negative')
    if pf.initial_amount == 0 and (
            pf.dca_amount == 0 or pf.dca_frequency == DcaFrequency.NONE.value):
        raise ValueError('nothing is ever invested: set initial_amount or a '
                         'dca_amount with a dca_frequency')

    if v.aspect not in ASPECT_RESOLUTIONS:
        raise ValueError(f'unknown aspect {v.aspect!r}; expected one of '
                         f'{sorted(ASPECT_RESOLUTIONS)}')
    if v.fps <= 0:
        raise ValueError('fps must be positive')
    if v.duration_seconds <= 0:
        raise ValueError('duration_seconds must be positive')
    if v.end_hold_seconds < 0:
        raise ValueError('end_hold_seconds must not be negative')
    if v.intro_seconds < 0 or v.outro_seconds < 0:
        raise ValueError('card durations must not be negative')

    known = list_themes()
    if config.theme.name not in known:
        raise ValueError(f'unknown theme {config.theme.name!r}; available: '
                         f'{", ".join(known)}')

    from .themes import MAX_SERIES
    if len(d.tickers) > MAX_SERIES:
        raise ValueError(
            f'{len(d.tickers)} tickers requested but the validated palettes carry '
            f'{MAX_SERIES} distinguishable colours; pick at most {MAX_SERIES}')


def build_config(argv: list = None) -> Config:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    validate(config)
    return config
