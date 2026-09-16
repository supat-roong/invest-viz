"""Config loading, CLI precedence and the validation gate."""
from __future__ import annotations

from pathlib import Path

import pytest

from invest_viz.config import (ASPECT_RESOLUTIONS, Config, apply_overrides,
                                     build_config, build_parser, load_config,
                                     validate)


def parse(argv):
    return build_parser().parse_args(argv)


def test_defaults_are_usable_without_a_config_file(tmp_path):
    config = load_config(tmp_path / 'missing.toml')
    validate(config)
    assert config.data.tickers


def test_toml_values_win_over_dataclass_defaults(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '[data]\ntickers = ["VOO"]\nstart = "2020-01-01"\nend = "2021-01-01"\n'
        '[theme]\nname = "neon"\n')
    config = load_config(path)
    assert config.data.tickers == ['VOO']
    assert config.theme.name == 'neon'


def test_cli_overrides_win_over_toml():
    config = apply_overrides(Config(), parse(
        ['--tickers', 'aapl, msft', '--theme', 'paper', '--initial', '500',
         '--dca', '25', '--dca-freq', 'weekly', '--aspect', '9:16']))
    assert config.data.tickers == ['AAPL', 'MSFT']
    assert config.theme.name == 'paper'
    assert config.portfolio.initial_amount == 500
    assert config.portfolio.dca_amount == 25
    assert config.portfolio.dca_frequency == 'weekly'
    assert config.video.aspect == '9:16'


def test_flags_invert_the_boolean_defaults():
    config = apply_overrides(Config(), parse(['--no-reinvest', '--whole-shares']))
    assert config.portfolio.reinvest_dividends is False
    assert config.portfolio.fractional_shares is False


def test_no_cache_flag_forces_every_entry_stale():
    config = apply_overrides(Config(), parse(['--no-cache']))
    assert config.data.cache_max_age_days == 0


def test_resolution_follows_the_aspect():
    assert Config().video.resolution == ASPECT_RESOLUTIONS['16:9']
    portrait = apply_overrides(Config(), parse(['--aspect', '9:16']))
    assert portrait.video.resolution == (1080, 1920)


def test_output_path_combines_dir_and_name():
    config = apply_overrides(Config(), parse(
        ['--output', 'race', '--output-dir', 'videos']))
    assert config.output_path == Path('videos/race.mp4')


@pytest.mark.parametrize('argv, fragment', [
    (['--tickers', ','], 'no tickers'),
    (['--tickers', 'AAPL,AAPL'], 'duplicate'),
    (['--start', '2025-01-01', '--end', '2020-01-01'], 'not before'),
    (['--theme', 'vaporwave'], 'unknown theme'),
    (['--fps', '0'], 'fps must be positive'),
    (['--duration', '0'], 'duration_seconds must be positive'),
    (['--initial', '-5'], 'must not be negative'),
    (['--initial', '0', '--dca', '0'], 'nothing is ever invested'),
    (['--tickers', 'A,B,C,D,E,F,G,H,I'], 'distinguishable colours'),
])
def test_validation_rejects_bad_input(argv, fragment):
    with pytest.raises(ValueError) as excinfo:
        validate(apply_overrides(Config(), parse(argv)))
    assert fragment in str(excinfo.value)


def test_unknown_dca_frequency_lists_the_valid_ones():
    config = Config()
    config.portfolio.dca_frequency = 'fortnightly'
    with pytest.raises(ValueError) as excinfo:
        validate(config)
    assert 'monthly' in str(excinfo.value)


def test_build_config_returns_a_validated_config(tmp_path):
    path = tmp_path / 'c.toml'
    path.write_text('[data]\ntickers = ["KO"]\n')
    config = build_config(['--config', str(path), '--theme', 'terminal'])
    assert config.data.tickers == ['KO']
    assert config.theme.name == 'terminal'
