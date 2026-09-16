#!/usr/bin/env python3
"""Generate a video comparing how N investments would have performed."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from investment_video.config import build_config, build_parser  # noqa: E402


def main(argv: list = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if '--list-themes' in argv:
        from investment_video.themes import get_theme, list_themes
        for name in list_themes():
            theme = get_theme(name)
            print(f'{name:<10} {len(theme.series)} series  bg {theme.background}')
        return 0

    try:
        config = build_config(argv)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from investment_video.generator import run
    try:
        output = run(config)
    except (RuntimeError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    print(f'Done: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
