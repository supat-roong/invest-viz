# invest-viz

Generates silent MP4 videos comparing how N investments would actually have performed over a date range, using Yahoo Finance data via `yfinance`.

Each ticker is simulated as a real portfolio rather than a normalized price line: an optional lump sum at the start, optional recurring DCA contributions, and dividends either reinvested or taken as cash. The video animates the portfolio value lines racing left to right against a dashed "money invested" baseline, with a live stats panel, an intro card and a final leaderboard. Every line is direct-labelled at its head with its current amount, including the invested baseline and each dividend line.

## Pipeline

```
config.toml + CLI flags
       |
       v
   config.py ---> Config
       |
       v
   fetcher.py <---> cache.py          (.cache/<ticker>.parquet)
       |  PriceData: close, dividends, splits, aligned calendar
       v
  portfolio.py ---> stats.py          pure, zero I/O
       |  PortfolioHistory: value, invested, shares, cash, cum_dividends
       v
  timeline.py                         dates -> FrameState list
       |
       v
  renderer.py + themes.py + cards.py
       |  frames/frame_%05d.png
       v
  assembler.py (ffmpeg)  --->  output/<name>.mp4  (silent)
```

`generator.py` orchestrates the above. Everything upstream of the renderer is pure and network free except `fetcher.py`, so the whole simulation can be tested without touching Yahoo Finance or ffmpeg.

## Setup

```bash
git clone https://github.com/supat-roong/invest-viz.git
cd invest-viz
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

## Usage

```bash
python main.py                                              # config.toml defaults
python main.py --tickers AAPL,MSFT,KO --start 2015-01-01
python main.py --initial 10000 --dca 500 --dca-freq monthly
python main.py --no-reinvest                                # dividends as cash
python main.py --theme neon --aspect 9:16                   # vertical reel
python main.py --list-themes
```

`--output` sets the file name, `--output-dir` sets the directory. The video always lands at `<output_dir>/<name>.mp4` and never has an audio stream.

## How the simulation works

On each trading day, in this order:

1. Contribution. The initial amount on the first bar, the DCA amount on scheduled bars. Contributions snap forward to the next trading day, so one landing on a market holiday is never silently dropped.
2. Dividend. Shares held times the per share dividend, always accumulated into the dividends total. It then either buys more shares or lands in cash, depending on `reinvest_dividends`.
3. Value. Shares times close, plus any uninvested cash.

Because contributions arrive over time, plain CAGR on portfolio value cannot tell a large early contribution apart from a good return. The headline metric is therefore money weighted return (XIRR), shown as MWR.

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.12 | 3.10 and 3.11 should work via the `tomli` fallback, untested |
| `ffmpeg` and `ffprobe` on PATH | `brew install ffmpeg` or `apt install ffmpeg` |
| Network access | First run only per ticker, afterwards served from `.cache/` |

## Configuration

Every key below can be overridden on the command line.

| Section | Key | Default | Meaning |
|---|---|---|---|
| data | `tickers` | AAPL, MSFT, KO | Up to 8, the validated palette size |
| data | `start` / `end` | 2015-01-01 / 2025-01-01 | Comparison window |
| data | `cache_max_age_days` | 1 | Age before a cached ticker is refetched |
| portfolio | `initial_amount` | 10000 | Lump sum on the first bar |
| portfolio | `dca_amount` | 500 | Recurring contribution |
| portfolio | `dca_frequency` | monthly | none, daily, weekly, biweekly, monthly, quarterly, yearly |
| portfolio | `reinvest_dividends` | true | False pays dividends into cash instead |
| portfolio | `fractional_shares` | true | False buys whole shares and carries the remainder |
| video | `fps` | 30 | Frames per second |
| video | `duration_seconds` | 45 | Length of the animated section |
| video | `aspect` | 16:9 | 16:9 gives 1920x1080, 9:16 gives 1080x1920 |
| video | `end_hold_seconds` | 3 | Freeze on the final frame |
| video | `intro_seconds` / `outro_seconds` | 2.5 / 4 | Card durations |
| theme | `name` | midnight | See the theme table below |
| panels | various | true | Toggle the stats panel, lines, markers and cards |

## Themes

| Theme | Mode | Background | Character |
|---|---|---|---|
| `midnight` | dark | `#0d1117` | Default, GitHub dark |
| `paper` | light | `#faf9f6` | Print friendly |
| `terminal` | dark | `#07100a` | CRT phosphor, monospace |
| `bloomberg` | dark | `#000000` | Amber finance terminal, monospace |
| `solarized` | dark | `#002b36` | Solarized dark accents |
| `neon` | dark | `#0a0a12` | High chroma synthwave |


## Tests

```bash
.venv/bin/python -m pytest -q -m 'not integration'   # offline, no ffmpeg
.venv/bin/python -m pytest -q                        # includes network and ffmpeg
```