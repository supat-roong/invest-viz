"""ffmpeg wrappers: frames to a silent MP4, and MP4 concatenation.

Follows the sibling minesweeper-video-generator assembler: every ffmpeg
invocation goes through `_run`, which captures output and raises a
RuntimeError carrying the tail of stderr. No audio stream is ever added.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

#: Frame filename pattern. Five digits — a 45s render at 60fps overruns four.
FRAME_PATTERN = 'frame_%05d.png'

REQUIRED_BINARIES = ('ffmpeg', 'ffprobe')


def check_ffmpeg() -> None:
    """Verify ffmpeg and ffprobe are on PATH, so we fail before rendering.

    Raises RuntimeError naming every binary that is missing.
    """
    missing = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            'ffmpeg tooling not found on PATH: ' + ', '.join(missing)
            + '. Install ffmpeg (e.g. `brew install ffmpeg`) and retry.'
        )


def frames_to_video(frames_dir: Path, fps: int, output_path: Path) -> Path:
    """Encode a directory of frame_%05d.png files into a silent MP4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(Path(frames_dir) / FRAME_PATTERN),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-an',
        str(output_path),
    ]
    _run(cmd, 'frames_to_video')
    return output_path


def concat_videos(video_paths: list, output_path: Path) -> Path:
    """Concatenate MP4 files using the ffmpeg concat demuxer (stream copy)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for p in video_paths:
            f.write(f"file '{Path(p).absolute()}'\n")
        filelist = Path(f.name)
    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', str(filelist),
            '-c', 'copy',
            '-an',
            str(output_path),
        ]
        _run(cmd, 'concat_videos')
    finally:
        filelist.unlink(missing_ok=True)
    return output_path


def _run(cmd: list, label: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr or ''
        raise RuntimeError(f'ffmpeg [{label}] failed:\n{stderr[-2000:]}')
