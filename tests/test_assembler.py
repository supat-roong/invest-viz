"""Tests for the ffmpeg assembler.

Anything that actually shells out to ffmpeg carries the `integration`
marker; the rest run offline by monkeypatching subprocess.run.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from investment_video import assembler


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = '', stderr: str = ''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --- check_ffmpeg -----------------------------------------------------------

def test_check_ffmpeg_passes_on_this_machine():
    assembler.check_ffmpeg()


def test_check_ffmpeg_names_missing_binaries(monkeypatch):
    monkeypatch.setenv('PATH', '')
    monkeypatch.setattr(assembler.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        assembler.check_ffmpeg()
    message = str(excinfo.value)
    assert 'ffmpeg' in message
    assert 'ffprobe' in message


def test_check_ffmpeg_names_only_the_absent_binary(monkeypatch):
    monkeypatch.setattr(
        assembler.shutil, 'which',
        lambda name: '/usr/bin/ffmpeg' if name == 'ffmpeg' else None,
    )
    with pytest.raises(RuntimeError) as excinfo:
        assembler.check_ffmpeg()
    message = str(excinfo.value)
    assert 'ffprobe' in message
    assert 'ffmpeg,' not in message


# --- argv and error handling ------------------------------------------------

def test_frames_to_video_builds_expected_argv(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        captured['kwargs'] = kwargs
        return _FakeCompleted(0)

    monkeypatch.setattr(assembler.subprocess, 'run', fake_run)

    frames_dir = tmp_path / 'frames'
    frames_dir.mkdir()
    out = tmp_path / 'out.mp4'

    result = assembler.frames_to_video(frames_dir, 30, out)

    assert result == out
    assert captured['kwargs']['capture_output'] is True
    assert captured['cmd'] == [
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', str(frames_dir / 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-an',
        str(out),
    ]


def test_frames_to_video_raises_with_stderr_tail(monkeypatch, tmp_path):
    tail = 'Error: no such file or directory'
    noise = 'x' * 5000

    monkeypatch.setattr(
        assembler.subprocess, 'run',
        lambda cmd, **kw: _FakeCompleted(1, stderr=noise + tail),
    )

    with pytest.raises(RuntimeError) as excinfo:
        assembler.frames_to_video(tmp_path, 30, tmp_path / 'out.mp4')

    message = str(excinfo.value)
    assert 'frames_to_video' in message
    assert tail in message
    assert len(message) < 2500


def test_concat_videos_builds_expected_argv_and_cleans_up(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        # The concat list must still exist while ffmpeg would be reading it.
        captured['filelist'] = Path(cmd[cmd.index('-i') + 1])
        captured['filelist_body'] = captured['filelist'].read_text()
        return _FakeCompleted(0)

    monkeypatch.setattr(assembler.subprocess, 'run', fake_run)

    a = tmp_path / 'a.mp4'
    b = tmp_path / 'b.mp4'
    a.touch()
    b.touch()
    out = tmp_path / 'joined.mp4'

    assert assembler.concat_videos([a, b], out) == out

    cmd = captured['cmd']
    assert cmd[:2] == ['ffmpeg', '-y']
    assert cmd[2:6] == ['-f', 'concat', '-safe', '0']
    assert '-c' in cmd and cmd[cmd.index('-c') + 1] == 'copy'
    assert '-an' in cmd
    assert cmd[-1] == str(out)
    assert str(a.absolute()) in captured['filelist_body']
    assert str(b.absolute()) in captured['filelist_body']
    assert not captured['filelist'].exists()


def test_concat_videos_removes_filelist_on_failure(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen['filelist'] = Path(cmd[cmd.index('-i') + 1])
        return _FakeCompleted(1, stderr='boom')

    monkeypatch.setattr(assembler.subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError, match='concat_videos'):
        assembler.concat_videos([tmp_path / 'a.mp4'], tmp_path / 'out.mp4')

    assert not seen['filelist'].exists()


# --- real ffmpeg ------------------------------------------------------------

@pytest.mark.integration
def test_frames_to_video_produces_a_silent_mp4(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    assembler.check_ffmpeg()

    frames_dir = tmp_path / 'frames'
    frames_dir.mkdir()
    colors = [
        (200, 30, 30), (30, 200, 30), (30, 30, 200),
        (200, 200, 30), (200, 30, 200), (30, 200, 200),
    ]
    for i, color in enumerate(colors, start=1):
        Image.new('RGB', (64, 64), color).save(frames_dir / f'frame_{i:05d}.png')

    out = tmp_path / 'tiny.mp4'
    assembler.frames_to_video(frames_dir, 6, out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _audio_stream_count(out) == 0


@pytest.mark.integration
def test_concat_videos_joins_two_clips(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    assembler.check_ffmpeg()

    clips = []
    for clip_index, color in enumerate([(10, 120, 200), (200, 120, 10)]):
        frames_dir = tmp_path / f'frames{clip_index}'
        frames_dir.mkdir()
        for i in range(1, 5):
            Image.new('RGB', (64, 64), color).save(
                frames_dir / f'frame_{i:05d}.png'
            )
        clip = tmp_path / f'clip{clip_index}.mp4'
        assembler.frames_to_video(frames_dir, 4, clip)
        clips.append(clip)

    out = tmp_path / 'joined.mp4'
    assembler.concat_videos(clips, out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _audio_stream_count(out) == 0


def _audio_stream_count(path: Path) -> int:
    probe = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'json',
            str(path),
        ],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    streams = json.loads(probe.stdout).get('streams', [])
    return sum(1 for s in streams if s.get('codec_type') == 'audio')
