"""ffmpeg normalize, center-crop, and burned-in ASS captions."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clip.domain import Span
from clip.errors import ClipError
from clip.transcript import Word

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
FRAME_RATE = 30


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    width: int
    height: int
    pix_fmt: str | None
    fps: float | None
    has_audio: bool


def probe(path: Path) -> MediaInfo:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        "ffprobe",
    )
    try:
        payload = json.loads(proc.stdout)
        streams = payload["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        duration = float(payload["format"]["duration"])
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ClipError(f"ffprobe could not read a video stream from {path}") from exc
    return MediaInfo(
        duration_s=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        pix_fmt=video.get("pix_fmt"),
        fps=_parse_rate(video.get("avg_frame_rate")),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def normalize(src: Path, dest: Path) -> MediaInfo:
    """Write a 1920x1080 constant-frame-rate yuv420p copy with timestamps from 0."""

    if dest.is_file() and dest.stat().st_size > 0:
        try:
            info = probe(dest)
        except ClipError:
            info = None
        if (
            info is not None
            and info.width == 1920
            and info.height == 1080
            and info.pix_fmt == "yuv420p"
        ):
            return info

    source_info = probe(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial.mp4")
    video_filter = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1,fps=30,setpts=PTS-STARTPTS"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        video_filter,
        "-fps_mode",
        "cfr",
        "-r",
        str(FRAME_RATE),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
    ]
    if source_info.has_audio:
        command += ["-af", "asetpts=PTS-STARTPTS", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    else:
        command += ["-an"]
    command.append(str(partial))
    _run(command, "ffmpeg")
    partial.replace(dest)
    return probe(dest)


def extract_wav(video: Path, wav_path: Path) -> None:
    info = probe(video)
    if not info.has_audio:
        raise ClipError(f"{video} has no audio stream to transcribe")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    partial = wav_path.with_suffix(".partial.wav")
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(partial),
        ],
        "ffmpeg",
    )
    partial.replace(wav_path)


def render_clip(
    normalized: Path,
    span: Span,
    words: tuple[Word, ...] | list[Word],
    mp4_path: Path,
    *,
    has_audio: bool,
) -> Path:
    """Cut, center-crop to 9:16, scale to 1080x1920, and burn captions."""

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path = mp4_path.with_suffix(".ass")
    cues = caption_cues(words, span.start_s, span.end_s)
    write_ass(ass_path, cues)
    if mp4_path.is_file() and mp4_path.stat().st_size > 0:
        return mp4_path

    info = probe(normalized)
    crop_w, crop_h, crop_x, crop_y = center_crop_9_16(info.width, info.height)
    video_filter = (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1,fps={FRAME_RATE},"
        f"ass={escape_filter_path(ass_path)}"
    )
    partial = mp4_path.with_suffix(".partial.mp4")
    duration = span.end_s - span.start_s
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{span.start_s:.3f}",
        "-i",
        str(normalized),
        "-t",
        f"{duration:.3f}",
        "-vf",
        video_filter,
        "-fps_mode",
        "cfr",
        "-r",
        str(FRAME_RATE),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        command += ["-c:a", "aac", "-b:a", "128k"]
    else:
        command += ["-an"]
    command.append(str(partial))
    _run(command, "ffmpeg")
    partial.replace(mp4_path)
    return mp4_path


def center_crop_9_16(width: int, height: int) -> tuple[int, int, int, int]:
    """Return crop width, height, x, y for a centered 9:16 window."""

    if width < 2 or height < 2:
        raise ClipError(f"video is too small to crop ({width}x{height})")
    if width * 16 > height * 9:
        crop_h = height - (height % 2)
        crop_w = _even(int(crop_h * 9 / 16))
        crop_x = max(0, (width - crop_w) // 2)
        crop_y = max(0, (height - crop_h) // 2)
    else:
        crop_w = width - (width % 2)
        crop_h = _even(int(crop_w * 16 / 9))
        if crop_h > height:
            crop_h = height - (height % 2)
        crop_x = max(0, (width - crop_w) // 2)
        crop_y = max(0, (height - crop_h) // 2)
    if crop_w < 2 or crop_h < 2:
        raise ClipError(f"crop collapsed for {width}x{height}")
    return crop_w, crop_h, crop_x, crop_y


def caption_cues(
    words: tuple[Word, ...] | list[Word], clip_start: float, clip_end: float
) -> list[tuple[float, float, str]]:
    selected: list[Word] = []
    for word in words:
        midpoint = (word.start_s + word.end_s) / 2
        if clip_start <= midpoint < clip_end:
            selected.append(word)
    cues: list[tuple[float, float, str]] = []
    bucket: list[Word] = []

    def flush() -> None:
        if not bucket:
            return
        start = max(0.0, bucket[0].start_s - clip_start)
        end = max(start + 0.05, bucket[-1].end_s - clip_start)
        text = " ".join(word.text for word in bucket)
        cues.append((start, end, text))
        bucket.clear()

    for word in selected:
        bucket.append(word)
        if len(bucket) >= 5 or word.text.endswith((".", "?", "!")):
            flush()
    flush()
    return cues


def write_ass(path: Path, cues: list[tuple[float, float, str]]) -> None:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {OUTPUT_WIDTH}",
        f"PlayResY: {OUTPUT_HEIGHT}",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,DejaVu Sans,68,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,4,0,2,80,80,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in cues:
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{_escape_ass(text)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_filter_path(path: Path) -> str:
    text = path.resolve().as_posix()
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def remove_output(path: str | None) -> None:
    if not path:
        return
    file = Path(path)
    file.unlink(missing_ok=True)
    file.with_suffix(".ass").unlink(missing_ok=True)
    file.with_suffix(".partial.mp4").unlink(missing_ok=True)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    seconds_part, centiseconds = divmod(centiseconds, 100)
    minutes, seconds_part = divmod(seconds_part, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds_part:02d}.{centiseconds:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, _, denominator = value.partition("/")
    try:
        if denominator:
            den = float(denominator)
            if den == 0:
                return None
            return float(numerator) / den
        return float(numerator)
    except ValueError:
        return None


def _run(command: list[str], program: str) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ClipError(f"{program} is not installed") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ClipError(f"{program} failed: {detail[-2000:]}")
    return proc
