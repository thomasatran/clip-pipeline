"""`clip run`: normalize, transcribe, rank, and render without duplicating a rerun."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from clip import db
from clip.deepgram import api_key_from_env, transcribe_file
from clip.domain import Rights, SourceVideo
from clip.errors import ClipError
from clip.media import extract_wav, normalize, remove_output, render_clip
from clip.rank import Pick, rank_candidates
from clip.transcript import Transcript, Word, parse_deepgram

Transcribe = Callable[[Path], dict]


@dataclass(frozen=True)
class RunRequest:
    path: Path
    rights: str
    db_path: Path
    out_dir: Path
    work_dir: Path
    transcript_path: Path | None = None
    top_n: int = 5
    transcribe: Transcribe | None = None


@dataclass(frozen=True)
class ClipOutput:
    id: str
    rank: int
    start_s: float
    end_s: float
    score: float
    reason: str
    file: Path
    state: str


@dataclass(frozen=True)
class RunResult:
    source: SourceVideo
    noop: bool
    clips: tuple[ClipOutput, ...]


def execute(request: RunRequest) -> RunResult:
    path = request.path.expanduser().resolve()
    if not path.is_file():
        raise ClipError(f"file not found: {path}")
    transcript_path = (
        request.transcript_path.expanduser().resolve() if request.transcript_path is not None else None
    )
    if transcript_path is not None and not transcript_path.is_file():
        raise ClipError(f"transcript not found: {transcript_path}")
    request = replace(request, transcript_path=transcript_path)
    rights = _rights(request.rights)
    if request.top_n < 1:
        raise ClipError("top_n must be at least 1")

    source_id = file_sha256(path)
    conn = db.connect(request.db_path)
    try:
        existing = db.get_source(conn, source_id)
        if existing is not None and existing["rights"] != rights:
            raise ClipError(
                f"source {source_id[:12]} is already ingested with rights {existing['rights']}"
            )
        if _is_complete(conn, source_id):
            if existing is not None and existing["original_path"] != str(path):
                conn.execute(
                    "UPDATE sources SET original_path = ? WHERE id = ?",
                    (str(path), source_id),
                )
                conn.commit()
            return _result(conn, source_id, noop=True)

        if not _can_transcribe(request, source_id):
            raise ClipError(
                "DEEPGRAM_API_KEY is not set. Pass --transcript to run from a fixture transcript."
            )

        now = _utcnow()
        work = (request.work_dir / source_id).resolve()
        work.mkdir(parents=True, exist_ok=True)
        normalized_path = work / "normalized.mp4"
        transcript_file = work / "transcript.json"
        try:
            info = normalize(path, normalized_path)
            source = SourceVideo(
                id=source_id,
                path=str(normalized_path),
                duration_s=info.duration_s,
                rights=rights,
            )
            db.upsert_source(
                conn,
                source_id=source.id,
                original_path=str(path),
                normalized_path=source.path,
                transcript_path=str(transcript_file) if transcript_file.is_file() else None,
                duration_ms=_millis(info.duration_s),
                rights=source.rights,
                now=now,
            )
            db.mark_run(conn, source.id, "running", now)
            conn.commit()

            payload = _load_transcript(request, transcript_file, normalized_path, info.has_audio)
            transcript = parse_deepgram(payload)
            db.upsert_source(
                conn,
                source_id=source.id,
                original_path=str(path),
                normalized_path=source.path,
                transcript_path=str(transcript_file),
                duration_ms=_millis(info.duration_s),
                rights=source.rights,
                now=now,
            )
            picks = rank_candidates(transcript, info.duration_s, request.top_n)
            out_dir = (request.out_dir / source.id).resolve()
            _apply_picks(
                conn,
                source_id=source.id,
                picks=picks,
                out_dir=out_dir,
                normalized=normalized_path,
                transcript=transcript,
                has_audio=info.has_audio,
                now=_utcnow(),
            )
            db.mark_run(conn, source.id, "rendered", _utcnow())
            conn.commit()
        except Exception as exc:
            conn.rollback()
            if db.get_source(conn, source_id) is not None:
                try:
                    db.mark_run(conn, source_id, "failed", _utcnow(), error=str(exc))
                    conn.commit()
                except Exception:
                    conn.rollback()
            raise
        return _result(conn, source_id, noop=False)
    finally:
        conn.close()


def _rights(value: str) -> Rights:
    if value == "own":
        return "own"
    if value == "licensed":
        return "licensed"
    raise ClipError("rights must be own or licensed")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_id_for(source_id: str, start_ms: int, end_ms: int) -> str:
    return hashlib.sha256(f"{source_id}:{start_ms}:{end_ms}".encode()).hexdigest()[:32]


def format_result(result: RunResult) -> str:
    status = "rendered, no-op" if result.noop else "rendered"
    lines = [
        f"source  {result.source.id}",
        f"status  {status}",
        f"clips   {len(result.clips)}",
    ]
    for clip in result.clips:
        lines.append(
            f"{clip.rank:02d}  {clip.start_s:.3f}-{clip.end_s:.3f}  "
            f"score={clip.score:.3f}  {clip.file}"
        )
        lines.append(f"    {clip.reason}")
    return "\n".join(lines)


def _apply_picks(
    conn,
    *,
    source_id: str,
    picks: list[Pick],
    out_dir: Path,
    normalized: Path,
    transcript: Transcript,
    has_audio: bool,
    now: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {row["id"]: row for row in db.list_clips(conn, source_id)}
    pick_ids = {
        clip_id_for(source_id, _millis(pick.span.start_s), _millis(pick.span.end_s))
        for pick in picks
    }
    for clip_id, row in list(existing.items()):
        if row["state"] in ("candidate", "rendered") and clip_id not in pick_ids:
            remove_output(row["file_path"])
            db.delete_clip(conn, clip_id)
            del existing[clip_id]

    for rank, pick in enumerate(picks, start=1):
        start_ms = _millis(pick.span.start_s)
        end_ms = _millis(pick.span.end_s)
        if end_ms <= start_ms or end_ms - start_ms > 60000:
            raise ClipError(f"clip span {start_ms}-{end_ms} is outside 0 to 60 seconds")
        clip_id = clip_id_for(source_id, start_ms, end_ms)
        row = existing.get(clip_id)
        if row is not None and row["state"] in ("approved", "rejected"):
            continue
        dest = out_dir / f"{rank:02d}_{start_ms}-{end_ms}.mp4"
        if row is not None and row["file_path"] and row["file_path"] != str(dest):
            remove_output(row["file_path"])
        words = _words_for_span(transcript, pick)
        render_clip(normalized, pick.span, words, dest, has_audio=has_audio)
        if row is None:
            db.insert_clip(
                conn,
                clip_id=clip_id,
                source_id=source_id,
                start_ms=start_ms,
                end_ms=end_ms,
                score=pick.score,
                reason=pick.reason,
                state="rendered",
                file_path=str(dest),
                rank=rank,
                now=now,
            )
        else:
            db.update_rendered_clip(
                conn,
                clip_id=clip_id,
                score=pick.score,
                reason=pick.reason,
                file_path=str(dest),
                rank=rank,
                now=now,
            )


def _words_for_span(transcript: Transcript, pick: Pick) -> tuple[Word, ...]:
    selected: list[Word] = []
    for sentence in transcript.sentences:
        if sentence.end_s > pick.span.start_s and sentence.start_s < pick.span.end_s:
            selected.extend(sentence.words)
    if selected:
        return tuple(selected)
    return transcript.words


def _load_transcript(
    request: RunRequest, transcript_file: Path, normalized: Path, has_audio: bool
) -> dict:
    if transcript_file.is_file():
        payload = json.loads(transcript_file.read_text(encoding="utf-8"))
        parse_deepgram(payload)
        return payload
    if request.transcript_path is not None:
        payload = json.loads(request.transcript_path.read_text(encoding="utf-8"))
        parse_deepgram(payload)
        _write_json(transcript_file, payload)
        return payload
    if not has_audio:
        raise ClipError(f"{normalized} has no audio stream to transcribe")
    wav_path = transcript_file.with_name("audio.wav")
    if not wav_path.is_file():
        extract_wav(normalized, wav_path)
    transcribe = request.transcribe or (lambda media: transcribe_file(media))
    payload = transcribe(wav_path)
    parse_deepgram(payload)
    _write_json(transcript_file, payload)
    return payload


def _can_transcribe(request: RunRequest, source_id: str) -> bool:
    if request.transcript_path is not None or request.transcribe is not None:
        return True
    stored = (request.work_dir / source_id / "transcript.json").resolve()
    if stored.is_file():
        return True
    return api_key_from_env() is not None


def _is_complete(conn, source_id: str) -> bool:
    run = db.get_run(conn, source_id)
    if run is None or run["status"] != "rendered":
        return False
    for row in db.list_clips(conn, source_id):
        if row["state"] == "candidate" or not row["file_path"]:
            return False
        file = Path(row["file_path"])
        if not file.is_file() or file.stat().st_size <= 0:
            return False
    return True


def _result(conn, source_id: str, *, noop: bool) -> RunResult:
    row = db.get_source(conn, source_id)
    if row is None:
        raise ClipError(f"source disappeared: {source_id}")
    source = SourceVideo(
        id=row["id"],
        path=row["normalized_path"],
        duration_s=row["duration_ms"] / 1000,
        rights=row["rights"],
    )
    clips: list[ClipOutput] = []
    for clip in db.list_clips(conn, source_id):
        clips.append(
            ClipOutput(
                id=clip["id"],
                rank=clip["rank"],
                start_s=clip["start_ms"] / 1000,
                end_s=clip["end_ms"] / 1000,
                score=clip["score"],
                reason=clip["reason"],
                file=Path(clip["file_path"]) if clip["file_path"] else Path(),
                state=clip["state"],
            )
        )
    return RunResult(source=source, noop=noop, clips=tuple(clips))


def _write_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload), encoding="utf-8")
    partial.replace(path)


def _millis(seconds: float) -> int:
    return int(round(seconds * 1000))


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
