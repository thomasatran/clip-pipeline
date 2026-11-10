import hashlib
import sqlite3
from pathlib import Path

import pytest

from clip import db
from clip.cli import run_cli
from clip.domain import Span
from clip.errors import ClipError
from clip.media import probe
from clip.pipeline import RunRequest, _apply_picks, execute, file_sha256
from clip.rank import Pick
from clip.transcript import Sentence, Transcript, Word
from tests.conftest import make_test_video


def test_missing_transcript_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"not a video")
    code = run_cli(
        [
            "run",
            str(video),
            "--rights",
            "own",
            "--transcript",
            str(tmp_path / "missing.json"),
            "--db",
            str(tmp_path / "clip.sqlite"),
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ]
    )
    assert code == 2
    assert not (tmp_path / "clip.sqlite").exists()


def test_file_hash_identifies_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"same-bytes")
    second.write_bytes(b"same-bytes")
    assert file_sha256(first) == file_sha256(second)
    second.write_bytes(b"other-bytes")
    assert file_sha256(first) != file_sha256(second)


def test_missing_deepgram_key_stops_before_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"not a video")
    code = run_cli(
        [
            "run",
            str(video),
            "--rights",
            "own",
            "--db",
            str(tmp_path / "clip.sqlite"),
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ]
    )
    assert code == 2
    conn = sqlite3.connect(tmp_path / "clip.sqlite")
    try:
        assert conn.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
    finally:
        conn.close()
    assert list((tmp_path / "work").rglob("*.mp4")) == []
    assert list((tmp_path / "out").rglob("*.mp4")) == []


def test_reapplying_picks_replaces_unreviewed_clips_and_keeps_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_render(normalized: Path, span: Span, words, mp4_path: Path, *, has_audio: bool) -> Path:
        del normalized, span, words, has_audio
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        if not mp4_path.exists():
            mp4_path.write_bytes(b"clip")
        mp4_path.with_suffix(".ass").write_text("captions", encoding="utf-8")
        return mp4_path

    monkeypatch.setattr("clip.pipeline.render_clip", fake_render)
    conn = db.connect(tmp_path / "clip.sqlite")
    now = "2026-11-10T00:00:00+00:00"
    db.upsert_source(
        conn,
        source_id="source",
        original_path="in.mp4",
        normalized_path="norm.mp4",
        transcript_path=None,
        duration_ms=40000,
        rights="own",
        now=now,
    )
    words = (
        Word("Alpha sentence here.", 0, 10, 0.9),
        Word("Beta sentence here.", 12, 22, 0.9),
        Word("Gamma sentence here.", 24, 34, 0.9),
    )
    sentences = tuple(Sentence(word.text, word.start_s, word.end_s, (word,)) for word in words)
    transcript = Transcript(words, sentences, 40)
    pick_a = Pick(Span(0, 10), 0.4, "first")
    pick_b = Pick(Span(12, 22), 0.9, "second")
    pick_c = Pick(Span(24, 34), 0.5, "third")
    out = tmp_path / "out"

    def apply(picks: list[Pick], stamp: str) -> None:
        _apply_picks(
            conn,
            source_id="source",
            picks=picks,
            out_dir=out,
            normalized=Path("norm.mp4"),
            transcript=transcript,
            has_audio=False,
            now=stamp,
        )

    apply([pick_b, pick_a], now)
    ids = _clip_ids(conn)
    assert len(ids) == 2
    apply([pick_b, pick_a], "2026-11-10T00:01:00+00:00")
    assert _clip_ids(conn) == ids

    approved_id = conn.execute("SELECT id FROM clips WHERE start_ms = 12000").fetchone()[0]
    db.approve_clip(conn, approved_id, frozenset({"youtube"}), now)
    removed = out / "02_0-10000.mp4"
    assert removed.is_file()
    apply([pick_c], "2026-11-10T00:02:00+00:00")
    rows = [tuple(row) for row in conn.execute("SELECT start_ms, state FROM clips ORDER BY start_ms")]
    assert rows == [(12000, "approved"), (24000, "rendered")]
    assert not removed.exists()
    assert (out / "01_12000-22000.mp4").is_file()
    assert (out / "01_24000-34000.mp4").is_file()
    conn.close()


def test_run_renders_9_16_captions_and_rerun_is_a_no_op(
    tmp_path: Path, transcript_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    calls: list[Path] = []

    def spy(path: Path) -> dict:
        calls.append(path)
        raise AssertionError("Deepgram should not be called when a transcript is supplied")

    video = make_test_video(tmp_path / "sample.mp4")
    db_path = tmp_path / "data" / "clip.sqlite"
    out = tmp_path / "out"
    work = tmp_path / "work"
    request = RunRequest(
        path=video,
        rights="own",
        db_path=db_path,
        out_dir=out,
        work_dir=work,
        transcript_path=transcript_path,
        transcribe=spy,
    )
    result = execute(request)
    assert calls == []
    assert result.noop is False
    assert len(result.clips) == 1
    clip = result.clips[0]
    assert "question hook" in clip.reason
    info = probe(clip.file)
    assert (info.width, info.height) == (1080, 1920)
    assert info.pix_fmt == "yuv420p"
    assert info.has_audio
    assert info.fps is not None and abs(info.fps - 30) < 0.2
    assert 10.2 <= info.duration_s <= 11.4
    assert "Why does this matter?" in clip.file.with_suffix(".ass").read_text(encoding="utf-8")

    normalized = work / result.source.id / "normalized.mp4"
    norm = probe(normalized)
    assert (norm.width, norm.height) == (1920, 1080)
    assert norm.pix_fmt == "yuv420p"

    before = _dump(db_path)
    hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (clip.file, clip.file.with_suffix(".ass"), normalized)
    }
    assert before["sources"] and len(before["sources"]) == 1
    assert len(before["runs"]) == 1
    assert len(before["clips"]) == 1

    code = run_cli(
        [
            "run",
            str(video),
            "--rights",
            "own",
            "--transcript",
            str(transcript_path),
            "--db",
            str(db_path),
            "--out",
            str(out),
            "--work",
            str(work),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "no-op" in printed
    assert _dump(db_path) == before
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert calls == []
    assert list(out.rglob("*.mp4")) == [clip.file]

    with pytest.raises(ClipError, match="already ingested"):
        execute(
            RunRequest(
                path=video,
                rights="licensed",
                db_path=db_path,
                out_dir=out,
                work_dir=work,
                transcript_path=transcript_path,
            )
        )
    assert _dump(db_path) == before


def _clip_ids(conn) -> list[str]:
    return [row[0] for row in conn.execute("SELECT id FROM clips ORDER BY start_ms")]


def _dump(db_path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(db_path)
    try:
        dumped: dict[str, list[tuple]] = {}
        for name in ("sources", "runs", "clips", "clip_platforms", "publications", "accounts"):
            rows = conn.execute(f"SELECT * FROM {name} ORDER BY 1").fetchall()
            dumped[name] = [tuple(row) for row in rows]
        return dumped
    finally:
        conn.close()
