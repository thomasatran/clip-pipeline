import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from clip import db
from clip.domain import (
    Candidate,
    PostSpec,
    RemoteState,
    Rendered,
    Span,
    approve,
    publications_for,
    reconcile,
    with_handle,
)
from clip.errors import ClipError

NOW = "2026-11-10T00:00:00+00:00"
LATER = "2026-11-10T00:05:00+00:00"


def test_schema_version_and_phase1_columns(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "clip.sqlite")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {
        "sources",
        "runs",
        "clips",
        "clip_platforms",
        "publications",
        "accounts",
    } <= tables
    columns = {row[1] for row in conn.execute("PRAGMA table_info(publications)")}
    assert {
        "handle",
        "remote_id",
        "remote_url",
        "state",
        "route",
        "retryable",
        "failure_reason",
    } <= columns

    again = db.connect(tmp_path / "clip.sqlite")
    assert again.execute("PRAGMA user_version").fetchone()[0] == 1
    again.close()
    conn.close()


def test_checks_reject_bad_rights_spans_and_states(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "clip.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO sources (
              id, original_path, normalized_path, duration_ms, rights, created_at
            ) VALUES ('src', 'a', 'b', 1, 'pirated', ?)
            """,
            (NOW,),
        )
    conn.rollback()

    db.upsert_source(
        conn,
        source_id="src",
        original_path="a",
        normalized_path="b",
        transcript_path=None,
        duration_ms=12000,
        rights="own",
        now=NOW,
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_clip(
            conn,
            clip_id="too-long",
            source_id="src",
            start_ms=0,
            end_ms=60001,
            score=1,
            reason="too long",
            state="candidate",
            file_path=None,
            rank=1,
            now=NOW,
        )
    conn.rollback()

    db.insert_clip(
        conn,
        clip_id="clip1",
        source_id="src",
        start_ms=0,
        end_ms=10000,
        score=1,
        reason="ok",
        state="rendered",
        file_path="/tmp/clip.mp4",
        rank=1,
        now=NOW,
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO publications (
              id, clip_id, platform, title, caption, hashtags, state, route, created_at, updated_at
            ) VALUES ('pub', 'clip1', 'youtube', 't', 'c', '[]', 'queued', NULL, ?, ?)
            """,
            (NOW, NOW),
        )
    conn.rollback()
    conn.close()


def test_publication_handle_is_saved_before_a_live_state_and_rerun_is_a_no_op(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "clip.sqlite")
    db.upsert_source(
        conn,
        source_id="src",
        original_path="in.mp4",
        normalized_path="norm.mp4",
        transcript_path=None,
        duration_ms=12000,
        rights="licensed",
        now=NOW,
    )
    db.insert_clip(
        conn,
        clip_id="clip1",
        source_id="src",
        start_ms=0,
        end_ms=10000,
        score=0.8,
        reason="duration 10.0s",
        state="rendered",
        file_path="/tmp/clip.mp4",
        rank=1,
        now=NOW,
    )
    db.approve_clip(conn, "clip1", frozenset({"youtube", "instagram"}), NOW)

    rendered = Rendered(
        Candidate("clip1", "src", Span(0, 10), 0.8, "duration 10.0s"),
        "/tmp/clip.mp4",
    )
    approved = approve(rendered, frozenset({"youtube", "instagram"}))
    queued_youtube, queued_instagram = publications_for(
        approved,
        [
            PostSpec("youtube", "Title", "Caption", ("#shorts",)),
            PostSpec("instagram", "Title", "Caption", ()),
        ],
    )
    stored = db.ensure_queued(conn, queued_youtube, NOW)
    again = db.ensure_queued(conn, queued_youtube, LATER)
    assert stored == again
    assert conn.execute("SELECT count(*) FROM publications").fetchone()[0] == 1
    conn.commit()

    uploading = with_handle(queued_youtube, "https://example.test/session")
    with pytest.raises(sqlite3.IntegrityError):
        db.save_publication(conn, replace(uploading, handle=None), LATER)
    conn.rollback()
    db.save_publication(conn, uploading, LATER)
    assert db.get_publication(conn, uploading.id) == uploading

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE publications SET state = 'container', handle = 'nope' WHERE id = ?",
            (uploading.id,),
        )
    conn.rollback()

    published = reconcile(
        uploading,
        RemoteState(processed=True, remote_id="vid123", remote_url="https://youtu.be/vid123"),
    )
    db.save_publication(conn, published, LATER)
    updated = conn.execute(
        "SELECT updated_at FROM publications WHERE id = ?",
        (published.id,),
    ).fetchone()[0]
    db.save_publication(conn, published, "2026-11-10T00:09:00+00:00")
    assert (
        conn.execute(
            "SELECT updated_at FROM publications WHERE id = ?",
            (published.id,),
        ).fetchone()[0]
        == updated
    )
    assert db.ensure_queued(conn, queued_youtube, "2026-11-10T00:09:00+00:00").state == "published"
    assert conn.execute("SELECT count(*) FROM publications").fetchone()[0] == 1

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE publications SET state = 'queued' WHERE id = ?",
            (published.id,),
        )
    conn.rollback()

    container = with_handle(queued_instagram, "container-1")
    db.ensure_queued(conn, queued_instagram, NOW)
    db.save_publication(conn, container, LATER)
    expired = reconcile(container, RemoteState(expired=True))
    db.save_publication(conn, expired, LATER)
    stored_expired = db.get_publication(conn, expired.id)
    assert stored_expired is not None
    assert stored_expired.state == "queued"
    assert stored_expired.handle is None

    db.set_account_pause(conn, "youtube", "spam")
    account = conn.execute("SELECT * FROM accounts WHERE platform = 'youtube'").fetchone()
    assert account["pause_reason"] == "spam"
    assert account["paused_until"] is None
    db.clear_account_pause(conn, "youtube")
    assert conn.execute("SELECT count(*) FROM accounts").fetchone()[0] == 0
    conn.close()


def test_unknown_schema_version_fails(tmp_path: Path) -> None:
    path = tmp_path / "clip.sqlite"
    conn = db.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(ClipError, match="unsupported schema version"):
        db.connect(path)
