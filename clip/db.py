"""SQLite storage for sources, runs, clips, and publication rows.

Schema version 1 is the phase 0 layout. Publication columns already hold the
per-platform states, the handle saved before a side effect, and the TikTok
route, so phase 1 can publish without rewriting tables.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from clip.domain import (
    InstagramPub,
    Publication,
    TikTokPub,
    YouTubePub,
)
from clip.errors import ClipError

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE sources (
  id TEXT PRIMARY KEY,
  original_path TEXT NOT NULL,
  normalized_path TEXT NOT NULL,
  transcript_path TEXT,
  duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
  rights TEXT NOT NULL CHECK (rights IN ('own', 'licensed')),
  created_at TEXT NOT NULL
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL UNIQUE REFERENCES sources(id),
  status TEXT NOT NULL CHECK (status IN ('running', 'rendered', 'failed')),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  error TEXT
);

CREATE TABLE clips (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(id),
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  score REAL NOT NULL,
  reason TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('candidate', 'rendered', 'approved', 'rejected')),
  file_path TEXT,
  rank INTEGER NOT NULL CHECK (rank >= 1),
  rejection_note TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK (start_ms >= 0),
  CHECK (end_ms > start_ms),
  CHECK (end_ms - start_ms <= 60000),
  CHECK (state != 'rejected' OR (rejection_note IS NOT NULL AND length(trim(rejection_note)) > 0)),
  CHECK (state = 'candidate' OR file_path IS NOT NULL),
  UNIQUE (source_id, start_ms, end_ms)
);

CREATE TABLE clip_platforms (
  clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  platform TEXT NOT NULL CHECK (platform IN ('youtube', 'instagram', 'tiktok')),
  PRIMARY KEY (clip_id, platform)
);

CREATE TABLE publications (
  id TEXT PRIMARY KEY,
  clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  platform TEXT NOT NULL CHECK (platform IN ('youtube', 'instagram', 'tiktok')),
  title TEXT NOT NULL,
  caption TEXT NOT NULL,
  hashtags TEXT NOT NULL,
  state TEXT NOT NULL,
  handle TEXT,
  remote_id TEXT,
  remote_url TEXT,
  failure_reason TEXT,
  retryable INTEGER CHECK (retryable IS NULL OR retryable IN (0, 1)),
  route TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (clip_id, platform),
  CHECK (
    (platform = 'youtube' AND state IN ('queued', 'uploading', 'verifying', 'published', 'failed') AND route IS NULL)
    OR (platform = 'instagram' AND state IN ('queued', 'container', 'ready', 'verifying', 'published', 'failed') AND route IS NULL)
    OR (platform = 'tiktok' AND state IN ('queued', 'submitted', 'verifying', 'published', 'failed') AND route IN ('upload_post', 'manual'))
  )
);

CREATE TABLE accounts (
  platform TEXT PRIMARY KEY CHECK (platform IN ('youtube', 'instagram', 'tiktok')),
  paused_until TEXT,
  pause_reason TEXT
);

CREATE INDEX idx_clips_source ON clips(source_id);
CREATE INDEX idx_publications_state ON publications(platform, state);

CREATE TRIGGER publications_require_approved
BEFORE INSERT ON publications
FOR EACH ROW
WHEN (SELECT state FROM clips WHERE id = NEW.clip_id) IS NOT 'approved'
BEGIN
  SELECT RAISE(ABORT, 'publication requires an approved clip');
END;

CREATE TRIGGER publications_require_platform
BEFORE INSERT ON publications
FOR EACH ROW
WHEN NOT EXISTS (
  SELECT 1 FROM clip_platforms
  WHERE clip_id = NEW.clip_id AND platform = NEW.platform
)
BEGIN
  SELECT RAISE(ABORT, 'platform was not approved');
END;

CREATE TRIGGER publications_handle_before_side_effect_insert
BEFORE INSERT ON publications
FOR EACH ROW
WHEN NEW.state NOT IN ('queued', 'failed')
 AND (NEW.handle IS NULL OR length(trim(NEW.handle)) = 0)
BEGIN
  SELECT RAISE(ABORT, 'save the handle before leaving queued');
END;

CREATE TRIGGER publications_handle_before_side_effect_update
BEFORE UPDATE ON publications
FOR EACH ROW
WHEN NEW.state NOT IN ('queued', 'failed')
 AND (NEW.handle IS NULL OR length(trim(NEW.handle)) = 0)
BEGIN
  SELECT RAISE(ABORT, 'save the handle before leaving queued');
END;

CREATE TRIGGER publications_published_stays
BEFORE UPDATE ON publications
FOR EACH ROW
WHEN OLD.state = 'published' AND NEW.state != 'published'
BEGIN
  SELECT RAISE(ABORT, 'published row cannot change state');
END;
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA_SQL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    elif version != SCHEMA_VERSION:
        conn.close()
        raise ClipError(f"unsupported schema version {version}")
    return conn


def get_source(conn: sqlite3.Connection, source_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()


def upsert_source(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    original_path: str,
    normalized_path: str,
    transcript_path: str | None,
    duration_ms: int,
    rights: str,
    now: str,
) -> None:
    if rights not in ("own", "licensed"):
        raise ClipError("rights must be own or licensed")
    existing = get_source(conn, source_id)
    if existing is None:
        conn.execute(
            """
            INSERT INTO sources (
              id, original_path, normalized_path, transcript_path, duration_ms, rights, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                original_path,
                normalized_path,
                transcript_path,
                duration_ms,
                rights,
                now,
            ),
        )
        return
    if existing["rights"] != rights:
        raise ClipError(
            f"source {source_id[:12]} is already ingested with rights {existing['rights']}"
        )
    conn.execute(
        """
        UPDATE sources
        SET original_path = ?, normalized_path = ?, transcript_path = ?, duration_ms = ?
        WHERE id = ?
        """,
        (original_path, normalized_path, transcript_path, duration_ms, source_id),
    )


def get_run(conn: sqlite3.Connection, source_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE source_id = ?", (source_id,)).fetchone()


def mark_run(
    conn: sqlite3.Connection,
    source_id: str,
    status: str,
    now: str,
    error: str | None = None,
) -> None:
    if status not in ("running", "rendered", "failed"):
        raise ClipError(f"unknown run status: {status}")
    existing = get_run(conn, source_id)
    finished_at = None if status == "running" else now
    if existing is None:
        conn.execute(
            """
            INSERT INTO runs (id, source_id, status, started_at, finished_at, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_id, source_id, status, now, finished_at, error),
        )
        return
    conn.execute(
        """
        UPDATE runs
        SET status = ?, finished_at = ?, error = ?
        WHERE source_id = ?
        """,
        (status, finished_at, error, source_id),
    )


def list_clips(conn: sqlite3.Connection, source_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM clips WHERE source_id = ? ORDER BY rank, start_ms",
            (source_id,),
        )
    )


def get_clip(conn: sqlite3.Connection, clip_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()


def insert_clip(
    conn: sqlite3.Connection,
    *,
    clip_id: str,
    source_id: str,
    start_ms: int,
    end_ms: int,
    score: float,
    reason: str,
    state: str,
    file_path: str | None,
    rank: int,
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO clips (
          id, source_id, start_ms, end_ms, score, reason, state, file_path, rank,
          rejection_note, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (clip_id, source_id, start_ms, end_ms, score, reason, state, file_path, rank, now, now),
    )


def update_rendered_clip(
    conn: sqlite3.Connection,
    *,
    clip_id: str,
    score: float,
    reason: str,
    file_path: str,
    rank: int,
    now: str,
) -> None:
    cursor = conn.execute(
        """
        UPDATE clips
        SET score = ?, reason = ?, state = 'rendered', file_path = ?, rank = ?, updated_at = ?
        WHERE id = ? AND state IN ('candidate', 'rendered')
        """,
        (score, reason, file_path, rank, now, clip_id),
    )
    if cursor.rowcount != 1:
        raise ClipError(f"clip {clip_id} was not updated")


def delete_clip(conn: sqlite3.Connection, clip_id: str) -> None:
    conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))


def approve_clip(
    conn: sqlite3.Connection,
    clip_id: str,
    platforms: frozenset[str] | set[str],
    now: str,
) -> None:
    if not platforms:
        raise ClipError("approval needs at least one platform")
    row = get_clip(conn, clip_id)
    if row is None:
        raise ClipError(f"clip not found: {clip_id}")
    if row["state"] not in ("rendered", "approved"):
        raise ClipError("only a rendered clip can be approved")
    for platform in sorted(platforms):
        if platform not in ("youtube", "instagram", "tiktok"):
            raise ClipError(f"unknown platform: {platform}")
        conn.execute(
            "INSERT OR IGNORE INTO clip_platforms (clip_id, platform) VALUES (?, ?)",
            (clip_id, platform),
        )
    conn.execute(
        "UPDATE clips SET state = 'approved', updated_at = ? WHERE id = ?",
        (now, clip_id),
    )


def reject_clip(conn: sqlite3.Connection, clip_id: str, note: str, now: str) -> None:
    if not note or not note.strip():
        raise ClipError("rejection needs a note")
    row = get_clip(conn, clip_id)
    if row is None:
        raise ClipError(f"clip not found: {clip_id}")
    if row["state"] not in ("rendered", "rejected"):
        raise ClipError("only a rendered clip can be rejected")
    conn.execute(
        """
        UPDATE clips
        SET state = 'rejected', rejection_note = ?, updated_at = ?
        WHERE id = ?
        """,
        (note.strip(), now, clip_id),
    )


def get_publication(conn: sqlite3.Connection, publication_id: str) -> Publication | None:
    row = conn.execute("SELECT * FROM publications WHERE id = ?", (publication_id,)).fetchone()
    if row is None:
        return None
    return publication_from_row(row)


def get_publication_for_clip(
    conn: sqlite3.Connection, clip_id: str, platform: str
) -> Publication | None:
    row = conn.execute(
        "SELECT * FROM publications WHERE clip_id = ? AND platform = ?",
        (clip_id, platform),
    ).fetchone()
    if row is None:
        return None
    return publication_from_row(row)


def ensure_queued(conn: sqlite3.Connection, publication: Publication, now: str) -> Publication:
    """Insert a queued publication once. A second call returns the stored row."""

    if publication.state != "queued" or publication.handle is not None:
        raise ClipError("a new publication starts queued and has no handle")
    existing = get_publication_for_clip(conn, publication.clip_id, publication.platform)
    if existing is not None:
        return existing
    save_publication(conn, publication, now)
    return publication


def save_publication(conn: sqlite3.Connection, publication: Publication, now: str) -> Publication:
    """Write a publication. An equal row is left untouched, including updated_at."""

    current = get_publication(conn, publication.id)
    if current == publication:
        return current
    payload = _publication_values(publication)
    if current is None:
        conn.execute(
            """
            INSERT INTO publications (
              id, clip_id, platform, title, caption, hashtags, state, handle,
              remote_id, remote_url, failure_reason, retryable, route, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*payload, now, now),
        )
        return publication
    conn.execute(
        """
        UPDATE publications
        SET title = ?, caption = ?, hashtags = ?, state = ?, handle = ?,
            remote_id = ?, remote_url = ?, failure_reason = ?, retryable = ?,
            route = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            payload[3],
            payload[4],
            payload[5],
            payload[6],
            payload[7],
            payload[8],
            payload[9],
            payload[10],
            payload[11],
            payload[12],
            now,
            publication.id,
        ),
    )
    return publication


def set_account_pause(conn: sqlite3.Connection, platform: str, reason: str) -> None:
    if platform not in ("youtube", "instagram", "tiktok"):
        raise ClipError(f"unknown platform: {platform}")
    if not reason or not reason.strip():
        raise ClipError("pause reason is required")
    conn.execute(
        """
        INSERT INTO accounts (platform, paused_until, pause_reason)
        VALUES (?, NULL, ?)
        ON CONFLICT(platform) DO UPDATE SET
          pause_reason = excluded.pause_reason,
          paused_until = NULL
        """,
        (platform, reason.strip()),
    )


def clear_account_pause(conn: sqlite3.Connection, platform: str) -> None:
    if platform not in ("youtube", "instagram", "tiktok"):
        raise ClipError(f"unknown platform: {platform}")
    conn.execute("DELETE FROM accounts WHERE platform = ?", (platform,))


def publication_from_row(row: sqlite3.Row) -> Publication:
    hashtags = tuple(json.loads(row["hashtags"]))
    retryable = None if row["retryable"] is None else bool(row["retryable"])
    shared = dict(
        id=row["id"],
        clip_id=row["clip_id"],
        title=row["title"],
        caption=row["caption"],
        hashtags=hashtags,
        state=row["state"],
        handle=row["handle"],
        remote_id=row["remote_id"],
        remote_url=row["remote_url"],
        failure_reason=row["failure_reason"],
        retryable=retryable,
    )
    if row["platform"] == "youtube":
        return YouTubePub(**shared)
    if row["platform"] == "instagram":
        return InstagramPub(**shared)
    if row["platform"] == "tiktok":
        return TikTokPub(**shared, route=row["route"])
    raise ClipError(f"unknown platform on publication: {row['platform']}")


def _publication_values(publication: Publication) -> tuple:
    route = publication.route if isinstance(publication, TikTokPub) else None
    retryable = None if publication.retryable is None else int(publication.retryable)
    return (
        publication.id,
        publication.clip_id,
        publication.platform,
        publication.title,
        publication.caption,
        json.dumps(list(publication.hashtags)),
        publication.state,
        publication.handle,
        publication.remote_id,
        publication.remote_url,
        publication.failure_reason,
        retryable,
        route,
    )
