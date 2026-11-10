"""Domain types for sources, clips, and per-platform publications.

Phase 0 renders clips. Phase 1 approves them and publishes to YouTube through
these types: a publication is built only from an approved clip, the upload
handle is stored before any live side effect, and reconcile is pure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal

Platform = Literal["youtube", "instagram", "tiktok"]
Rights = Literal["own", "licensed"]
TikTokRoute = Literal["upload_post", "manual"]
YouTubeState = Literal["queued", "uploading", "verifying", "published", "failed"]
InstagramState = Literal["queued", "container", "ready", "verifying", "published", "failed"]
TikTokState = Literal["queued", "submitted", "verifying", "published", "failed"]

PLATFORMS: frozenset[str] = frozenset({"youtube", "instagram", "tiktok"})
YOUTUBE_STATES: frozenset[str] = frozenset(
    {"queued", "uploading", "verifying", "published", "failed"}
)
INSTAGRAM_STATES: frozenset[str] = frozenset(
    {"queued", "container", "ready", "verifying", "published", "failed"}
)
TIKTOK_STATES: frozenset[str] = frozenset(
    {"queued", "submitted", "verifying", "published", "failed"}
)
TIKTOK_ROUTES: frozenset[str] = frozenset({"upload_post", "manual"})
SHORTS_CAP_S = 60.0


def _publication_id(clip_id: str, platform: str) -> str:
    return hashlib.sha256(f"{clip_id}:{platform}".encode()).hexdigest()[:32]


@dataclass(frozen=True)
class Span:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.start_s >= self.end_s:
            raise ValueError(
                f"span start must be >= 0 and before end, got {self.start_s}..{self.end_s}"
            )
        if self.end_s - self.start_s > SHORTS_CAP_S:
            raise ValueError(
                f"span exceeds the 60s Shorts cap ({self.end_s - self.start_s:.3f}s)"
            )


@dataclass(frozen=True)
class SourceVideo:
    id: str
    path: str
    duration_s: float
    rights: Rights

    def __post_init__(self) -> None:
        if self.rights not in ("own", "licensed"):
            raise ValueError("rights must be own or licensed")
        if self.duration_s < 0:
            raise ValueError("duration must be >= 0")
        if not self.id or not self.path:
            raise ValueError("source id and path are required")


@dataclass(frozen=True)
class Candidate:
    id: str
    source_id: str
    span: Span
    score: float
    reason: str


@dataclass(frozen=True)
class Rendered:
    candidate: Candidate
    file: str

    def __post_init__(self) -> None:
        if not self.file:
            raise ValueError("a rendered clip has a file")


@dataclass(frozen=True)
class Approved:
    rendered: Rendered
    platforms: frozenset[Platform]

    def __post_init__(self) -> None:
        if not self.platforms:
            raise ValueError("approval needs at least one platform")
        unknown = set(self.platforms) - PLATFORMS
        if unknown:
            raise ValueError(f"unknown platform: {sorted(unknown)}")


@dataclass(frozen=True)
class Rejected:
    rendered: Rendered
    note: str

    def __post_init__(self) -> None:
        if not self.note or not self.note.strip():
            raise ValueError("rejection needs a note")


Clip = Candidate | Rendered | Approved | Rejected


@dataclass(frozen=True)
class PostSpec:
    platform: Platform
    title: str
    caption: str
    hashtags: tuple[str, ...]
    route: TikTokRoute | None = None

    def __post_init__(self) -> None:
        if self.platform not in PLATFORMS:
            raise ValueError(f"unknown platform: {self.platform}")
        if not self.title.strip():
            raise ValueError("title is required")
        if self.platform == "tiktok" and self.route not in TIKTOK_ROUTES:
            raise ValueError("TikTok publications use Upload-Post or manual posting")
        if self.platform != "tiktok" and self.route is not None:
            raise ValueError("route is only set on TikTok publications")


@dataclass(frozen=True)
class YouTubePub:
    id: str
    clip_id: str
    title: str
    caption: str
    hashtags: tuple[str, ...]
    state: YouTubeState
    handle: str | None = None
    remote_id: str | None = None
    remote_url: str | None = None
    failure_reason: str | None = None
    retryable: bool | None = None
    platform: Literal["youtube"] = "youtube"

    def __post_init__(self) -> None:
        if self.platform != "youtube":
            raise ValueError("YouTubePub.platform is youtube")
        if self.state not in YOUTUBE_STATES:
            raise ValueError(f"unknown YouTube state: {self.state}")


@dataclass(frozen=True)
class InstagramPub:
    id: str
    clip_id: str
    title: str
    caption: str
    hashtags: tuple[str, ...]
    state: InstagramState
    handle: str | None = None
    remote_id: str | None = None
    remote_url: str | None = None
    failure_reason: str | None = None
    retryable: bool | None = None
    platform: Literal["instagram"] = "instagram"

    def __post_init__(self) -> None:
        if self.platform != "instagram":
            raise ValueError("InstagramPub.platform is instagram")
        if self.state not in INSTAGRAM_STATES:
            raise ValueError(f"unknown Instagram state: {self.state}")


@dataclass(frozen=True)
class TikTokPub:
    id: str
    clip_id: str
    title: str
    caption: str
    hashtags: tuple[str, ...]
    state: TikTokState
    route: TikTokRoute
    handle: str | None = None
    remote_id: str | None = None
    remote_url: str | None = None
    failure_reason: str | None = None
    retryable: bool | None = None
    platform: Literal["tiktok"] = "tiktok"

    def __post_init__(self) -> None:
        if self.platform != "tiktok":
            raise ValueError("TikTokPub.platform is tiktok")
        if self.state not in TIKTOK_STATES:
            raise ValueError(f"unknown TikTok state: {self.state}")
        if self.route not in TIKTOK_ROUTES:
            raise ValueError("TikTok publications use Upload-Post or manual posting")


Publication = YouTubePub | InstagramPub | TikTokPub


@dataclass(frozen=True)
class RemoteState:
    """What a publisher observed. reconcile never performs I/O."""

    upload_complete: bool = False
    processed: bool = False
    live: bool = False
    policy_error: str | None = None
    retryable: bool = False
    remote_id: str | None = None
    remote_url: str | None = None
    expired: bool = False
    copyright_blocked: bool = False


def approve(rendered: Rendered, platforms: frozenset[Platform]) -> Approved:
    return Approved(rendered, platforms)


def reject(rendered: Rendered, note: str) -> Rejected:
    return Rejected(rendered, note.strip())


def publications_for(clip: Approved, specs: list[PostSpec]) -> list[Publication]:
    """Build queued publication rows. The only constructor takes an Approved clip."""

    by_platform: dict[str, PostSpec] = {}
    for spec in specs:
        if spec.platform in by_platform:
            raise ValueError(f"duplicate spec for {spec.platform}")
        by_platform[spec.platform] = spec
    missing = set(clip.platforms) - by_platform.keys()
    extra = by_platform.keys() - set(clip.platforms)
    if extra:
        raise ValueError(f"spec platform is not approved: {sorted(extra)}")
    if missing:
        raise ValueError(f"missing spec for approved platform: {sorted(missing)}")

    publications: list[Publication] = []
    for platform in ("youtube", "instagram", "tiktok"):
        if platform not in clip.platforms:
            continue
        spec = by_platform[platform]
        shared = dict(
            id=_publication_id(clip.rendered.candidate.id, platform),
            clip_id=clip.rendered.candidate.id,
            title=spec.title,
            caption=spec.caption,
            hashtags=spec.hashtags,
            state="queued",
        )
        if platform == "youtube":
            publications.append(YouTubePub(**shared))
        elif platform == "instagram":
            publications.append(InstagramPub(**shared))
        else:
            if spec.route is None:
                raise ValueError("TikTok publications use Upload-Post or manual posting")
            publications.append(TikTokPub(**shared, route=spec.route))
    return publications


def with_handle(row: Publication, handle: str) -> Publication:
    """Persist the platform handle and leave `queued` in one value.

    The publisher writes this row before the call that can make a post live.
    A later crash resumes from the handle and does not open a second session.
    """

    cleaned = handle.strip()
    if not cleaned:
        raise ValueError("handle is required before a side effect")
    if row.state in ("published", "failed"):
        return row
    if isinstance(row, YouTubePub):
        state = "uploading" if row.state == "queued" else row.state
        return replace(row, handle=cleaned, state=state)
    if isinstance(row, InstagramPub):
        state = "container" if row.state == "queued" else row.state
        return replace(row, handle=cleaned, state=state)
    if isinstance(row, TikTokPub):
        state = "submitted" if row.state == "queued" else row.state
        return replace(row, handle=cleaned, state=state)
    raise TypeError(f"unknown publication: {type(row)!r}")


def reconcile(row: Publication, platform_state: RemoteState) -> Publication:
    """Pure. A published or failed row comes back unchanged."""

    if isinstance(row, YouTubePub):
        return _reconcile_youtube(row, platform_state)
    if isinstance(row, InstagramPub):
        return _reconcile_instagram(row, platform_state)
    if isinstance(row, TikTokPub):
        return _reconcile_tiktok(row, platform_state)
    raise TypeError(f"unknown publication: {type(row)!r}")


def _reconcile_youtube(row: YouTubePub, remote: RemoteState) -> YouTubePub:
    if row.state in ("published", "failed"):
        return row
    if remote.policy_error:
        return replace(row, state="failed", failure_reason=remote.policy_error, retryable=False)
    # No session URI yet: do not advance, and do not call videos.insert from here.
    if row.handle is None:
        return row
    if remote.processed and remote.remote_id:
        return replace(
            row,
            state="published",
            remote_id=remote.remote_id,
            remote_url=remote.remote_url,
            failure_reason=None,
            retryable=None,
        )
    if remote.upload_complete and remote.remote_id:
        if row.state == "verifying" and row.remote_id == remote.remote_id:
            return row
        return replace(row, state="verifying", remote_id=remote.remote_id)
    if row.state == "uploading":
        return row
    return replace(row, state="uploading")


def _reconcile_instagram(row: InstagramPub, remote: RemoteState) -> InstagramPub:
    if row.state in ("published", "failed"):
        return row
    if remote.policy_error or remote.copyright_blocked:
        reason = remote.policy_error or "copyright claim"
        return replace(row, state="failed", failure_reason=reason, retryable=False)
    # An expired container never went public, so it is safe to create another.
    if row.state == "container" and remote.expired:
        return replace(row, state="queued", handle=None, failure_reason=None, retryable=None)
    if row.handle is None:
        return row
    if remote.live and remote.remote_id and row.state in ("verifying", "ready"):
        return replace(
            row,
            state="published",
            remote_id=remote.remote_id,
            remote_url=remote.remote_url,
            failure_reason=None,
            retryable=None,
        )
    if row.state == "container" and remote.processed:
        return replace(row, state="ready")
    if row.state == "queued":
        return replace(row, state="container")
    return row


def _reconcile_tiktok(row: TikTokPub, remote: RemoteState) -> TikTokPub:
    if row.state in ("published", "failed"):
        return row
    if remote.policy_error:
        return replace(row, state="failed", failure_reason=remote.policy_error, retryable=False)
    if row.handle is None:
        return row
    if remote.live and remote.remote_id:
        return replace(
            row,
            state="published",
            remote_id=remote.remote_id,
            remote_url=remote.remote_url,
            failure_reason=None,
            retryable=None,
        )
    if row.state == "submitted" and (remote.upload_complete or remote.processed):
        return replace(row, state="verifying")
    if row.state == "verifying":
        return row
    if row.state == "queued":
        return replace(row, state="submitted")
    return row
