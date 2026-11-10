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


def _approved(platforms):
    rendered = Rendered(
        Candidate("clip1", "source1", Span(0, 10), 0.8, "duration 10.0s"),
        "/tmp/clip.mp4",
    )
    return approve(rendered, platforms)


def test_span_rejects_non_positive_and_over_60_seconds() -> None:
    Span(0, 60)
    for start, end in ((0, 0), (5, 5), (-1, 4), (0, 60.001), (10, 80)):
        try:
            Span(start, end)
        except ValueError:
            continue
        raise AssertionError(f"span {start}..{end} should have been rejected")


def test_publications_come_only_from_approved_platforms() -> None:
    clip = _approved(frozenset({"youtube"}))
    try:
        publications_for(
            clip,
            [PostSpec("youtube", "Title", "Caption", ("#shorts",)), PostSpec("instagram", "T", "C", ())],
        )
    except ValueError as exc:
        assert "not approved" in str(exc)
    else:
        raise AssertionError("expected a platform error")

    try:
        PostSpec("tiktok", "Title", "Caption", ())
    except ValueError as exc:
        assert "TikTok" in str(exc)
    else:
        raise AssertionError("expected a TikTok route error")


def test_youtube_reconcile_keeps_the_handle_and_a_published_row_is_a_no_op() -> None:
    queued = publications_for(
        _approved(frozenset({"youtube"})),
        [PostSpec("youtube", "Title", "Caption", ("#shorts",))],
    )[0]
    assert queued.state == "queued"
    assert queued.handle is None
    assert reconcile(queued, RemoteState(processed=True, remote_id="vid")) is queued
    try:
        with_handle(queued, "  ")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty handle must be rejected")

    uploading = with_handle(queued, "https://example.test/session")
    assert uploading.state == "uploading"
    assert uploading.handle == "https://example.test/session"
    assert reconcile(uploading, RemoteState()) is uploading

    verifying = reconcile(uploading, RemoteState(upload_complete=True, remote_id="vid123"))
    assert verifying.state == "verifying"
    assert verifying.handle == uploading.handle
    assert verifying.remote_id == "vid123"

    published = reconcile(
        verifying,
        RemoteState(processed=True, remote_id="vid123", remote_url="https://youtu.be/vid123"),
    )
    assert published.state == "published"
    assert reconcile(published, RemoteState(policy_error="late")) is published

    failed = reconcile(uploading, RemoteState(policy_error="spam"))
    assert failed.state == "failed"
    assert failed.retryable is False
    assert failed.handle == uploading.handle
    assert reconcile(failed, RemoteState(live=True, remote_id="other")) is failed


def test_instagram_expired_container_can_be_created_again() -> None:
    queued = publications_for(
        _approved(frozenset({"instagram"})),
        [PostSpec("instagram", "Title", "Caption", ())],
    )[0]
    container = with_handle(queued, "container-1")
    assert container.state == "container"
    expired = reconcile(container, RemoteState(expired=True))
    assert expired.state == "queued"
    assert expired.handle is None
    claimed = reconcile(container, RemoteState(copyright_blocked=True))
    assert claimed.state == "failed"
    assert claimed.failure_reason == "copyright claim"


def test_tiktok_uses_upload_post_or_manual_and_keeps_its_handle() -> None:
    queued = publications_for(
        _approved(frozenset({"tiktok"})),
        [PostSpec("tiktok", "Title", "Caption", (), route="upload_post")],
    )[0]
    assert queued.route == "upload_post"
    submitted = with_handle(queued, "upload-post-request-9")
    assert submitted.state == "submitted"
    assert submitted.handle == "upload-post-request-9"
    published = reconcile(submitted, RemoteState(live=True, remote_id="tt1", remote_url="https://tiktok.example/tt1"))
    assert published.state == "published"
    assert published.route == "upload_post"
    assert reconcile(published, RemoteState()) is published

    manual = publications_for(
        _approved(frozenset({"tiktok"})),
        [PostSpec("tiktok", "Title", "Caption", (), route="manual")],
    )[0]
    assert manual.route == "manual"
