"""Pick sentence-aligned clips and reject anything that fails the span checks.

Phase 0 scores windows locally so a run does not need a second API key.
`snap_span` and `validate_picks` are the checks an LLM picker has to pass:
sentence edges, range, order, overlap, and the 60 second cap.
"""

from __future__ import annotations

from dataclasses import dataclass

from clip.domain import SHORTS_CAP_S, Span
from clip.transcript import Sentence, Transcript

MIN_CLIP_S = 8.0
TARGET_S = 24.0
HOOK_BONUS = 0.25


class PickError(ValueError):
    """A candidate failed a span check after sentence snapping."""


@dataclass(frozen=True)
class Pick:
    span: Span
    score: float
    reason: str


def rank_candidates(
    transcript: Transcript, duration_s: float, top_n: int = 5
) -> list[Pick]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if duration_s < 0:
        raise ValueError("duration must be >= 0")

    sentences = transcript.sentences
    proposed: list[tuple[float, Span, str]] = []
    for start_index in range(len(sentences)):
        for end_index in range(start_index, len(sentences)):
            group = sentences[start_index : end_index + 1]
            start_s = group[0].start_s
            end_s = group[-1].end_s
            length = end_s - start_s
            if length > SHORTS_CAP_S:
                break
            if end_s > duration_s + 0.05:
                break
            if length < MIN_CLIP_S:
                continue
            score, reason = score_window(group)
            proposed.append((score, Span(start_s, end_s), reason))

    ordered = sorted(proposed, key=lambda item: (-item[0], item[1].start_s, item[1].end_s))
    chosen: list[Pick] = []
    for score, span, reason in ordered:
        if any(_overlaps(span, pick.span) for pick in chosen):
            continue
        snapped = snap_span(span.start_s, span.end_s, sentences)
        if snapped is None:
            continue
        chosen.append(Pick(snapped, round(score, 6), reason))
        if len(chosen) == top_n:
            break

    validate_picks(chosen, sentences, duration_s)
    return chosen


def score_window(sentences: tuple[Sentence, ...] | list[Sentence]) -> tuple[float, str]:
    start_s = sentences[0].start_s
    end_s = sentences[-1].end_s
    duration = end_s - start_s
    length_score = max(0.0, 1.0 - abs(duration - TARGET_S) / TARGET_S)
    confidences = [word.confidence for sentence in sentences for word in sentence.words]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    hook = "?" in sentences[0].text
    score = 0.65 * length_score + 0.35 * confidence + (HOOK_BONUS if hook else 0.0)
    reason_parts = [
        f"duration {duration:.1f}s",
        f"confidence {confidence:.2f}",
        f"sentences {len(sentences)}",
    ]
    if hook:
        reason_parts.append("question hook")
    return score, ", ".join(reason_parts)


def snap_span(
    start_s: float, end_s: float, sentences: tuple[Sentence, ...] | list[Sentence]
) -> Span | None:
    """Expand a proposed range to the sentence edges it touches."""

    covering = [
        sentence
        for sentence in sentences
        if sentence.end_s > start_s and sentence.start_s < end_s
    ]
    if not covering:
        return None
    snapped_start = covering[0].start_s
    snapped_end = covering[-1].end_s
    if snapped_end - snapped_start > SHORTS_CAP_S:
        return None
    try:
        return Span(snapped_start, snapped_end)
    except ValueError:
        return None


def validate_picks(
    picks: list[Pick],
    sentences: tuple[Sentence, ...] | list[Sentence],
    duration_s: float,
) -> None:
    starts = {_millis(sentence.start_s) for sentence in sentences}
    ends = {_millis(sentence.end_s) for sentence in sentences}
    ordered = sorted(picks, key=lambda pick: (pick.span.start_s, pick.span.end_s))
    previous_end = 0
    for pick in ordered:
        span = pick.span
        if span.end_s - span.start_s > SHORTS_CAP_S:
            raise PickError(f"clip is over 60s ({span.end_s - span.start_s:.3f}s)")
        if span.end_s > duration_s + 0.05:
            raise PickError(
                f"clip ends at {span.end_s:.3f}s, past the source duration {duration_s:.3f}s"
            )
        if _millis(span.start_s) not in starts or _millis(span.end_s) not in ends:
            raise PickError(
                f"clip {span.start_s:.3f}-{span.end_s:.3f} is not on sentence edges"
            )
        start_ms = _millis(span.start_s)
        end_ms = _millis(span.end_s)
        if start_ms < previous_end:
            raise PickError("clips overlap")
        previous_end = end_ms


def _overlaps(left: Span, right: Span) -> bool:
    return left.start_s < right.end_s and right.start_s < left.end_s


def _millis(seconds: float) -> int:
    return int(round(seconds * 1000))
