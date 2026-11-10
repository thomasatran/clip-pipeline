import json

from clip.domain import Span
from clip.rank import Pick, PickError, rank_candidates, score_window, snap_span, validate_picks
from clip.transcript import Sentence, Transcript, Word, parse_deepgram
from tests.conftest import FIXTURE


def sent(text: str, start: float, end: float, confidence: float = 0.9) -> Sentence:
    word = Word(text, start, end, confidence)
    return Sentence(text, start, end, (word,))


def transcript_of(*sentences: Sentence, duration: float | None = None) -> Transcript:
    words = tuple(word for sentence in sentences for word in sentence.words)
    end = duration if duration is not None else (sentences[-1].end_s if sentences else 0.0)
    return Transcript(words, sentences, end)


def test_question_hook_outranks_an_equal_statement() -> None:
    hook = sent("Is this the hook?", 0, 12)
    statement = sent("This is a statement.", 0, 12)
    hook_score, hook_reason = score_window((hook,))
    statement_score, _statement_reason = score_window((statement,))
    assert hook_score > statement_score
    assert "question hook" in hook_reason

    later = sent("This is a statement.", 80, 92)
    picks = rank_candidates(transcript_of(hook, later, duration=100), 100, top_n=2)
    assert [pick.span for pick in picks] == [Span(0, 12), Span(80, 92)]


def test_spans_stay_within_60_seconds_and_do_not_overlap() -> None:
    sentences = tuple(sent(f"Sentence {index}.", index * 10, (index + 1) * 10) for index in range(9))
    picks = rank_candidates(transcript_of(*sentences, duration=90), 90, top_n=5)
    assert len(picks) == 5
    assert all(pick.span.end_s - pick.span.start_s <= 60 for pick in picks)
    ordered = sorted(picks, key=lambda pick: pick.span.start_s)
    for left, right in zip(ordered, ordered[1:]):
        assert left.span.end_s <= right.span.start_s


def test_short_and_empty_transcripts_produce_no_clips() -> None:
    short = sent("Too short.", 0, 3)
    assert rank_candidates(transcript_of(short, duration=3), 3) == []
    assert rank_candidates(Transcript((), (), 0.0), 10) == []


def test_windows_past_the_source_duration_are_dropped() -> None:
    sentence = sent("Hello there friend.", 0, 12)
    assert rank_candidates(transcript_of(sentence, duration=12), 5) == []


def test_top_n_limits_the_result() -> None:
    sentences = tuple(sent(f"Part {index}.", index * 20, index * 20 + 12) for index in range(4))
    picks = rank_candidates(transcript_of(*sentences, duration=80), 80, top_n=2)
    assert len(picks) == 2


def test_snap_aligns_to_sentence_edges_and_rejects_over_60() -> None:
    sentences = (
        sent("First.", 0.5, 2.0),
        sent("Second.", 2.2, 4.0),
    )
    assert snap_span(1.0, 3.0, sentences) == Span(0.5, 4.0)
    long = (sent("Left.", 0, 30), sent("Right.", 30, 70))
    assert snap_span(0, 50, long) is None


def test_sample_fixture_keeps_the_question_hook() -> None:
    transcript = parse_deepgram(json.loads(FIXTURE.read_text(encoding="utf-8")))
    picks = rank_candidates(transcript, 12, top_n=5)
    assert len(picks) == 1
    assert picks[0].span == Span(0.4, 11.2)
    assert "question hook" in picks[0].reason


def test_validate_rejects_overlap() -> None:
    sentences = (sent("Left.", 0, 10), sent("Right.", 9, 20))
    picks = [Pick(Span(0, 10), 1.0, "a"), Pick(Span(9, 20), 1.0, "b")]
    try:
        validate_picks(picks, sentences, 30)
    except PickError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected an overlap error")
