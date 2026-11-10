"""Parse a Deepgram pre-recorded Nova-3 payload into sentences and words."""

from __future__ import annotations

from dataclasses import dataclass

from clip.errors import ClipError


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float
    confidence: float


@dataclass(frozen=True)
class Sentence:
    text: str
    start_s: float
    end_s: float
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Transcript:
    words: tuple[Word, ...]
    sentences: tuple[Sentence, ...]
    duration_s: float


def parse_deepgram(payload: dict) -> Transcript:
    try:
        alternative = payload["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClipError(
            "Deepgram response is missing results.channels[0].alternatives[0]"
        ) from exc

    raw_words = alternative.get("words") or []
    words = tuple(_parse_word(item) for item in raw_words)
    sentences = _sentences_from_paragraphs(alternative, words)
    if not sentences:
        sentences = _sentences_from_punctuation(words)
    duration = _duration(payload, words)
    return Transcript(words=words, sentences=sentences, duration_s=duration)


def _parse_word(item: dict) -> Word:
    try:
        text = item.get("punctuated_word") or item["word"]
        return Word(
            text=str(text),
            start_s=float(item["start"]),
            end_s=float(item["end"]),
            confidence=float(item.get("confidence", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ClipError(f"Deepgram word is missing start, end, or text: {item!r}") from exc


def _duration(payload: dict, words: tuple[Word, ...]) -> float:
    metadata = payload.get("metadata") or {}
    if "duration" in metadata:
        return float(metadata["duration"])
    if words:
        return words[-1].end_s
    return 0.0


def _sentences_from_paragraphs(
    alternative: dict, words: tuple[Word, ...]
) -> tuple[Sentence, ...]:
    paragraphs = (alternative.get("paragraphs") or {}).get("paragraphs") or []
    specs: list[tuple[str, float, float]] = []
    for paragraph in paragraphs:
        for sentence in paragraph.get("sentences") or []:
            try:
                specs.append(
                    (str(sentence["text"]), float(sentence["start"]), float(sentence["end"]))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ClipError(f"Deepgram sentence is missing text or times: {sentence!r}") from exc
    if not specs:
        return ()
    return _assign_words(specs, words)


def _assign_words(
    specs: list[tuple[str, float, float]], words: tuple[Word, ...]
) -> tuple[Sentence, ...]:
    sentences: list[Sentence] = []
    index = 0
    for text, start_s, end_s in specs:
        grouped: list[Word] = []
        while index < len(words) and words[index].start_s < end_s - 1e-3:
            word = words[index]
            index += 1
            if word.end_s > start_s + 1e-6:
                grouped.append(word)
        sentences.append(Sentence(text=text, start_s=start_s, end_s=end_s, words=tuple(grouped)))
    return tuple(sentences)


def _sentences_from_punctuation(words: tuple[Word, ...]) -> tuple[Sentence, ...]:
    if not words:
        return ()
    sentences: list[Sentence] = []
    bucket: list[Word] = []
    for word in words:
        bucket.append(word)
        if word.text.endswith((".", "?", "!")):
            sentences.append(_sentence_from_words(bucket))
            bucket = []
    if bucket:
        sentences.append(_sentence_from_words(bucket))
    return tuple(sentences)


def _sentence_from_words(words: list[Word]) -> Sentence:
    return Sentence(
        text=" ".join(word.text for word in words),
        start_s=words[0].start_s,
        end_s=words[-1].end_s,
        words=tuple(words),
    )
