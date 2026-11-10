import json
import urllib.parse
from pathlib import Path

from clip.deepgram import content_type_for, listen_url, request_headers
from clip.errors import ClipError
from clip.transcript import parse_deepgram
from tests.conftest import FIXTURE


def test_fixture_parses_sentences_and_assigns_every_word() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    transcript = parse_deepgram(payload)
    assert [sentence.text for sentence in transcript.sentences] == [
        "Why does this matter?",
        "The useful point is that captions follow the spoken words.",
        "That is the whole clip and it stands on its own.",
    ]
    assigned = [word.text for sentence in transcript.sentences for word in sentence.words]
    assert assigned == [word.text for word in transcript.words]
    assert assigned[0] == "Why"
    assert assigned[3] == "matter?"
    assert len(transcript.words) == 25


def test_punctuation_builds_sentences_when_paragraphs_are_absent() -> None:
    payload = {
        "metadata": {"duration": 2.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello there. Next line.",
                            "words": [
                                {
                                    "word": "hello",
                                    "start": 0.0,
                                    "end": 0.4,
                                    "confidence": 0.9,
                                    "punctuated_word": "Hello",
                                },
                                {
                                    "word": "there",
                                    "start": 0.4,
                                    "end": 0.8,
                                    "confidence": 0.9,
                                    "punctuated_word": "there.",
                                },
                                {
                                    "word": "next",
                                    "start": 1.0,
                                    "end": 1.3,
                                    "confidence": 0.9,
                                    "punctuated_word": "Next",
                                },
                                {
                                    "word": "line",
                                    "start": 1.3,
                                    "end": 1.8,
                                    "confidence": 0.9,
                                    "punctuated_word": "line.",
                                },
                            ],
                        }
                    ]
                }
            ]
        },
    }
    transcript = parse_deepgram(payload)
    assert [sentence.text for sentence in transcript.sentences] == ["Hello there.", "Next line."]
    assert transcript.sentences[0].start_s == 0.0
    assert transcript.sentences[0].end_s == 0.8


def test_missing_alternative_is_an_error() -> None:
    try:
        parse_deepgram({"results": {"channels": []}})
    except ClipError as exc:
        assert "alternatives" in str(exc)
    else:
        raise AssertionError("expected a parse error")


def test_listen_request_is_nova3_monolingual_with_word_formatting() -> None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(listen_url()).query)
    assert query == {
        "model": ["nova-3"],
        "language": ["en"],
        "smart_format": ["true"],
        "paragraphs": ["true"],
    }
    headers = request_headers("secret", content_type_for(Path("audio.wav")))
    assert headers["Authorization"] == "Token secret"
    assert headers["Content-Type"] == "audio/wav"
