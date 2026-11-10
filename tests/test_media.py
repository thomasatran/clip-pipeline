from clip.media import caption_cues, center_crop_9_16, escape_filter_path
from clip.transcript import Word
from pathlib import Path


def test_center_crop_is_even_and_inside_the_frame() -> None:
    wide_w, wide_h, wide_x, wide_y = center_crop_9_16(1920, 1080)
    assert wide_w % 2 == 0 and wide_h % 2 == 0
    assert wide_w <= 1920 and wide_h <= 1080
    assert wide_x >= 0 and wide_y >= 0
    assert wide_w / wide_h == 606 / 1080

    portrait_w, portrait_h, portrait_x, portrait_y = center_crop_9_16(1080, 1920)
    assert (portrait_w, portrait_h, portrait_x, portrait_y) == (1080, 1920, 0, 0)

    square_w, square_h, _, _ = center_crop_9_16(720, 1280)
    assert square_w == 720
    assert square_h == 1280


def test_caption_cues_are_relative_to_the_clip_and_keep_punctuation() -> None:
    words = (
        Word("Why", 0.40, 0.90, 0.99),
        Word("does", 0.90, 1.40, 0.99),
        Word("this", 1.40, 1.90, 0.99),
        Word("matter?", 1.90, 2.40, 0.99),
        Word("Later.", 7.00, 7.40, 0.99),
    )
    cues = caption_cues(words, 0.40, 11.20)
    assert cues[0][2] == "Why does this matter?"
    assert cues[0][0] == 0
    assert cues[-1][2] == "Later."


def test_filter_paths_escape_ffmpeg_separators() -> None:
    escaped = escape_filter_path(Path("/tmp/a:b,c.ass"))
    assert "\\:" in escaped
    assert "\\," in escaped
