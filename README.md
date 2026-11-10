# clip-pipeline

Phase 0 turns one video you own or have licensed into ranked, captioned 9:16 clips.

`clip run file.mp4` normalizes the file, transcribes it with Deepgram Nova-3, picks sentence-aligned clips of at most 60 seconds, and writes 1080x1920 files with burned-in captions. Runs and clips are stored in SQLite. Running the same file again does not add rows or rewrite files.

Publishers are not built yet. The database and types already have per-platform approval, a publication handle saved before any live side effect, and a pure `reconcile` step so phase 1 can add a YouTube publisher on this schema.

## Setup

Python 3.11 or newer, and ffmpeg with libx264, AAC, and libass. Captions use DejaVu Sans (`fonts-dejavu-core` on Debian or Ubuntu).

```bash
python -m pip install -e ".[dev]"
```

There is no Deepgram key in a fixture run. A live transcription reads `DEEPGRAM_API_KEY`.

## Fixture run

Generate a short sample, then rank and render it from the checked-in transcript:

```bash
ffmpeg -y -f lavfi -i testsrc=size=640x360:rate=30:duration=12 \
  -f lavfi -i sine=frequency=440:sample_rate=48000:duration=12 \
  -shortest -pix_fmt yuv420p -c:v libx264 -c:a aac sample.mp4

clip run sample.mp4 --rights own --transcript tests/fixtures/sample_transcript.json
```

`--rights` is `own` or `licensed`. Anything else is rejected.

Clips land in `out/<sha256>/`. The normalized 1920x1080 file and the transcript JSON land in `data/work/<sha256>/`. The database is `data/clip.sqlite`.

Run the same command again. The status line says `rendered, no-op`. The source, run, and clip rows stay as they are, and the mp4 bytes stay as they are. The source id is the SHA-256 of the file bytes, so renaming the file still matches the same source. A different `--rights` value on that source is an error.

## Live run

```bash
export DEEPGRAM_API_KEY=your_key
clip run /path/to/video.mp4 --rights own
```

The live path extracts 16 kHz mono WAV from the normalized video and sends it to Deepgram's synchronous pre-recorded endpoint:

```text
POST https://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true&paragraphs=true
Authorization: Token $DEEPGRAM_API_KEY
Content-Type: audio/wav
```

Word `start` and `end` times come back in seconds without an extra timestamp parameter. `smart_format=true` adds `punctuated_word`. `paragraphs=true` adds sentence start and end times, which the ranker snaps to. MP4 is also a documented upload format; this pipeline sends WAV so the body is the audio track. Timestamps still start at zero because the WAV is the full normalized soundtrack.

Checked against Deepgram's docs on 2026-11-10:

- Pre-recorded Nova-3 monolingual pay as you go is $0.0043 per minute, billed per second, which is $0.258 per hour. That is the rate this call uses (`language=en`). Streaming Nova-3 is $0.0077 per minute. Pre-recorded multilingual is $0.0052 per minute. Smart formatting is included.
- Pre-recorded uploads are limited to 2 GB. Nova requests whose processing exceeds 10 minutes can return HTTP 504. A long source can fail on this synchronous call. A callback upload is not implemented.
- `keywords` is not supported on Nova-3 (`keyterm` replaces it). This client does not send either. `diarize` is deprecated. This client does not diarize.

Useful flags: `--top 5` (default), `--db`, `--out`, and `--work`.

## How a clip is chosen

Picks are deterministic in this phase. Consecutive sentences become windows of 8 to 60 seconds, scored by length (centered on 24 seconds), average word confidence, and a bonus when the first sentence is a question. Overlapping windows keep the higher score. Each kept window is snapped to sentence edges and checked for range, order, overlap, and the 60 second cap. The cap matches the plan: a YouTube Short over one minute with an active copyright claim is blocked worldwide, even though unclaimed Shorts can run up to three minutes.

An LLM picker can replace the scorer later. Its spans still have to pass `snap_span` and `validate_picks`.

## What phase 1 plugs into

Phase 1 is CLI approval and a YouTube publisher. It should use the types already in `clip.domain` and the rows already in SQLite, schema version 1:

- `approve_clip` records YouTube, Instagram, or TikTok on a rendered clip. Approval is per platform.
- `publications_for` builds publication rows only from an `Approved` clip. New rows start `queued` with a null handle.
- `with_handle` stores the platform handle and leaves `queued` in one value. For YouTube that handle is the resumable session URI, written before the upload. For Instagram it is the container id. For TikTok it is the Upload-Post request id or the id you store for a manual post. TikTok `route` is `upload_post` or `manual`. There is no direct TikTok publisher.
- `save_publication` writes that row. Writing the same value again does not change `updated_at`. A database trigger rejects a live state with no handle, and rejects moving a `published` row backward.
- `reconcile` is pure. A published or failed row is returned unchanged. YouTube rows with a session URI stay on that URI: an incomplete upload remains `uploading`, a finished upload moves to `verifying`, and a processed video id moves to `published`.
- `accounts` stores `paused_until` and `pause_reason` for a policy error. Nothing in phase 0 pauses an account.

Publishing order when those publishers exist: YouTube Shorts, then Instagram. TikTok goes through Upload-Post or is posted by hand from the rendered file.

## Tests

```bash
python -m pytest
```

The suite uses the fixture transcript and a generated test video. It does not call Deepgram.
