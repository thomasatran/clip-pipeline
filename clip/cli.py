"""Command line entry point. Phase 0 implements `clip run`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clip.errors import ClipError
from clip.pipeline import RunRequest, execute, format_result


def main() -> None:
    sys.exit(run_cli())


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clip", description="Render ranked vertical clips.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="transcribe, rank, and write captioned 9:16 clips")
    run.add_argument("file", type=Path, help="source video")
    run.add_argument(
        "--rights",
        required=True,
        choices=("own", "licensed"),
        help="only your own footage or footage you have a license for can be ingested",
    )
    run.add_argument("--db", type=Path, default=Path("data/clip.sqlite"))
    run.add_argument("--out", type=Path, default=Path("out"))
    run.add_argument("--work", type=Path, default=Path("data/work"))
    run.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Deepgram-shaped JSON to use instead of calling the API",
    )
    run.add_argument("--top", type=int, default=5, help="maximum clips to keep")

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error(f"unknown command: {args.command}")
        return 2
    try:
        result = execute(
            RunRequest(
                path=args.file,
                rights=args.rights,
                db_path=args.db,
                out_dir=args.out,
                work_dir=args.work,
                transcript_path=args.transcript,
                top_n=args.top,
            )
        )
    except ClipError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_result(result))
    return 0
