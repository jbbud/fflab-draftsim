from __future__ import annotations

import argparse
import json
from typing import Any

from .projections import sync_projection_payload


def command_gui(args: argparse.Namespace) -> int:
    from .web import main as web_main

    argv = ["--host", args.host, "--port", str(args.port)]
    return web_main(argv)


def command_sync(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "league_id": args.league_id,
        "year": args.year,
        "week_start": args.week_start,
        "week_end": args.week_end,
    }
    if args.espn_s2:
        payload["espn_s2"] = args.espn_s2
    if args.swid:
        payload["swid"] = args.swid
    print(json.dumps(sync_projection_payload(payload), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fflab")
    subparsers = parser.add_subparsers(dest="command")

    gui = subparsers.add_parser("gui", help="start the local draft simulator")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.set_defaults(func=command_gui)

    sync = subparsers.add_parser("sync-projections", help="fetch ESPN projections as JSON")
    sync.add_argument("--league-id", type=int, required=True)
    sync.add_argument("--year", type=int, required=True)
    sync.add_argument("--espn-s2")
    sync.add_argument("--swid")
    sync.add_argument("--week-start", type=int, default=1)
    sync.add_argument("--week-end", type=int, default=17)
    sync.set_defaults(func=command_sync)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args = parser.parse_args(["gui"])
    return int(args.func(args))
