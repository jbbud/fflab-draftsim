from __future__ import annotations

from fflab.cli import build_parser


def test_cli_exposes_gui_and_projection_sync_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["gui", "--port", "9999"]).port == 9999
    sync = parser.parse_args(["sync-projections", "--league-id", "1", "--year", "2026"])
    assert sync.league_id == 1
    assert sync.year == 2026
