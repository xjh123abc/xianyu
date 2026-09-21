"""Regression tests for the shared Xianyu channel-database location."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.api import conversations
from config.paths import PROJECT_ROOT, resolve_project_path
from config.settings import settings
from scripts import run_xianyu_stage3 as runner
from scripts import xianyu_stage3_control as control


def test_relative_and_absolute_paths_resolve_without_string_concatenation(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "项目 根目录"
    project_root.mkdir()
    relative = Path("logs") / "渠道库.sqlite3"
    absolute = tmp_path / "外部 数据" / "channel.sqlite3"

    assert resolve_project_path(relative, project_root=project_root) == project_root / relative
    assert resolve_project_path(absolute, project_root=project_root) == absolute


def test_three_entries_share_the_default_configured_database(
    monkeypatch,
) -> None:
    configured_path = Path("logs/xianyu_stage3.sqlite3")
    monkeypatch.setattr(settings, "xianyu_channel_database_path", str(configured_path))
    expected = PROJECT_ROOT / configured_path

    assert conversations.resolve_channel_database_path() == expected
    assert runner.resolve_channel_database_path(None, project_root=PROJECT_ROOT) == expected
    assert control.resolve_channel_database_path(None) == expected


def test_configured_database_can_point_to_a_temporary_absolute_database(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configured_path = tmp_path / "临时 渠道库.sqlite3"
    monkeypatch.setattr(settings, "xianyu_channel_database_path", str(configured_path))

    assert conversations.resolve_channel_database_path() == configured_path
    assert runner.resolve_channel_database_path(None, project_root=PROJECT_ROOT) == configured_path
    assert control.resolve_channel_database_path(None) == configured_path


def test_explicit_database_override_wins_for_both_command_line_tools(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "项目 根目录"
    project_root.mkdir()
    override = Path("overrides") / "channel.sqlite3"
    expected = project_root / override

    assert runner.resolve_channel_database_path(override, project_root=project_root) == expected
    assert control.resolve_channel_database_path(override, project_root=project_root) == expected


def test_control_cli_uses_a_configured_temporary_database(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    configured_path = tmp_path / "temporary-channel.sqlite3"
    monkeypatch.setattr(settings, "xianyu_channel_database_path", str(configured_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["xianyu_stage3_control.py", "--account", "seller", "pause"],
    )

    assert control.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["database_path"] == str(configured_path)
    assert result["database_source"] == "XIANYU_CHANNEL_DATABASE_PATH"
    assert result["result"] == "paused"
    assert configured_path.exists()


def test_control_cli_keeps_the_explicit_database_override(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    configured_path = tmp_path / "configured.sqlite3"
    override_path = tmp_path / "explicit.sqlite3"
    monkeypatch.setattr(settings, "xianyu_channel_database_path", str(configured_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xianyu_stage3_control.py",
            "--db",
            str(override_path),
            "--account",
            "seller",
            "pause",
        ],
    )

    assert control.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["database_path"] == str(override_path)
    assert result["database_source"] == "--db"
    assert override_path.exists()
    assert not configured_path.exists()
