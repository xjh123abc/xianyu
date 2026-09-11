"""SQLite ledger and control state for one Xianyu account."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ChannelStore:
    """Persist delivery state before any network send is attempted."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS channel_state (
                    account_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    login_state TEXT NOT NULL DEFAULT 'UNKNOWN',
                    control_version INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS channel_sessions (
                    account_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    buyer_id TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'AUTO' CHECK (mode IN ('AUTO', 'HUMAN')),
                    takeover_reason TEXT,
                    control_version INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_id, chat_id)
                );
                CREATE TABLE IF NOT EXISTS channel_messages (
                    account_id TEXT NOT NULL,
                    platform_message_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    buyer_id TEXT NOT NULL,
                    platform_item_id TEXT,
                    message_type TEXT NOT NULL,
                    sender_is_seller INTEGER NOT NULL DEFAULT 0,
                    is_system_event INTEGER NOT NULL DEFAULT 0,
                    text TEXT NOT NULL DEFAULT '',
                    received_at TEXT,
                    status TEXT NOT NULL DEFAULT 'RECEIVED',
                    delivery_state TEXT NOT NULL DEFAULT 'NONE',
                    candidate_text TEXT,
                    request_id TEXT,
                    error TEXT,
                    PRIMARY KEY (account_id, platform_message_id),
                    FOREIGN KEY (account_id, chat_id)
                        REFERENCES channel_sessions(account_id, chat_id)
                );
                CREATE INDEX IF NOT EXISTS idx_channel_messages_chat
                    ON channel_messages(account_id, chat_id, received_at);
                """
            )

    def ensure_account(self, account_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO channel_state(account_id) VALUES (?)",
                (account_id,),
            )

    def set_enabled(self, account_id: str, enabled: bool) -> int:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO channel_state(account_id, enabled, control_version)
                VALUES (?, ?, 1)
                ON CONFLICT(account_id) DO UPDATE SET
                    enabled=excluded.enabled,
                    control_version=channel_state.control_version + 1,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (account_id, int(enabled)),
            )
            row = connection.execute(
                "SELECT control_version FROM channel_state WHERE account_id = ?",
                (account_id,),
            ).fetchone()
            return int(row[0])

    def account_state(self, account_id: str) -> dict[str, Any]:
        self.ensure_account(account_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM channel_state WHERE account_id = ?", (account_id,)
            ).fetchone()
        return dict(row)

    def set_login_state(self, account_id: str, state: str) -> None:
        self.ensure_account(account_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE channel_state SET login_state = ?, updated_at = CURRENT_TIMESTAMP WHERE account_id = ?",
                (state, account_id),
            )

    def _ensure_session(self, connection: sqlite3.Connection, account_id: str, chat_id: str, buyer_id: str) -> None:
        connection.execute(
            """
            INSERT INTO channel_sessions(account_id, chat_id, buyer_id)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id, chat_id) DO UPDATE SET buyer_id=excluded.buyer_id
            """,
            (account_id, chat_id, buyer_id),
        )

    def set_session_mode(
        self,
        account_id: str,
        chat_id: str,
        buyer_id: str,
        mode: str,
        reason: str | None = None,
    ) -> int:
        if mode not in {"AUTO", "HUMAN"}:
            raise ValueError("mode must be AUTO or HUMAN")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_session(connection, account_id, chat_id, buyer_id)
            connection.execute(
                """
                UPDATE channel_sessions
                SET mode = ?, takeover_reason = ?, control_version = control_version + 1
                WHERE account_id = ? AND chat_id = ?
                """,
                (mode, reason, account_id, chat_id),
            )
            row = connection.execute(
                "SELECT control_version FROM channel_sessions WHERE account_id = ? AND chat_id = ?",
                (account_id, chat_id),
            ).fetchone()
            return int(row[0])

    def session_state(self, account_id: str, chat_id: str, buyer_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            self._ensure_session(connection, account_id, chat_id, buyer_id)
            row = connection.execute(
                "SELECT * FROM channel_sessions WHERE account_id = ? AND chat_id = ?",
                (account_id, chat_id),
            ).fetchone()
        return dict(row)

    def record_inbound(self, message: Any) -> bool:
        """Insert one event; duplicate platform IDs are a no-op."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO channel_state(account_id) VALUES (?)",
                (message.account_id,),
            )
            self._ensure_session(connection, message.account_id, message.chat_id, message.buyer_id)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO channel_messages(
                    account_id, platform_message_id, chat_id, buyer_id, platform_item_id,
                    message_type, sender_is_seller, is_system_event, text, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.account_id,
                    message.platform_message_id,
                    message.chat_id,
                    message.buyer_id,
                    message.platform_item_id,
                    message.message_type,
                    int(message.sender_is_seller),
                    int(message.is_system_event),
                    message.text,
                    message.received_at,
                ),
            )
            return cursor.rowcount == 1

    def mark_ignored(self, account_id: str, message_id: str, reason: str) -> None:
        self._update_message(account_id, message_id, status="IGNORED", error=reason)

    def claim_for_send(self, account_id: str, message_id: str, candidate_text: str) -> dict[str, Any] | None:
        if not candidate_text.strip():
            raise ValueError("candidate_text must not be blank")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.*, s.enabled, s.control_version AS account_version,
                       cs.mode, cs.control_version AS session_version
                FROM channel_messages m
                JOIN channel_state s ON s.account_id = m.account_id
                JOIN channel_sessions cs ON cs.account_id = m.account_id AND cs.chat_id = m.chat_id
                WHERE m.account_id = ? AND m.platform_message_id = ?
                """,
                (account_id, message_id),
            ).fetchone()
            if row is None or not row["enabled"] or row["mode"] != "AUTO":
                return None
            if row["status"] != "RECEIVED" or row["delivery_state"] != "NONE":
                return None
            request_id = str(uuid.uuid4())
            connection.execute(
                """
                UPDATE channel_messages
                SET status = 'SENDING', delivery_state = 'NONE', candidate_text = ?, request_id = ?
                WHERE account_id = ? AND platform_message_id = ? AND status = 'RECEIVED'
                """,
                (candidate_text, request_id, account_id, message_id),
            )
            return {
                "request_id": request_id,
                "chat_id": row["chat_id"],
                "buyer_id": row["buyer_id"],
                "account_version": row["account_version"],
                "session_version": row["session_version"],
            }

    def mark_delivery(self, account_id: str, message_id: str, state: str, *, error: str | None = None) -> None:
        if state not in {"LOCAL_SUBMITTED", "CONFIRMED", "FAILED", "UNKNOWN"}:
            raise ValueError("invalid delivery state")
        status = "SENDING" if state in {"LOCAL_SUBMITTED", "UNKNOWN"} else state
        self._update_message(
            account_id,
            message_id,
            status=status,
            delivery_state=state,
            error=error,
        )

    def _update_message(self, account_id: str, message_id: str, **fields: object) -> None:
        allowed = {"status", "delivery_state", "error"}
        if not fields or not set(fields).issubset(allowed):
            raise ValueError("unsupported message update")
        assignments = ", ".join(f"{field} = ?" for field in fields)
        values = [fields[field] for field in fields]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE channel_messages SET {assignments} WHERE account_id = ? AND platform_message_id = ?",
                (*values, account_id, message_id),
            )

    def message(self, account_id: str, message_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM channel_messages WHERE account_id = ? AND platform_message_id = ?",
                (account_id, message_id),
            ).fetchone()
        return dict(row) if row else None
