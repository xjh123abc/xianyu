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
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (account_id, chat_id)
                );
                CREATE TABLE IF NOT EXISTS channel_messages (
                    account_id TEXT NOT NULL,
                    platform_message_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    platform_chat_id TEXT,
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
                CREATE TABLE IF NOT EXISTS xianyu_item_bindings (
                    account_id TEXT NOT NULL,
                    platform_item_id TEXT NOT NULL,
                    internal_item_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (account_id, platform_item_id)
                );
                """
            )
            # The stage-2 database may already exist.  Keep this migration local
            # and additive so its accepted delivery ledger remains intact.
            self._add_column_if_missing(connection, "channel_sessions", "current_item_id TEXT")
            self._add_column_if_missing(connection, "channel_sessions", "updated_at TEXT")
            self._add_column_if_missing(connection, "channel_messages", "action TEXT")
            self._add_column_if_missing(connection, "channel_messages", "platform_chat_id TEXT")
            self._add_column_if_missing(connection, "channel_messages", "handoff_reason TEXT")
            self._add_column_if_missing(connection, "channel_messages", "notification_state TEXT NOT NULL DEFAULT 'NONE'")
            self._add_column_if_missing(connection, "channel_messages", "notification_error TEXT")

    @staticmethod
    def _add_column_if_missing(connection: sqlite3.Connection, table: str, definition: str) -> None:
        column = definition.split()[0]
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

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
            INSERT INTO channel_sessions(account_id, chat_id, buyer_id, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(account_id, chat_id) DO UPDATE SET
                buyer_id=excluded.buyer_id,
                updated_at=CURRENT_TIMESTAMP
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
                SET mode = ?, takeover_reason = ?, control_version = control_version + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ? AND chat_id = ?
                """,
                (mode, reason, account_id, chat_id),
            )
            row = connection.execute(
                "SELECT control_version FROM channel_sessions WHERE account_id = ? AND chat_id = ?",
                (account_id, chat_id),
            ).fetchone()
            return int(row[0])

    def resume_session_auto(self, account_id: str, chat_id: str) -> dict[str, Any] | None:
        """Return one existing conversation to automatic replies.

        This deliberately updates only the durable conversation-control fields.
        It does not create a missing session or alter its buyer/item context.
        """

        normalized_account_id = account_id.strip()
        normalized_chat_id = chat_id.strip()
        if not normalized_account_id or not normalized_chat_id:
            raise ValueError("account_id and chat_id must not be empty")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT account_id, chat_id FROM channel_sessions
                WHERE account_id = ? AND chat_id = ?
                """,
                (normalized_account_id, normalized_chat_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE channel_sessions
                SET mode = 'AUTO', takeover_reason = NULL,
                    control_version = control_version + 1, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ? AND chat_id = ?
                """,
                (normalized_account_id, normalized_chat_id),
            )
            state = connection.execute(
                """
                SELECT account_id, chat_id, buyer_id, mode, takeover_reason, control_version
                FROM channel_sessions
                WHERE account_id = ? AND chat_id = ?
                """,
                (normalized_account_id, normalized_chat_id),
            ).fetchone()
        return dict(state) if state is not None else None

    def session_state(self, account_id: str, chat_id: str, buyer_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            self._ensure_session(connection, account_id, chat_id, buyer_id)
            row = connection.execute(
                "SELECT * FROM channel_sessions WHERE account_id = ? AND chat_id = ?",
                (account_id, chat_id),
            ).fetchone()
        return dict(row)

    def bind_item(self, account_id: str, platform_item_id: str, internal_item_id: str) -> None:
        """Bind a trusted Xianyu listing ID to one local product ID."""

        if not account_id.strip() or not platform_item_id.strip() or not internal_item_id.strip():
            raise ValueError("account, platform item and internal item are required")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO xianyu_item_bindings(account_id, platform_item_id, internal_item_id, enabled)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(account_id, platform_item_id) DO UPDATE SET
                    internal_item_id=excluded.internal_item_id, enabled=1, updated_at=CURRENT_TIMESTAMP
                """,
                (account_id, platform_item_id, internal_item_id),
            )

    def is_item_bound(self, account_id: str, platform_item_id: str) -> bool:
        """Return whether this account actively owns the platform listing."""

        return self.get_bound_item_id(account_id, platform_item_id) is not None

    def get_bound_item_id(self, account_id: str, platform_item_id: str) -> str | None:
        """Read the local item mapped to an active platform listing, if any."""

        if not platform_item_id.strip():
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT internal_item_id FROM xianyu_item_bindings
                WHERE account_id = ? AND platform_item_id = ? AND enabled = 1
                """,
                (account_id, platform_item_id),
            ).fetchone()
        return str(row["internal_item_id"]) if row else None

    def get_current_bound_item_id(self, account_id: str, chat_id: str) -> str | None:
        """Read a chat's remembered item only while it remains actively bound.

        The context is established from a previously verified platform listing.
        Rechecking the active binding prevents a stale or disabled listing from
        reopening AI handling for a later item-less message.
        """

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session.current_item_id
                FROM channel_sessions AS session
                WHERE session.account_id = ?
                  AND session.chat_id = ?
                  AND session.current_item_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1
                      FROM xianyu_item_bindings AS binding
                      WHERE binding.account_id = session.account_id
                        AND binding.internal_item_id = session.current_item_id
                        AND binding.enabled = 1
                  )
                """,
                (account_id, chat_id),
            ).fetchone()
        return str(row["current_item_id"]) if row else None

    def resolve_item(self, account_id: str, chat_id: str, buyer_id: str, platform_item_id: str | None) -> str | None:
        """Resolve a trusted listing ID, otherwise retain the session's prior item.

        A non-empty unbound platform item deliberately resolves to ``None``: it
        must never silently reuse another listing's facts.
        """

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_session(connection, account_id, chat_id, buyer_id)
            if platform_item_id:
                row = connection.execute(
                    """
                    SELECT internal_item_id FROM xianyu_item_bindings
                    WHERE account_id = ? AND platform_item_id = ? AND enabled = 1
                    """,
                    (account_id, platform_item_id),
                ).fetchone()
                if row is None:
                    return None
                item_id = str(row["internal_item_id"])
                connection.execute(
                    """
                    UPDATE channel_sessions SET current_item_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE account_id = ? AND chat_id = ?
                    """,
                    (item_id, account_id, chat_id),
                )
                return item_id
            row = connection.execute(
                """
                SELECT session.current_item_id
                FROM channel_sessions AS session
                WHERE session.account_id = ?
                  AND session.chat_id = ?
                  AND session.current_item_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1
                      FROM xianyu_item_bindings AS binding
                      WHERE binding.account_id = session.account_id
                        AND binding.internal_item_id = session.current_item_id
                        AND binding.enabled = 1
                  )
                """,
                (account_id, chat_id),
            ).fetchone()
            return str(row["current_item_id"]) if row and row["current_item_id"] else None

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
                    account_id, platform_message_id, chat_id, platform_chat_id, buyer_id, platform_item_id,
                    message_type, sender_is_seller, is_system_event, text, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.account_id,
                    message.platform_message_id,
                    message.chat_id,
                    message.platform_chat_id,
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

    def claim_for_generation(self, account_id: str, message_id: str) -> dict[str, Any] | None:
        """Atomically reserve an inbound message before calling /chat."""

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
                UPDATE channel_messages SET status = 'GENERATING', request_id = ?
                WHERE account_id = ? AND platform_message_id = ? AND status = 'RECEIVED'
                """,
                (request_id, account_id, message_id),
            )
            return {
                "request_id": request_id,
                "chat_id": row["chat_id"],
                "platform_chat_id": row["platform_chat_id"] or row["chat_id"],
                "buyer_id": row["buyer_id"],
                "account_version": int(row["account_version"]),
                "session_version": int(row["session_version"]),
            }

    def prepare_candidate(
        self,
        account_id: str,
        message_id: str,
        *,
        action: str,
        candidate_text: str,
        account_version: int,
        session_version: int,
    ) -> dict[str, Any] | None:
        """Persist a candidate after /chat, only while controls are unchanged."""

        if action not in {"answer", "clarify"} or not candidate_text.strip():
            raise ValueError("a non-empty answer or clarification is required")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.request_id, s.enabled, s.control_version AS account_version,
                       cs.mode, cs.control_version AS session_version
                FROM channel_messages m
                JOIN channel_state s ON s.account_id = m.account_id
                JOIN channel_sessions cs ON cs.account_id = m.account_id AND cs.chat_id = m.chat_id
                WHERE m.account_id = ? AND m.platform_message_id = ?
                """,
                (account_id, message_id),
            ).fetchone()
            if (
                row is None
                or not row["enabled"]
                or row["mode"] != "AUTO"
                or int(row["account_version"]) != account_version
                or int(row["session_version"]) != session_version
            ):
                connection.execute(
                    """UPDATE channel_messages SET status = 'SUPERSEDED', error = 'control_changed_during_generation'
                       WHERE account_id = ? AND platform_message_id = ? AND status = 'GENERATING'""",
                    (account_id, message_id),
                )
                return None
            connection.execute(
                """
                UPDATE channel_messages SET status = 'READY', action = ?, candidate_text = ?
                WHERE account_id = ? AND platform_message_id = ? AND status = 'GENERATING'
                """,
                (action, candidate_text.strip(), account_id, message_id),
            )
            return {"request_id": row["request_id"]}

    def claim_ready_delivery(self, account_id: str, message_id: str) -> dict[str, Any] | None:
        """Make a durable ready candidate the sole permitted send attempt."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.*, s.enabled, cs.mode FROM channel_messages m
                JOIN channel_state s ON s.account_id = m.account_id
                JOIN channel_sessions cs ON cs.account_id = m.account_id AND cs.chat_id = m.chat_id
                WHERE m.account_id = ? AND m.platform_message_id = ?
                """,
                (account_id, message_id),
            ).fetchone()
            if row is None or not row["enabled"] or row["mode"] != "AUTO" or row["status"] != "READY":
                return None
            connection.execute(
                """UPDATE channel_messages SET status = 'SENDING'
                   WHERE account_id = ? AND platform_message_id = ? AND status = 'READY'""",
                (account_id, message_id),
            )
            return {
                key: row[key]
                for key in ("request_id", "chat_id", "platform_chat_id", "buyer_id", "candidate_text", "action")
            } | {"platform_chat_id": row["platform_chat_id"] or row["chat_id"]}

    def handoff(
        self,
        account_id: str,
        message_id: str,
        *,
        reason: str,
        candidate_text: str,
    ) -> dict[str, Any] | None:
        """Switch the session to HUMAN before any seller notification or send."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.*, s.enabled, cs.mode FROM channel_messages m
                JOIN channel_state s ON s.account_id = m.account_id
                JOIN channel_sessions cs ON cs.account_id = m.account_id AND cs.chat_id = m.chat_id
                WHERE m.account_id = ? AND m.platform_message_id = ?
                """,
                (account_id, message_id),
            ).fetchone()
            if row is None or not row["enabled"] or row["mode"] != "AUTO" or row["status"] not in {"RECEIVED", "GENERATING"}:
                return None
            request_id = row["request_id"] or str(uuid.uuid4())
            connection.execute(
                """
                UPDATE channel_sessions SET mode = 'HUMAN', takeover_reason = ?,
                    control_version = control_version + 1
                WHERE account_id = ? AND chat_id = ?
                """,
                (reason, account_id, row["chat_id"]),
            )
            connection.execute(
                """
                UPDATE channel_messages SET status = 'HUMAN_REQUIRED', action = 'human_handoff',
                    candidate_text = ?, request_id = ?, handoff_reason = ?, notification_state = 'PENDING'
                WHERE account_id = ? AND platform_message_id = ?
                """,
                (candidate_text.strip(), request_id, reason, account_id, message_id),
            )
            return {
                "request_id": request_id,
                "chat_id": row["chat_id"],
                "platform_chat_id": row["platform_chat_id"] or row["chat_id"],
                "buyer_id": row["buyer_id"],
                "text": row["text"],
            }

    def claim_handoff_notice(self, account_id: str, message_id: str) -> dict[str, Any] | None:
        """Reserve the optional fixed handoff notice only while account is live."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.*, s.enabled FROM channel_messages m
                JOIN channel_state s ON s.account_id = m.account_id
                WHERE m.account_id = ? AND m.platform_message_id = ?
                """,
                (account_id, message_id),
            ).fetchone()
            if row is None or not row["enabled"] or row["status"] != "HUMAN_REQUIRED" or row["delivery_state"] != "NONE":
                return None
            connection.execute(
                """UPDATE channel_messages SET status = 'SENDING'
                   WHERE account_id = ? AND platform_message_id = ? AND status = 'HUMAN_REQUIRED'""",
                (account_id, message_id),
            )
            return {
                key: row[key]
                for key in ("request_id", "chat_id", "platform_chat_id", "buyer_id", "candidate_text")
            } | {"platform_chat_id": row["platform_chat_id"] or row["chat_id"]}

    def mark_notification(self, account_id: str, message_id: str, state: str, error: str | None = None) -> None:
        if state not in {"SENT", "FAILED", "DISABLED"}:
            raise ValueError("invalid notification state")
        with self._connect() as connection:
            connection.execute(
                """UPDATE channel_messages SET notification_state = ?, notification_error = ?
                   WHERE account_id = ? AND platform_message_id = ?""",
                (state, error, account_id, message_id),
            )

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
