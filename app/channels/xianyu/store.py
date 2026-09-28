"""SQLite ledger and control state for one Xianyu account."""

from __future__ import annotations

import sqlite3
import threading
import uuid
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
                    candidate_account_version INTEGER,
                    candidate_session_version INTEGER,
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
            self._add_column_if_missing(connection, "channel_messages", "turn_id TEXT")
            self._add_column_if_missing(connection, "channel_messages", "proposal_id TEXT")
            self._add_column_if_missing(connection, "channel_messages", "delivery_report_state TEXT NOT NULL DEFAULT 'NONE'")
            self._add_column_if_missing(connection, "channel_messages", "delivery_report_error TEXT")
            self._add_column_if_missing(connection, "channel_messages", "candidate_account_version INTEGER")
            self._add_column_if_missing(connection, "channel_messages", "candidate_session_version INTEGER")

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

    def recover_interrupted(self, account_id: str) -> dict[str, int]:
        """Classify work left by process exit without repeating an uncertain send."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            interrupted = connection.execute(
                """UPDATE channel_messages
                   SET status = 'INTERRUPTED', error = 'process_interrupted_during_generation'
                   WHERE account_id = ? AND status = 'GENERATING'""",
                (account_id,),
            ).rowcount
            unknown = connection.execute(
                """UPDATE channel_messages
                   SET delivery_state = 'UNKNOWN', error = 'process_interrupted_during_send',
                       delivery_report_state = CASE
                           WHEN turn_id IS NOT NULL AND proposal_id IS NOT NULL THEN 'PENDING'
                           ELSE delivery_report_state
                       END
                   WHERE account_id = ? AND status = 'SENDING' AND delivery_state = 'NONE'""",
                (account_id,),
            ).rowcount
        return {"generation_interrupted": interrupted, "send_unknown": unknown}

    def pending_inbound(self, account_id: str) -> list[dict[str, Any]]:
        """Return received events that were durably queued before an interruption."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT account_id, platform_message_id, chat_id, platform_chat_id, buyer_id,
                          text, message_type, platform_item_id, sender_is_seller,
                          is_system_event, received_at, status
                   FROM channel_messages
                   WHERE account_id = ? AND status IN ('RECEIVED', 'CONTEXT_PENDING')
                   ORDER BY received_at, rowid""",
                (account_id,),
            ).fetchall()
        return [dict(row) for row in rows]

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

    def get_session_state(self, account_id: str, chat_id: str) -> dict[str, Any] | None:
        """Read an existing session without changing its trusted buyer binding."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM channel_sessions WHERE account_id = ? AND chat_id = ?",
                (account_id, chat_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def is_delivery_echo(self, account_id: str, chat_id: str, text: str) -> bool:
        """Identify seller-side echoes that match a reply submitted by this worker."""
        normalized = " ".join(text.split())
        if not normalized:
            return False
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM channel_messages
                   WHERE account_id = ? AND chat_id = ? AND delivery_state IN ('LOCAL_SUBMITTED', 'CONFIRMED')
                     AND candidate_text IS NOT NULL
                   ORDER BY received_at DESC LIMIT 100""",
                (account_id, chat_id),
            ).fetchall()
        return any(" ".join(str(item[0]).split()) == normalized for item in row)

    def confirm_delivery_echo(self, account_id: str, chat_id: str, text: str) -> str | None:
        """Confirm the newest local submission whose exact text echoed from Xianyu."""
        normalized = " ".join(text.split())
        if not normalized:
            return None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT platform_message_id, candidate_text
                   FROM channel_messages
                   WHERE account_id = ? AND chat_id = ?
                     AND delivery_state = 'LOCAL_SUBMITTED'
                     AND candidate_text IS NOT NULL
                   ORDER BY CAST(received_at AS INTEGER) DESC
                   LIMIT 100""",
                (account_id, chat_id),
            ).fetchall()
            match = next(
                (
                    str(row[0])
                    for row in rows
                    if " ".join(str(row[1]).split()) == normalized
                ),
                None,
            )
            if match is None:
                return None
            connection.execute(
                """UPDATE channel_messages
                   SET status = 'CONFIRMED', delivery_state = 'CONFIRMED', error = NULL,
                       delivery_report_state = CASE
                           WHEN turn_id IS NOT NULL AND proposal_id IS NOT NULL THEN 'PENDING'
                           ELSE delivery_report_state
                       END,
                       delivery_report_error = NULL
                   WHERE account_id = ? AND platform_message_id = ?
                     AND delivery_state = 'LOCAL_SUBMITTED'""",
                (account_id, match),
            )
            return match

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

    def list_bound_items(self, account_id: str) -> list[dict[str, str]]:
        """Return this seller's explicitly enabled platform-to-local bindings."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT platform_item_id, internal_item_id FROM xianyu_item_bindings
                   WHERE account_id = ? AND enabled = 1 ORDER BY platform_item_id""",
                (account_id,),
            ).fetchall()
        return [
            {"platform_item_id": str(row[0]), "item_id": str(row[1])}
            for row in rows
        ]

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
            session_buyer_id = message.buyer_id
            if message.sender_is_seller:
                existing_session = connection.execute(
                    "SELECT buyer_id FROM channel_sessions WHERE account_id = ? AND chat_id = ?",
                    (message.account_id, message.chat_id),
                ).fetchone()
                if existing_session is not None:
                    session_buyer_id = str(existing_session["buyer_id"])
            self._ensure_session(connection, message.account_id, message.chat_id, session_buyer_id)
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

    def mark_human_context_pending(self, account_id: str, message_id: str) -> None:
        self._update_message(account_id, message_id, status="CONTEXT_PENDING")

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
                UPDATE channel_messages SET status = 'READY', action = ?, candidate_text = ?,
                    candidate_account_version = ?, candidate_session_version = ?
                WHERE account_id = ? AND platform_message_id = ? AND status = 'GENERATING'
                """,
                (
                    action,
                    candidate_text.strip(),
                    account_version,
                    session_version,
                    account_id,
                    message_id,
                ),
            )
            return {"request_id": row["request_id"]}

    def pending_ready_deliveries(self, account_id: str) -> list[dict[str, Any]]:
        """Return unsent candidates; claim_ready_delivery rechecks control versions."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT platform_message_id FROM channel_messages
                   WHERE account_id = ? AND status = 'READY' AND delivery_state = 'NONE'
                   ORDER BY received_at, rowid""",
                (account_id,),
            ).fetchall()
        return [{"platform_message_id": str(row[0])} for row in rows]

    def claim_ready_delivery(self, account_id: str, message_id: str) -> dict[str, Any] | None:
        """Make a durable ready candidate the sole permitted send attempt."""

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
            if row is None or not row["enabled"] or row["mode"] != "AUTO" or row["status"] != "READY":
                return None
            if (
                row["candidate_account_version"] is None
                or row["candidate_session_version"] is None
                or int(row["candidate_account_version"]) != int(row["account_version"])
                or int(row["candidate_session_version"]) != int(row["session_version"])
            ):
                connection.execute(
                    """UPDATE channel_messages SET status = 'SUPERSEDED',
                              error = 'control_changed_before_delivery'
                       WHERE account_id = ? AND platform_message_id = ? AND status = 'READY'""",
                    (account_id, message_id),
                )
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

    def mark_delivery(self, account_id: str, message_id: str, state: str, *, error: str | None = None) -> None:
        if state not in {"LOCAL_SUBMITTED", "CONFIRMED", "FAILED", "UNKNOWN"}:
            raise ValueError("invalid delivery state")
        status = "SENDING" if state in {"LOCAL_SUBMITTED", "UNKNOWN"} else state
        with self._connect() as connection:
            connection.execute(
                """UPDATE channel_messages
                   SET status = ?, delivery_state = ?, error = ?,
                       delivery_report_state = CASE
                           WHEN turn_id IS NOT NULL AND proposal_id IS NOT NULL THEN 'PENDING'
                           ELSE delivery_report_state
                       END,
                       delivery_report_error = NULL
                   WHERE account_id = ? AND platform_message_id = ?""",
                (status, state, error, account_id, message_id),
            )

    def record_delivery_reference(
        self,
        account_id: str,
        message_id: str,
        *,
        turn_id: str,
        proposal_id: str,
    ) -> None:
        """Persist the opaque API proposal identifier before attempting send."""

        if not turn_id.strip() or not proposal_id.strip():
            raise ValueError("turn_id and proposal_id must not be blank")
        with self._connect() as connection:
            connection.execute(
                """UPDATE channel_messages
                   SET turn_id = ?, proposal_id = ?, delivery_report_state = 'PENDING',
                       delivery_report_error = NULL
                   WHERE account_id = ? AND platform_message_id = ?""",
                (turn_id.strip(), proposal_id.strip(), account_id, message_id),
            )

    def pending_delivery_reports(self, account_id: str) -> list[dict[str, Any]]:
        """Return receipts that may be retried without sending buyer text again."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT chat_id, platform_message_id, turn_id, proposal_id, delivery_state
                   FROM channel_messages
                   WHERE account_id = ?
                     AND turn_id IS NOT NULL AND proposal_id IS NOT NULL
                     AND delivery_state IN ('LOCAL_SUBMITTED', 'CONFIRMED', 'FAILED', 'UNKNOWN')
                     AND delivery_report_state IN ('PENDING', 'RETRY')""",
                (account_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_delivery_report(
        self,
        account_id: str,
        message_id: str,
        state: str,
        *,
        error: str | None = None,
    ) -> None:
        if state not in {"SENT", "RETRY", "REJECTED"}:
            raise ValueError("invalid delivery report state")
        with self._connect() as connection:
            connection.execute(
                """UPDATE channel_messages
                   SET delivery_report_state = ?, delivery_report_error = ?
                   WHERE account_id = ? AND platform_message_id = ?""",
                (state, error, account_id, message_id),
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
