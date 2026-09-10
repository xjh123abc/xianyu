"""Bounded short-term session state with an optional shared SQLite backend."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


_EMPTY_STATE = {
    "order_id": None,
    "current_item_id": None,
    "last_intent": None,
}


class SessionManager:
    """Store session state with TTL, capacity limits, and per-session ordering.

    Passing ``database_path`` enables a process-shared SQLite store suitable for
    multiple application workers. Omitting it keeps an isolated in-memory store
    for unit tests and explicitly ephemeral uses.
    """

    def __init__(
        self,
        max_history: int = 10,
        *,
        database_path: str | Path | None = None,
        ttl_seconds: int = 3600,
        max_sessions: int = 10000,
        lock_timeout_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be greater than zero")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be greater than zero")

        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.lock_timeout_seconds = lock_timeout_seconds
        self._clock = clock
        self.database_path = Path(database_path).resolve() if database_path else None
        self.sessions: dict[str, dict[str, Any]] = {}
        self._updated_at: dict[str, float] = {}
        self._memory_guard = threading.RLock()
        self._session_locks: dict[str, threading.Lock] = {}
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_database()

    def get_or_create(self, chat_id: str | None = None) -> tuple[str, dict[str, Any]]:
        """Return an isolated session, creating it when necessary."""
        normalized_id = chat_id.strip() if isinstance(chat_id, str) else ""
        if not normalized_id:
            normalized_id = str(uuid4())
        if self.database_path is not None:
            return normalized_id, self._database_get_or_create(normalized_id)

        now = self._clock()
        with self._memory_guard:
            self._prune_memory(now)
            session = self.sessions.setdefault(normalized_id, self._empty_session())
            self._updated_at[normalized_id] = now
            self._evict_memory_over_capacity(exclude=normalized_id)
            return normalized_id, self._copy_session(session)

    def set_current_item_id(self, chat_id: str, item_id: str) -> None:
        """Set the confirmed current item for exactly one chat session."""
        normalized_item_id = item_id.strip().upper()
        if not normalized_item_id:
            raise ValueError("item_id must not be empty")
        _, session = self.get_or_create(chat_id)
        session["state"]["current_item_id"] = normalized_item_id
        self._save(chat_id, session)

    def get_current_item_id(self, chat_id: str) -> str | None:
        """Return the current item stored for one chat without crossing sessions."""
        _, session = self.get_or_create(chat_id)
        current_item_id = session["state"].get("current_item_id")
        return str(current_item_id) if current_item_id else None

    def append_turn(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
        *,
        order_id: str | None = None,
        last_intent: str | None = None,
    ) -> None:
        """Save one user/assistant turn and update known business state."""
        _, session = self.get_or_create(chat_id)
        history = session["history"]
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        del history[:-self.max_history]
        state = session["state"]
        if order_id:
            state["order_id"] = order_id
        if last_intent:
            state["last_intent"] = last_intent
        self._save(chat_id, session)

    @asynccontextmanager
    async def session_lock(self, chat_id: str) -> AsyncIterator[None]:
        """Serialize a full request for one chat, including across SQLite workers."""
        normalized_id = chat_id.strip()
        if not normalized_id:
            raise ValueError("chat_id must not be empty")

        if self.database_path is None:
            with self._memory_guard:
                lock = self._session_locks.setdefault(normalized_id, threading.Lock())
            acquired = await asyncio.to_thread(
                lock.acquire,
                True,
                self.lock_timeout_seconds,
            )
            if not acquired:
                raise TimeoutError(f"Timed out waiting for session {normalized_id}")
            try:
                yield
            finally:
                lock.release()
            return

        owner = str(uuid4())
        deadline = time.monotonic() + self.lock_timeout_seconds
        while not self._try_acquire_database_lock(normalized_id, owner):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for session {normalized_id}")
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            self._release_database_lock(normalized_id, owner)

    @staticmethod
    def read_context(session: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Return defensive copies of history and state for one request."""
        history = [dict(item) for item in session.get("history", [])]
        state = dict(session.get("state", {}))
        return history, state

    @staticmethod
    def _empty_session() -> dict[str, Any]:
        return {"history": [], "state": dict(_EMPTY_STATE)}

    @staticmethod
    def _copy_session(session: Mapping[str, Any]) -> dict[str, Any]:
        history, state = SessionManager.read_context(session)
        return {"history": history, "state": state}

    def _save(self, chat_id: str, session: Mapping[str, Any]) -> None:
        normalized_id = chat_id.strip()
        now = self._clock()
        if self.database_path is None:
            with self._memory_guard:
                self.sessions[normalized_id] = self._copy_session(session)
                self._updated_at[normalized_id] = now
                self._prune_memory(now)
                self._evict_memory_over_capacity(exclude=normalized_id)
            return

        with self._connect() as connection:
            connection.execute(
                """INSERT INTO chat_sessions(chat_id, history_json, state_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(chat_id) DO UPDATE SET
                       history_json=excluded.history_json,
                       state_json=excluded.state_json,
                       updated_at=excluded.updated_at""",
                (
                    normalized_id,
                    json.dumps(session["history"], ensure_ascii=False),
                    json.dumps(session["state"], ensure_ascii=False),
                    now,
                ),
            )

    def _prune_memory(self, now: float) -> None:
        expired = [
            chat_id
            for chat_id, updated_at in self._updated_at.items()
            if updated_at <= now - self.ttl_seconds
            and not (
                chat_id in self._session_locks
                and self._session_locks[chat_id].locked()
            )
        ]
        for chat_id in expired:
            self.sessions.pop(chat_id, None)
            self._updated_at.pop(chat_id, None)
            self._session_locks.pop(chat_id, None)

    def _evict_memory_over_capacity(self, *, exclude: str) -> None:
        while len(self.sessions) > self.max_sessions:
            candidates = (
                (updated_at, chat_id)
                for chat_id, updated_at in self._updated_at.items()
                if chat_id != exclude
                and not (
                    chat_id in self._session_locks
                    and self._session_locks[chat_id].locked()
                )
            )
            try:
                _, oldest_id = min(candidates)
            except ValueError:
                break
            self.sessions.pop(oldest_id, None)
            self._updated_at.pop(oldest_id, None)
            self._session_locks.pop(oldest_id, None)

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chat_sessions (
                       chat_id TEXT PRIMARY KEY,
                       history_json TEXT NOT NULL,
                       state_json TEXT NOT NULL,
                       updated_at REAL NOT NULL
                   )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chat_session_locks (
                       chat_id TEXT PRIMARY KEY,
                       owner TEXT NOT NULL,
                       expires_at REAL NOT NULL
                   )"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        assert self.database_path is not None
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.lock_timeout_seconds,
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _database_get_or_create(self, chat_id: str) -> dict[str, Any]:
        now = self._clock()
        empty = self._empty_session()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM chat_sessions
                   WHERE updated_at <= ?
                     AND NOT EXISTS (
                         SELECT 1 FROM chat_session_locks
                         WHERE chat_session_locks.chat_id = chat_sessions.chat_id
                     )""",
                (now - self.ttl_seconds,),
            )
            row = connection.execute(
                "SELECT history_json, state_json FROM chat_sessions WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO chat_sessions VALUES (?, ?, ?, ?)",
                    (
                        chat_id,
                        json.dumps(empty["history"]),
                        json.dumps(empty["state"]),
                        now,
                    ),
                )
                session = empty
            else:
                connection.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE chat_id = ?",
                    (now, chat_id),
                )
                session = {"history": json.loads(row[0]), "state": json.loads(row[1])}
            excess = connection.execute(
                "SELECT MAX(COUNT(*) - ?, 0) FROM chat_sessions",
                (self.max_sessions,),
            ).fetchone()[0]
            if excess:
                connection.execute(
                    """DELETE FROM chat_sessions WHERE chat_id IN (
                           SELECT chat_id FROM chat_sessions
                           WHERE chat_id != ?
                             AND NOT EXISTS (
                                 SELECT 1 FROM chat_session_locks
                                 WHERE chat_session_locks.chat_id = chat_sessions.chat_id
                             )
                           ORDER BY updated_at ASC LIMIT ?
                       )""",
                    (chat_id, excess),
                )
        return self._copy_session(session)

    def _try_acquire_database_lock(self, chat_id: str, owner: str) -> bool:
        now = self._clock()
        expires_at = now + max(
            float(self.ttl_seconds),
            self.lock_timeout_seconds * 2,
            300.0,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM chat_session_locks WHERE expires_at <= ?", (now,))
            cursor = connection.execute(
                "INSERT OR IGNORE INTO chat_session_locks VALUES (?, ?, ?)",
                (chat_id, owner, expires_at),
            )
            return cursor.rowcount == 1

    def _release_database_lock(self, chat_id: str, owner: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chat_session_locks WHERE chat_id = ? AND owner = ?",
                (chat_id, owner),
            )
