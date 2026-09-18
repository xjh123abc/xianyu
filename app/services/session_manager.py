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

from app.services.chat_contracts import (
    TASK_TYPES,
    SessionContext,
    TaskType,
    default_negotiation_state,
)


_EMPTY_STATE = {
    "order_id": None,
    "current_item_id": None,
    "last_intent": None,
    "xianyu_context": {
        "item_id": None,
        "recent_price_topic": None,
        "shipping_condition": None,
    },
    "last_task_type": None,
    "negotiation": default_negotiation_state(),
}
_UNSET = object()
_SHIPPING_CONDITIONS = {"seller_pays", "buyer_pays"}
_NEGOTIATION_KEYS = frozenset({"item_id", "round", "last_ai_offer", "last_buyer_offer"})


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

    def load(self, chat_id: str) -> SessionContext:
        """Read one existing session through the unified S2 contract.

        SQLite, TTL, capacity pruning, and the existing state schema continue
        to be owned by this manager. Callers receive only the fields required
        to plan the next turn.
        """

        _, session = self.get_or_create(chat_id)
        history, state = self.read_context(session)
        return SessionContext(
            history=history,
            current_item_id=state.get("current_item_id"),
            current_order_id=state.get("order_id"),
            last_task_type=state.get("last_task_type"),
            negotiation=self._negotiation_from_state(state),
            platform_context={"xianyu": self._xianyu_context_from_state(state)},
        )

    def set_current_item_id(self, chat_id: str, item_id: str) -> None:
        """Set the confirmed current item for exactly one chat session."""
        normalized_item_id = item_id.strip().upper()
        if not normalized_item_id:
            raise ValueError("item_id must not be empty")
        _, session = self.get_or_create(chat_id)
        state = session["state"]
        previous_item_id = state.get("current_item_id")
        state["current_item_id"] = normalized_item_id
        xianyu_context = self._xianyu_context_from_state(state)
        if previous_item_id != normalized_item_id or xianyu_context["item_id"] != normalized_item_id:
            xianyu_context = self._empty_xianyu_context(normalized_item_id)
        state["xianyu_context"] = xianyu_context
        negotiation = self._negotiation_from_state(state)
        if previous_item_id != normalized_item_id or negotiation["item_id"] != normalized_item_id:
            negotiation = self._empty_negotiation_state(normalized_item_id)
        state["negotiation"] = negotiation
        self._save(chat_id, session)

    def get_current_item_id(self, chat_id: str) -> str | None:
        """Return the current item stored for one chat without crossing sessions."""
        _, session = self.get_or_create(chat_id)
        current_item_id = session["state"].get("current_item_id")
        return str(current_item_id) if current_item_id else None

    def get_xianyu_context(self, chat_id: str) -> dict[str, Any]:
        """Return one chat's normalized product-scoped follow-up context."""

        _, session = self.get_or_create(chat_id)
        return dict(self._xianyu_context_from_state(session["state"]))

    def update_xianyu_context(
        self,
        chat_id: str,
        *,
        item_id: str | None | object = _UNSET,
        recent_price_topic: str | None | object = _UNSET,
        shipping_condition: str | None | object = _UNSET,
    ) -> None:
        """Persist seller-neutral Xianyu follow-up context through a public API.

        A different product clears prior price topic and shipping selection.
        ``shipping_condition`` represents only an explicit buyer selection;
        comparison questions must leave it unset.
        """

        _, session = self.get_or_create(chat_id)
        state = session["state"]
        context = self._xianyu_context_from_state(state)
        if item_id is not _UNSET:
            normalized_item_id = self._normalise_item_id(item_id)
            if normalized_item_id != context["item_id"]:
                context = self._empty_xianyu_context(normalized_item_id)
            else:
                context["item_id"] = normalized_item_id
        if recent_price_topic is not _UNSET:
            if recent_price_topic is not None and (
                not isinstance(recent_price_topic, str) or not recent_price_topic.strip()
            ):
                raise ValueError("recent_price_topic must be a non-empty string or None")
            context["recent_price_topic"] = (
                recent_price_topic.strip() if isinstance(recent_price_topic, str) else None
            )
        if shipping_condition is not _UNSET:
            if shipping_condition is not None and shipping_condition not in _SHIPPING_CONDITIONS:
                raise ValueError("shipping_condition must be seller_pays, buyer_pays, or None")
            context["shipping_condition"] = shipping_condition
        state["xianyu_context"] = context
        self._save(chat_id, session)

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

    def save_turn(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
        *,
        current_item_id: str | None | object = _UNSET,
        current_order_id: str | None = None,
        last_task_type: TaskType | None = None,
        legacy_intent: str | None = None,
        xianyu_context_updates: Mapping[str, object] | None = None,
        negotiation: Mapping[str, object] | None = None,
    ) -> None:
        """Persist a completed turn through one caller-facing session entrypoint.

        The pre-S2 methods remain the implementation and compatibility layer.
        ``legacy_intent`` preserves current router behaviour until S3 replaces
        it with planner task types.
        """

        if last_task_type is not None and last_task_type not in TASK_TYPES:
            raise ValueError("last_task_type must be product, price, service, or order")
        if xianyu_context_updates is not None and not isinstance(
            xianyu_context_updates, Mapping
        ):
            raise ValueError("xianyu_context_updates must be a mapping")
        if negotiation is not None and not isinstance(negotiation, Mapping):
            raise ValueError("negotiation must be a mapping")
        normalized_item_id = _UNSET
        if current_item_id is not _UNSET:
            if current_item_id is None:
                raise ValueError("current_item_id cannot be cleared through save_turn")
            normalized_item_id = self._normalise_item_id(current_item_id)
        updates = self._validated_xianyu_context_updates(xianyu_context_updates)
        if negotiation is not None:
            self._updated_negotiation_state({}, negotiation)

        self.append_turn(
            chat_id,
            user_message,
            assistant_message,
            order_id=current_order_id,
            last_intent=legacy_intent,
        )
        if normalized_item_id is not _UNSET:
            assert isinstance(normalized_item_id, str)
            self.set_current_item_id(chat_id, normalized_item_id)
        if updates:
            if normalized_item_id is not _UNSET:
                updates.setdefault("item_id", normalized_item_id)
            self.update_xianyu_context(chat_id, **updates)
        if last_task_type is not None or negotiation is not None:
            _, session = self.get_or_create(chat_id)
            state = session["state"]
            if last_task_type is not None:
                state["last_task_type"] = last_task_type
            if negotiation is not None:
                state["negotiation"] = self._updated_negotiation_state(state, negotiation)
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
        state = SessionManager._normalise_state(session.get("state", {}))
        return history, state

    @staticmethod
    def _empty_session() -> dict[str, Any]:
        return {"history": [], "state": SessionManager._normalise_state({})}

    @staticmethod
    def _copy_session(session: Mapping[str, Any]) -> dict[str, Any]:
        history, state = SessionManager.read_context(session)
        return {"history": history, "state": state}

    @staticmethod
    def _empty_xianyu_context(item_id: str | None = None) -> dict[str, Any]:
        return {
            "item_id": item_id,
            "recent_price_topic": None,
            "shipping_condition": None,
        }

    @staticmethod
    def _empty_negotiation_state(item_id: str | None = None) -> dict[str, object]:
        state = default_negotiation_state()
        state["item_id"] = item_id
        return state

    @classmethod
    def _xianyu_context_from_state(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        raw_context = state.get("xianyu_context")
        context = cls._empty_xianyu_context()
        if isinstance(raw_context, Mapping):
            raw_item_id = raw_context.get("item_id")
            if isinstance(raw_item_id, str) and raw_item_id.strip():
                context["item_id"] = raw_item_id.strip().upper()
            raw_price_topic = raw_context.get("recent_price_topic")
            if isinstance(raw_price_topic, str) and raw_price_topic.strip():
                context["recent_price_topic"] = raw_price_topic.strip()
            raw_shipping = raw_context.get("shipping_condition")
            if raw_shipping in _SHIPPING_CONDITIONS:
                context["shipping_condition"] = raw_shipping
        return context

    @classmethod
    def _negotiation_from_state(cls, state: Mapping[str, Any]) -> dict[str, object]:
        raw_negotiation = state.get("negotiation")
        negotiation = cls._empty_negotiation_state()
        if not isinstance(raw_negotiation, Mapping):
            return negotiation
        raw_item_id = raw_negotiation.get("item_id")
        if isinstance(raw_item_id, str) and raw_item_id.strip():
            negotiation["item_id"] = raw_item_id.strip().upper()
        raw_round = raw_negotiation.get("round")
        if isinstance(raw_round, int) and not isinstance(raw_round, bool) and raw_round >= 0:
            negotiation["round"] = raw_round
        for key in ("last_ai_offer", "last_buyer_offer"):
            negotiation[key] = raw_negotiation.get(key)
        return negotiation

    @classmethod
    def _updated_negotiation_state(
        cls,
        state: Mapping[str, Any],
        updates: Mapping[str, object],
    ) -> dict[str, object]:
        unexpected = set(updates).difference(_NEGOTIATION_KEYS)
        if unexpected:
            raise ValueError("Unsupported negotiation keys: " + ", ".join(sorted(unexpected)))
        negotiation = cls._negotiation_from_state(state)
        if "item_id" in updates:
            negotiation["item_id"] = cls._normalise_item_id(updates["item_id"])
        if "round" in updates:
            round_number = updates["round"]
            if (
                not isinstance(round_number, int)
                or isinstance(round_number, bool)
                or round_number < 0
            ):
                raise ValueError("negotiation.round must be a non-negative integer")
            negotiation["round"] = round_number
        for key in ("last_ai_offer", "last_buyer_offer"):
            if key in updates:
                negotiation[key] = updates[key]
        return negotiation

    @classmethod
    def _validated_xianyu_context_updates(
        cls,
        updates: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if updates is None:
            return {}
        validated = dict(updates)
        unexpected = set(validated).difference(
            {"item_id", "recent_price_topic", "shipping_condition"}
        )
        if unexpected:
            raise ValueError("Unsupported Xianyu context keys: " + ", ".join(sorted(unexpected)))
        if "item_id" in validated:
            cls._normalise_item_id(validated["item_id"])
        recent_price_topic = validated.get("recent_price_topic", _UNSET)
        if recent_price_topic is not _UNSET and recent_price_topic is not None and (
            not isinstance(recent_price_topic, str) or not recent_price_topic.strip()
        ):
            raise ValueError("recent_price_topic must be a non-empty string or None")
        shipping_condition = validated.get("shipping_condition", _UNSET)
        if shipping_condition is not _UNSET and shipping_condition is not None and (
            shipping_condition not in _SHIPPING_CONDITIONS
        ):
            raise ValueError("shipping_condition must be seller_pays, buyer_pays, or None")
        return validated

    @classmethod
    def _normalise_state(cls, state: object) -> dict[str, Any]:
        raw_state = state if isinstance(state, Mapping) else {}
        normalized = {
            "order_id": raw_state.get("order_id"),
            "current_item_id": raw_state.get("current_item_id"),
            "last_intent": raw_state.get("last_intent"),
            "last_task_type": (
                raw_state.get("last_task_type")
                if raw_state.get("last_task_type") in TASK_TYPES
                else None
            ),
            "xianyu_context": cls._xianyu_context_from_state(raw_state),
            "negotiation": cls._negotiation_from_state(raw_state),
        }
        for key, value in raw_state.items():
            normalized.setdefault(key, value)
        return normalized

    @staticmethod
    def _normalise_item_id(value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("item_id must be a non-empty string or None")
        return value.strip().upper()

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
