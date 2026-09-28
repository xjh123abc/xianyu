"""Bounded channel runtime primitives for receiving, processing and heartbeats."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass
class ReconnectPolicy:
    """Bound ordinary reconnects and reset the budget after a stable session."""

    max_attempts: int = 5
    stable_reset_seconds: float = 300.0
    maximum_delay_seconds: float = 30.0
    attempts: int = 0

    def note_disconnect(self, connected_seconds: float) -> None:
        if connected_seconds >= self.stable_reset_seconds:
            self.attempts = 0

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts

    def next_delay(self) -> float:
        if self.exhausted:
            raise RuntimeError("reconnect attempts are exhausted")
        self.attempts += 1
        return min(2 ** (self.attempts - 1), self.maximum_delay_seconds)


class SerializedWebSocket:
    """Serialize protocol acknowledgements, heartbeats and replies on one socket."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self._send_lock = asyncio.Lock()

    async def send(self, payload: str) -> None:
        async with self._send_lock:
            await self.websocket.send(payload)


class HeartbeatTimeout(ConnectionError):
    """Raised when the platform does not acknowledge the matching heartbeat."""


class HeartbeatTracker:
    """Match protocol acknowledgements to the heartbeat mid that was sent."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[None]] = {}

    def expect(self, message_id: str) -> asyncio.Future[None]:
        if not message_id or message_id in self._pending:
            raise ValueError("heartbeat message id must be unique and non-empty")
        future = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        return future

    def acknowledge(self, message_id: object) -> bool:
        key = str(message_id or "")
        future = self._pending.pop(key, None)
        if future is None or future.done():
            return False
        future.set_result(None)
        return True

    def discard(self, message_id: str) -> None:
        future = self._pending.pop(message_id, None)
        if future is not None and not future.done():
            future.cancel()


async def heartbeat_loop(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    generate_mid: Callable[[], str],
    tracker: HeartbeatTracker,
    stop: asyncio.Event,
    *,
    interval_seconds: float = 15.0,
    acknowledgement_timeout_seconds: float = 10.0,
) -> None:
    """Send periodic heartbeats and fail the connection on a missing/mismatched ack."""

    if interval_seconds <= 0 or acknowledgement_timeout_seconds <= 0:
        raise ValueError("heartbeat intervals must be greater than zero")
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            return
        except asyncio.TimeoutError:
            pass
        message_id = str(generate_mid())
        acknowledgement = tracker.expect(message_id)
        try:
            await send({"lwp": "/!", "headers": {"mid": message_id}})
            await asyncio.wait_for(
                asyncio.shield(acknowledgement),
                timeout=acknowledgement_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise HeartbeatTimeout("matching heartbeat acknowledgement timed out") from error
        finally:
            tracker.discard(message_id)


class BoundedEventProcessor(Generic[T]):
    """A bounded FIFO with a fixed number of consumers and deterministic shutdown."""

    def __init__(
        self,
        handler: Callable[[T], Awaitable[None]],
        *,
        capacity: int = 128,
        consumers: int = 1,
        on_error: Callable[[T, Exception], None] | None = None,
    ) -> None:
        if capacity <= 0 or consumers <= 0:
            raise ValueError("capacity and consumers must be greater than zero")
        self._handler = handler
        self._on_error = on_error
        self._queue: asyncio.Queue[T | None] = asyncio.Queue(maxsize=capacity)
        self._consumer_count = consumers
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._tasks:
            raise RuntimeError("event processor is already running")
        self._tasks = [
            asyncio.create_task(self._consume(), name=f"xianyu-event-consumer-{index}")
            for index in range(self._consumer_count)
        ]

    async def submit(self, event: T) -> None:
        if not self._tasks:
            raise RuntimeError("event processor has not been started")
        await self._queue.put(event)

    async def close(self) -> None:
        if not self._tasks:
            return
        await self._queue.join()
        for _ in self._tasks:
            await self._queue.put(None)
        await asyncio.gather(*self._tasks)
        self._tasks.clear()

    async def _consume(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                try:
                    await self._handler(event)
                except Exception as error:
                    if self._on_error is not None:
                        self._on_error(event, error)
            finally:
                self._queue.task_done()
