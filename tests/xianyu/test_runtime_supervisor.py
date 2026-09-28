from __future__ import annotations

import asyncio

import pytest

from app.channels.xianyu.runtime_supervisor import (
    BoundedEventProcessor,
    HeartbeatTimeout,
    HeartbeatTracker,
    ReconnectPolicy,
    heartbeat_loop,
)


def test_heartbeat_requires_matching_acknowledgement() -> None:
    async def scenario() -> None:
        tracker = HeartbeatTracker()
        stop = asyncio.Event()
        sent: list[dict[str, object]] = []

        async def send(payload: dict[str, object]) -> None:
            sent.append(payload)
            tracker.acknowledge(payload["headers"]["mid"])
            stop.set()

        await heartbeat_loop(
            send,
            lambda: "heartbeat-1",
            tracker,
            stop,
            interval_seconds=0.001,
            acknowledgement_timeout_seconds=0.1,
        )

        assert len(sent) == 1

    asyncio.run(scenario())


def test_heartbeat_times_out_when_only_an_unrelated_ack_arrives() -> None:
    async def scenario() -> None:
        tracker = HeartbeatTracker()

        async def send(_payload: dict[str, object]) -> None:
            tracker.acknowledge("unrelated-message")

        with pytest.raises(HeartbeatTimeout):
            await heartbeat_loop(
                send,
                lambda: "heartbeat-1",
                tracker,
                asyncio.Event(),
                interval_seconds=0.001,
                acknowledgement_timeout_seconds=0.001,
            )

    asyncio.run(scenario())


def test_bounded_event_processor_preserves_fifo_order() -> None:
    async def scenario() -> None:
        seen: list[int] = []
        release = asyncio.Event()

        async def handle(value: int) -> None:
            if value == 1:
                await release.wait()
            seen.append(value)

        processor = BoundedEventProcessor(handle, capacity=1)
        processor.start()
        await processor.submit(1)
        await asyncio.sleep(0)
        await processor.submit(2)
        blocked_submit = asyncio.create_task(processor.submit(3))
        await asyncio.sleep(0)
        assert not blocked_submit.done()
        release.set()
        await blocked_submit
        await processor.close()

        assert seen == [1, 2, 3]
        assert processor.queue_size == 0

    asyncio.run(scenario())


def test_heartbeat_and_receive_queue_keep_running_while_handler_is_slow() -> None:
    async def scenario() -> None:
        handler_started = asyncio.Event()
        release_handler = asyncio.Event()
        heartbeat_sent = asyncio.Event()
        tracker = HeartbeatTracker()
        stop_heartbeat = asyncio.Event()
        received: list[int] = []

        async def handle(value: int) -> None:
            if value == 1:
                handler_started.set()
                await release_handler.wait()
            received.append(value)

        async def send_heartbeat(payload: dict[str, object]) -> None:
            tracker.acknowledge(payload["headers"]["mid"])
            heartbeat_sent.set()
            stop_heartbeat.set()

        processor = BoundedEventProcessor(handle, capacity=2)
        processor.start()
        await processor.submit(1)
        await handler_started.wait()
        await processor.submit(2)
        heartbeat = asyncio.create_task(
            heartbeat_loop(
                send_heartbeat,
                lambda: "heartbeat-during-slow-handler",
                tracker,
                stop_heartbeat,
                interval_seconds=0.001,
                acknowledgement_timeout_seconds=0.1,
            )
        )
        await asyncio.wait_for(heartbeat_sent.wait(), timeout=0.1)
        assert not release_handler.is_set()
        assert processor.queue_size == 1
        await heartbeat
        release_handler.set()
        await processor.close()
        assert received == [1, 2]

    asyncio.run(scenario())


def test_reconnect_policy_uses_bounded_backoff_and_resets_after_stable_session() -> None:
    policy = ReconnectPolicy()

    assert [policy.next_delay() for _ in range(5)] == [1, 2, 4, 8, 16]
    assert policy.exhausted

    policy.note_disconnect(300)
    assert not policy.exhausted
    assert policy.next_delay() == 1
