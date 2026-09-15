"""Unit tests for the Step 4 query router."""

from __future__ import annotations

import pytest

from app.services.chat_service import route_query


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("退货需要什么条件？", ("rag", None)),
        ("订单一般多久发货？", ("rag", None)),
        ("帮我查订单 TEST1001 发货了吗？", ("order", "TEST1001")),
        ("帮我查一下我的订单。", ("missing_order_id", None)),
        ("帮我取消订单 TEST1001。", ("unsupported_action", None)),
        ("退款条件是什么？", ("rag", None)),
    ],
)
def test_route_query_matches_documented_examples(
    query: str,
    expected: tuple[str, str | None],
) -> None:
    assert route_query(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("test1002 的物流信息是什么？", ("order", "TEST1002")),
        ("查询订单状态", ("missing_order_id", None)),
        ("我的订单什么时候发货？", ("missing_order_id", None)),
        ("帮我查物流 TEST123。", ("missing_order_id", None)),
        ("我要退款 TEST1001", ("unsupported_action", None)),
        ("请帮我修改收货地址", ("unsupported_action", None)),
        ("", ("rag", None)),
    ],
)
def test_route_query_handles_boundary_cases(
    query: str,
    expected: tuple[str, str | None],
) -> None:
    assert route_query(query) == expected
