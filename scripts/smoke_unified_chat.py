"""Read-only acceptance runner for the live unified ``/chat`` endpoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
PostChat = Callable[[dict[str, str]], dict[str, Any]]


def _source_names(response: Mapping[str, Any]) -> set[str]:
    sources = response.get("sources", [])
    if not isinstance(sources, list):
        return set()
    return {
        str(source.get("source", "")).replace("\\", "/")
        for source in sources
        if isinstance(source, Mapping)
    }


def _has_source(response: Mapping[str, Any], suffix: str) -> bool:
    normalized_suffix = suffix.replace("\\", "/")
    return any(source.endswith(normalized_suffix) for source in _source_names(response))


def _answer(response: Mapping[str, Any]) -> str:
    return str(response.get("answer") or "")


def _record(
    case_id: str,
    inputs: Sequence[Mapping[str, str]],
    expected: str,
    responses: Sequence[Mapping[str, Any]],
    checks: Sequence[tuple[bool, str]],
    *,
    manual_review: str | None = None,
) -> dict[str, Any]:
    request_errors = [
        str(response["_request_error"])
        for response in responses
        if "_request_error" in response
    ]
    if request_errors:
        return {
            "case_id": case_id,
            "status": "阻塞",
            "input": list(inputs),
            "expected": expected,
            "response": list(responses),
            "reason": "真实 /chat 调用未完成：" + "；".join(request_errors),
        }
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        status = "失败"
        reason = "；".join(failures)
    elif manual_review:
        status = "阻塞"
        reason = f"自动边界检查通过；需人工对照真实资料复核：{manual_review}"
    else:
        status = "通过"
        reason = "；".join(reason for passed, reason in checks if passed)
    return {
        "case_id": case_id,
        "status": status,
        "input": list(inputs),
        "expected": expected,
        "response": list(responses),
        "reason": reason,
    }


def _blocked_record(case_id: str, expected: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "阻塞",
        "input": [],
        "expected": expected,
        "response": [],
        "reason": reason,
    }


def _request_many(post_chat: PostChat, payloads: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for payload in payloads:
        try:
            responses.append(post_chat(payload))
        except Exception as exc:
            responses.append({"_request_error": str(exc)})
    return responses


def run_acceptance(post_chat: PostChat) -> list[dict[str, Any]]:
    """Run A01-A14 without mutating product data or external chat state."""

    records: list[dict[str, Any]] = []

    a01_inputs = [
        {"query": "这个商品标价多少？", "chat_id": "qa_a01_001", "item_id": "DEMO_ITEM_001"},
        {"query": "这个商品还能买吗？", "chat_id": "qa_a01_002", "item_id": "DEMO_ITEM_002"},
        {"query": "这个商品还在售吗？", "chat_id": "qa_a01_003", "item_id": "DEMO_ITEM_003"},
    ]
    a01 = _request_many(post_chat, a01_inputs)
    records.append(
        _record(
            "A01",
            a01_inputs,
            "001 标价 1280 元；002 已售出；003 状态未知且不承诺可买。",
            a01,
            [
                ("1280.00" in _answer(a01[0]), "001 返回 1280.00 元"),
                ("已售出" in _answer(a01[1]), "002 返回已售出"),
                ("未知" in _answer(a01[2]) and a01[2].get("can_answer") is False, "003 明确状态未知且不作肯定承诺"),
                (all(_has_source(item, "mcp:get_item_info") for item in a01), "三项均包含真实商品工具来源"),
            ],
        )
    )

    a02_inputs = [{"query": "你们店售后怎么处理？", "chat_id": "qa_a02_general"}]
    a02 = _request_many(post_chat, a02_inputs)
    records.append(
        _record(
            "A02",
            a02_inputs,
            "新会话无商品也使用通用规则，不要求模式或商品。",
            a02,
            [
                (a02[0].get("action") != "clarify", "没有要求提供商品编号"),
                (_has_source(a02[0], "seller_rules.md"), "来源包含 seller_rules.md"),
            ],
            manual_review="售后表述与 seller_rules.md 原文和适用边界一致",
        )
    )

    a03_inputs = [{"query": "多少钱？收到有质量问题怎么办？", "chat_id": "qa_a03_combined", "item_id": "DEMO_ITEM_001"}]
    a03 = _request_many(post_chat, a03_inputs)
    records.append(
        _record(
            "A03",
            a03_inputs,
            "同次请求保留 1280 元商品事实和真实通用规则证据。",
            a03,
            [
                ("1280.00" in _answer(a03[0]), "保留 1280.00 元"),
                (_has_source(a03[0], "mcp:get_item_info"), "包含 MCP 商品来源"),
                (_has_source(a03[0], "seller_rules.md"), "包含 seller_rules.md"),
            ],
            manual_review="质量问题答复与已确认售后条款逐项一致",
        )
    )

    a04_inputs = [{"query": "这台相机以前摔过吗？多少钱？", "chat_id": "qa_a04_unknown", "item_id": "DEMO_ITEM_001"}]
    a04 = _request_many(post_chat, a04_inputs)
    records.append(
        _record(
            "A04",
            a04_inputs,
            "未知事实不得编造，且组合问题仍保留 1280 元。",
            a04,
            [
                ("1280.00" in _answer(a04[0]), "保留 1280.00 元"),
                (_has_source(a04[0], "mcp:get_item_info"), "包含 MCP 商品来源"),
                (_has_source(a04[0], "items/DEMO_ITEM_001.md"), "未知事实只检索 001 商品资料"),
                (not _has_source(a04[0], "items/DEMO_ITEM_002.md"), "未引入 002 商品资料"),
            ],
            manual_review="没有声称相机从未摔过，并明确资料未记录",
        )
    )

    a05_inputs = [{"query": "这个多少钱？", "chat_id": "qa_a05_missing_item"}]
    a05 = _request_many(post_chat, a05_inputs)
    records.append(
        _record(
            "A05",
            a05_inputs,
            "无商品上下文时询问具体商品，不随机选择。",
            a05,
            [
                (a05[0].get("action") == "clarify", "返回 clarify"),
                (a05[0].get("item_id") is None, "未随机绑定商品"),
                ("商品编号" in _answer(a05[0]), "要求提供商品编号"),
            ],
        )
    )

    a06_inputs = [{"query": "这个商品多少钱？", "chat_id": "qa_a06_missing", "item_id": "XXX999"}]
    a06 = _request_many(post_chat, a06_inputs)
    records.append(
        _record(
            "A06",
            a06_inputs,
            "明确未找到 XXX999，不用其他商品或旧价格代替。",
            a06,
            [
                (a06[0].get("item_id") == "XXX999", "保留请求编号 XXX999"),
                ("未找到商品 XXX999" in _answer(a06[0]), "明确告知 XXX999 未找到"),
                (a06[0].get("can_answer") is False, "未对不存在商品作肯定回答"),
                ("1280" not in _answer(a06[0]) and "560" not in _answer(a06[0]) and "980" not in _answer(a06[0]), "未使用其他商品价格"),
            ],
        )
    )

    a07_inputs = [
        {"query": "这个商品多少钱？", "chat_id": "qa_a07_memory", "item_id": "DEMO_ITEM_001"},
        {"query": "那售后怎么处理？", "chat_id": "qa_a07_memory"},
        {"query": "它有什么配件？", "chat_id": "qa_a07_memory"},
    ]
    a07 = _request_many(post_chat, a07_inputs)
    records.append(
        _record(
            "A07",
            a07_inputs,
            "同一会话三轮保持 001；规则和配件来自各自真实资料。",
            a07,
            [
                (a07[0].get("item_id") == "DEMO_ITEM_001" and a07[2].get("item_id") == "DEMO_ITEM_001", "第一、三轮均关联 001"),
                (a07[1].get("action") != "clarify", "第二轮没有要求重新选择商品或模式"),
                (_has_source(a07[1], "seller_rules.md"), "第二轮包含通用规则来源"),
                (_has_source(a07[2], "items/DEMO_ITEM_001.md"), "第三轮包含 001 商品资料"),
                (not _has_source(a07[2], "items/DEMO_ITEM_002.md"), "第三轮未混入 002"),
            ],
            manual_review="第三轮配件与 001 商品资料一致，第二轮规则无额外承诺",
        )
    )

    a08_inputs = [
        {"query": "这个商品多少钱？", "chat_id": "qa_a08_switch", "item_id": "DEMO_ITEM_001"},
        {"query": "这个商品还能买吗？", "chat_id": "qa_a08_switch", "item_id": "DEMO_ITEM_002"},
        {"query": "那它还能买吗？", "chat_id": "qa_a08_switch"},
        {"query": "这个商品多少钱？", "chat_id": "qa_a08_fresh"},
    ]
    a08 = _request_many(post_chat, a08_inputs)
    records.append(
        _record(
            "A08",
            a08_inputs,
            "显式切换后保持 002；新会话不继承商品。",
            a08,
            [
                (a08[1].get("item_id") == "DEMO_ITEM_002" and a08[2].get("item_id") == "DEMO_ITEM_002", "切换后及后续轮次均为 002"),
                ("已售出" in _answer(a08[1]) and "已售出" in _answer(a08[2]), "002 当轮及后续状态正确"),
                (a08[1].get("can_answer") is True and a08[2].get("can_answer") is True, "002 两轮均由结构化事实直接回答"),
                (a08[3].get("action") == "clarify" and a08[3].get("item_id") is None, "新会话没有继承商品"),
            ],
        )
    )

    a09_inputs = [{"query": "这个商品带 USB 数据线吗？", "chat_id": "qa_a09_filter", "item_id": "DEMO_ITEM_001"}]
    a09 = _request_many(post_chat, a09_inputs)
    records.append(
        _record(
            "A09",
            a09_inputs,
            "检索和最终来源不得引入只属于 002 的 USB 数据线资料。",
            a09,
            [
                (a09[0].get("item_id") == "DEMO_ITEM_001", "解析商品为 001"),
                (not _has_source(a09[0], "items/DEMO_ITEM_002.md"), "最终来源未引入 002"),
                (all("DEMO_ITEM_002.md" not in str(result.get("source", "")) for result in a09[0].get("results", [])), "返回的检索结果未引入 002"),
            ],
            manual_review="答案没有把 002 的 USB 数据线错误承诺给 001；Dense/BM25 过滤另由单元测试证明",
        )
    )

    a10_inputs = [{"query": "你们店支持七天无理由退货吗？", "chat_id": "qa_a10_old_promise"}]
    a10 = _request_many(post_chat, a10_inputs)
    records.append(
        _record(
            "A10",
            a10_inputs,
            "不承诺未确认的七天无理由退货，不回退旧全库。",
            a10,
            [
                (not any("data/raw/" in source for source in _source_names(a10[0])), "来源未回退 data/raw 旧全库"),
            ],
            manual_review="答案没有承诺七天无理由退货，且仅依据当前通用规则说明边界",
        )
    )

    a11_inputs = [{"query": "这个商品多少钱？", "chat_id": "qa_a11_price_only", "item_id": "DEMO_ITEM_001"}]
    a11 = _request_many(post_chat, a11_inputs)
    records.append(
        _record(
            "A11",
            a11_inputs,
            "纯价格请求由 MCP 事实直接回答，不受 RAG/Rerank 阈值阻断。",
            a11,
            [
                ("1280.00" in _answer(a11[0]), "返回 1280.00 元"),
                (a11[0].get("can_answer") is True, "有效 MCP 事实可回答"),
                (_source_names(a11[0]) == {"mcp:get_item_info"}, "来源仅为 MCP 商品工具"),
            ],
        )
    )

    records.append(
        _blocked_record(
            "A12",
            "模拟 MCP 失败时不编价格，仍保留可用售后规则并标明价格查询失败。",
            "真实服务验收脚本不得篡改运行配置来制造 MCP 故障；由 test_mcp_failure_keeps_independent_common_knowledge_answer 单元测试注入故障验证。",
        )
    )

    a13_inputs = [{"query": "这个商品多少钱？收到有质量问题怎么办？", "chat_id": "qa_a13_no_mode", "item_id": "DEMO_ITEM_001"}]
    a13 = _request_many(post_chat, a13_inputs)
    records.append(
        _record(
            "A13",
            a13_inputs,
            "请求仅含 query/chat_id/item_id，商品和规则证据都生效。",
            a13,
            [
                (set(a13_inputs[0]) == {"query", "chat_id", "item_id"}, "请求没有 scene/scenario"),
                ("1280.00" in _answer(a13[0]), "商品价格生效"),
                (_has_source(a13[0], "mcp:get_item_info") and _has_source(a13[0], "seller_rules.md"), "MCP 与规则来源同时存在"),
            ],
            manual_review="售后内容与 seller_rules.md 一致",
        )
    )

    records.append(
        _blocked_record(
            "A14",
            "旧测试、真实联调和清理检查均完成，公开输出无底价/Cookie 且无双主链路。",
            "本脚本只负责真实 /chat；pytest、真实 MCP 和静态清理检查必须在脚本外实际执行并记录，未执行前不能写通过。",
        )
    )
    return records


def _http_client(base_url: str, timeout_seconds: float) -> PostChat:
    endpoint = f"{base_url.rstrip('/')}/chat"

    def post_chat(payload: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST /chat returned HTTP {exc.code}: {detail}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cannot call live /chat at {endpoint}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"POST /chat returned non-object JSON: {decoded!r}")
        return decoded

    return post_chat


def _write_report(records: Sequence[Mapping[str, Any]], output: Path, base_url: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": base_url,
        "notes": [
            "This runner called the live HTTP /chat endpoint and did not replace application dependencies.",
            "阻塞 means the case still requires a manual or separately executed check; it is never counted as PASS.",
        ],
        "summary": {
            status: sum(record["status"] == status for record in records)
            for status in ("通过", "失败", "阻塞")
        },
        "cases": list(records),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or RESULTS_DIR / f"unified_chat_acceptance_{stamp}.json"
    try:
        records = run_acceptance(_http_client(args.base_url, args.timeout))
    except Exception as exc:
        print(f"Unified /chat acceptance could not complete: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    _write_report(records, output, args.base_url)
    for record in records:
        print(f"{record['case_id']}: {record['status']} - {record['reason']}")
    print(f"Report: {output.resolve()}")
    if any(record["status"] == "失败" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
