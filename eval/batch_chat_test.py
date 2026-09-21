"""批量调用真实本地 /chat 接口，记录 60 条闲鱼专家验收结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.query_planner import build_expert_plan


DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "results" / "batch_chat_results.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ITEM_ID = "DEMO_ITEM_001"


QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("T01", "是否在售", "这个还在吗？"),
    ("T02", "是否在售", "还没出吧？"),
    ("T03", "是否在售", "现在还能拍吗？"),
    ("T04", "是否在售", "东西卖掉了吗？"),
    ("T05", "是否在售", "现在下单还有货吗？"),
    ("T06", "价格", "这个多少钱？"),
    ("T07", "价格", "现在什么价出？"),
    ("T08", "价格", "标价就是最终价格吗？"),
    ("T09", "价格", "这个价格包含全部东西吗？"),
    ("T10", "价格", "我拍的话是多少钱？"),
    ("T11", "议价", "能便宜点吗？"),
    ("T12", "议价", "最低多少？"),
    ("T13", "议价", "诚心要，能少一点不？"),
    ("T14", "议价", "还能刀吗？"),
    ("T15", "议价", "直接拍能优惠多少？"),
    ("T16", "成色", "成色怎么样？"),
    ("T17", "成色", "外观看起来新不新？"),
    ("T18", "成色", "使用痕迹明显吗？"),
    ("T19", "成色", "有划痕吗？"),
    ("T20", "成色", "有没有磕碰？"),
    ("T21", "瑕疵/历史", "这个有什么毛病吗？"),
    ("T22", "瑕疵/历史", "有什么隐藏问题吗？"),
    ("T23", "瑕疵/历史", "以前摔过吗？"),
    ("T24", "瑕疵/历史", "有没有维修过？"),
    ("T25", "瑕疵/历史", "有拆修记录吗？"),
    ("T26", "功能", "功能都正常吗？"),
    ("T27", "功能", "现在可以正常使用吗？"),
    ("T28", "功能", "有没有哪个功能是坏的？"),
    ("T29", "功能", "快门正常吗？"),
    ("T30", "功能", "买回去能直接用吗？"),
    ("T31", "配件", "都带什么东西？"),
    ("T32", "配件", "配件齐全吗？"),
    ("T33", "配件", "有原装配件吗？"),
    ("T34", "配件", "图片里的东西都一起给吗？"),
    ("T35", "配件", "有没有说明书或者包装？"),
    ("T36", "发货", "什么时候能发？"),
    ("T37", "发货", "今天买今天能发吗？"),
    ("T38", "发货", "从哪里发货？"),
    ("T39", "发货", "包邮吗？"),
    ("T40", "发货", "用什么快递？"),
    ("T41", "商品信息", "这个具体是什么型号？"),
    ("T42", "商品信息", "哪一年生产的？"),
    ("T43", "商品信息", "这个适合新手吗？"),
    ("T44", "商品信息", "这个东西怎么用？"),
    ("T45", "商品信息", "为什么要卖？"),
    ("T46", "交易/售后", "可以走闲鱼交易吗？"),
    ("T47", "交易/售后", "收到发现有问题怎么办？"),
    ("T48", "交易/售后", "可以退吗？"),
    ("T49", "交易/售后", "能保证和描述的一样吗？"),
    ("T50", "交易/售后", "可以验货以后再确认收货吗？"),
    ("T51", "复合问题", "还在吗？有没有维修过？包邮吗？"),
    ("T52", "复合问题", "还在吗有没有维修过不包邮最低多少"),
    ("T53", "复合问题", "今天能发吗？走顺丰吗？"),
    ("T54", "价格条件", "包邮最低多少？不包邮呢？"),
    ("T55", "价格条件", "包邮1495可以吗？"),
    ("T56", "价格条件", "1470包邮可以吗？"),
    ("T57", "未知事实", "测光和手机对比过吗？"),
    ("T58", "条件依赖", "如果没修过，1470不包邮我就买"),
    ("T59", "型号知识", "Canon FTb 应该怎么上卷？"),
    ("T60", "复合问题", "多少钱？带哪些配件？你们店售后怎么处理？"),
)


def _post_chat(base_url: str, payload: dict[str, str], timeout: float) -> dict[str, Any]:
    """向真实本地 /chat 接口发送一条请求并返回 JSON 对象。"""
    endpoint = f"{base_url.rstrip('/')}/chat"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"无法连接 {endpoint}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("/chat 返回的内容不是有效 JSON") from exc

    if not isinstance(decoded, dict):
        raise RuntimeError(f"/chat 返回的 JSON 不是对象: {decoded!r}")
    _validate_response(decoded)
    return decoded


def _validate_response(response: dict[str, Any]) -> None:
    """Fail the batch on an unsafe or contradictory expert response contract."""

    action = response.get("action")
    answer = response.get("answer")
    can_answer = response.get("can_answer")
    if response.get("route") not in {"xianyu", "unified"}:
        raise RuntimeError("响应没有进入 xianyu 专家链路")
    if action not in {"reply", "clarify"}:
        raise RuntimeError(f"响应 action 无效: {action!r}")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("响应 answer 为空")
    if action == "reply" and can_answer is not True:
        raise RuntimeError("reply 与 can_answer 状态矛盾")
    if action == "clarify":
        if can_answer is not False:
            raise RuntimeError("clarify 与 can_answer 状态矛盾")
        if not isinstance(response.get("reason"), str) or not response["reason"].strip():
            raise RuntimeError("clarify 缺少内部原因")


def _result_record(
    question_id: str,
    category: str,
    question: str,
    chat_id: str,
    item_id: str,
    tasks: list[dict[str, Any]],
    elapsed_ms: float,
    response: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """转换为约定的结果格式，字段顺序与示例保持一致。"""
    if error_message is not None:
        return {
            "id": question_id,
            "category": category,
            "question": question,
            "chat_id": chat_id,
            "item_id": item_id,
            "answer": f"请求失败：{error_message}",
            "action": "error",
            "can_answer": False,
            "route": None,
            "tasks": tasks,
            "reason": error_message,
            "sources": [],
            "elapsed_ms": elapsed_ms,
        }

    assert response is not None
    return {
        "id": question_id,
        "category": category,
        "question": question,
        "chat_id": chat_id,
        "item_id": item_id,
        "answer": response.get("answer"),
        "action": response.get("action"),
        "can_answer": response.get("can_answer"),
        "route": response.get("route"),
        "tasks": tasks,
        "reason": response.get("reason"),
        "sources": response.get("sources", []),
        "elapsed_ms": elapsed_ms,
    }


def _task_records(question: str, item_id: str) -> list[dict[str, Any]]:
    """Record the validated deterministic task baseline without exposing it via HTTP."""

    return [
        {
            "task_id": task.task_id,
            "expert": task.expert,
            "question_fragment": task.question_fragment,
            "original_question": task.original_question,
            "normalized_question": task.normalized_question,
            "query_target": task.query_target,
            "knowledge_scope": task.knowledge_scope,
            "transaction_conditions": dict(task.transaction_conditions),
            "depends_on_task_ids": list(task.depends_on_task_ids),
        }
        for task in build_expert_plan(
            question,
            xianyu_context={"item_id": item_id},
        )
    ]


def run_batch(
    *,
    base_url: str = DEFAULT_BASE_URL,
    item_id: str = DEFAULT_ITEM_ID,
    timeout: float = 120.0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """按顺序测试全部或前 ``limit`` 条问题，且每题使用独立会话。"""

    if limit is not None and not 1 <= limit <= len(QUESTIONS):
        raise ValueError(f"limit must be between 1 and {len(QUESTIONS)}")

    selected_questions = QUESTIONS if limit is None else QUESTIONS[:limit]
    results: list[dict[str, Any]] = []
    for question_id, category, question in selected_questions:
        chat_id = f"qa_batch_chat_{question_id.lower()}"
        payload = {
            "query": question,
            "chat_id": chat_id,
            "item_id": item_id,
        }
        tasks = _task_records(question, item_id)
        started = time.perf_counter()
        try:
            response = _post_chat(base_url, payload, timeout)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            result = _result_record(
                question_id,
                category,
                question,
                chat_id,
                item_id,
                tasks,
                elapsed_ms,
                response=response,
            )
            print(f"{question_id} OK: {question}")
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            result = _result_record(
                question_id,
                category,
                question,
                chat_id,
                item_id,
                tasks,
                elapsed_ms,
                error_message=str(exc),
            )
            print(f"{question_id} ERROR: {question} - {exc}", file=sys.stderr)
        results.append(result)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI 服务地址")
    parser.add_argument("--item-id", default=DEFAULT_ITEM_ID, help="测试时绑定的商品编号")
    parser.add_argument("--timeout", type=float, default=120.0, help="单题请求超时时间（秒）")
    parser.add_argument(
        "--limit",
        type=int,
        help="只运行前 N 条固定用例；抽样结果不能作为 S6 的 60 条全量验收",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON 结果文件路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_batch(
        base_url=args.base_url,
        item_id=args.item_id,
        timeout=args.timeout,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = sum(result["action"] == "error" for result in results)
    report = {
        "schema_version": 1,
        "kind": "xianyu_expert_fixed_batch" if args.limit is None else "xianyu_expert_sample_batch",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "item_id": args.item_id,
        "summary": {
            "status": "completed" if failed == 0 else "failed",
            "total": len(results),
            "catalog_total": len(QUESTIONS),
            "scope": "full" if args.limit is None else "sample",
            "failed": failed,
            "requires_manual_review": True,
        },
        "cases": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已完成 {len(results)} 条测试，结果写入：{args.output.resolve()}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
