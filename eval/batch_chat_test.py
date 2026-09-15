"""批量调用本地 /chat 接口，测试 50 条闲鱼买家问题。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    return decoded


def _result_record(
    question_id: str,
    question: str,
    response: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """转换为约定的结果格式，字段顺序与示例保持一致。"""
    if error_message is not None:
        return {
            "id": question_id,
            "question": question,
            "answer": f"请求失败：{error_message}",
            "action": "error",
            "can_answer": False,
            "route": None,
        }

    assert response is not None
    return {
        "id": question_id,
        "question": question,
        "answer": response.get("answer"),
        "action": response.get("action"),
        "can_answer": response.get("can_answer"),
        "route": response.get("route"),
    }


def run_batch(
    *,
    base_url: str = DEFAULT_BASE_URL,
    item_id: str = DEFAULT_ITEM_ID,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """按顺序测试全部问题；每题使用独立 chat_id，避免跨题共享会话。"""
    results: list[dict[str, Any]] = []
    for question_id, _category, question in QUESTIONS:
        payload = {
            "query": question,
            "chat_id": f"qa_batch_chat_{question_id.lower()}",
            "item_id": item_id,
        }
        try:
            response = _post_chat(base_url, payload, timeout)
            result = _result_record(question_id, question, response=response)
            print(f"{question_id} OK: {question}")
        except Exception as exc:
            result = _result_record(question_id, question, error_message=str(exc))
            print(f"{question_id} ERROR: {question} - {exc}", file=sys.stderr)
        results.append(result)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI 服务地址")
    parser.add_argument("--item-id", default=DEFAULT_ITEM_ID, help="测试时绑定的商品编号")
    parser.add_argument("--timeout", type=float, default=120.0, help="单题请求超时时间（秒）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON 结果文件路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_batch(
        base_url=args.base_url,
        item_id=args.item_id,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已完成 {len(results)} 条测试，结果写入：{args.output.resolve()}")


if __name__ == "__main__":
    main()
