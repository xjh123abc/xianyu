"""Acceptance check for the S3 unified Planner boundary.

The script intentionally obtains the Planner from a real ``ChatService``
instance so it exercises the same Planner wiring as the application.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chat_contracts import SessionContext
from app.services.chat_service import ChatService


QUESTIONS = (
    "这个相机修过吗？",
    "最低多少？",
    "一般多久发货？",
    "TEST1001 到哪了？",
    "这个修过吗？最低多少？多久发货？",
    "这个相机修过吗？最低多少？周日能到吗？"
)


def main() -> None:
    chat_service = ChatService()
    planner = chat_service.planner
    context = SessionContext()

    print("S3 Planner verification")
    for question in QUESTIONS:
        tasks = planner.plan(question, context)
        task_types = [task.task_type for task in tasks]
        print(f"{question} -> {task_types}")


if __name__ == "__main__":
    main()
