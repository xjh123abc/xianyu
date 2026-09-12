"""Explicit operational controls for the S3 Xianyu worker.

Examples (all state changes are durable):
  python scripts/xianyu_stage3_control.py --db logs/xianyu_stage3.sqlite3 --account SELLER bind --listing PLATFORM_ITEM --item LOCAL_ITEM
  python scripts/xianyu_stage3_control.py --db logs/xianyu_stage3.sqlite3 --account SELLER pause
  python scripts/xianyu_stage3_control.py --db logs/xianyu_stage3.sqlite3 --account SELLER resume
  python scripts/xianyu_stage3_control.py --db logs/xianyu_stage3.sqlite3 --account SELLER release --chat xianyu:SELLER:CHAT --buyer BUYER
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.channels.xianyu.control import ChannelControl
from app.channels.xianyu.store import ChannelStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--account", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind")
    bind.add_argument("--listing", required=True, help="trusted platform item ID")
    bind.add_argument("--item", required=True, help="local item_id used by /chat")
    commands.add_parser("pause")
    commands.add_parser("resume")
    commands.add_parser("status")
    for name in ("takeover", "release"):
        command = commands.add_parser(name)
        command.add_argument("--chat", required=True)
        command.add_argument("--buyer", required=True)
    args = parser.parse_args()

    store = ChannelStore(args.db)
    control = ChannelControl(store, args.account)
    if args.command == "bind":
        store.bind_item(args.account, args.listing, args.item)
        output = {"result": "bound", "listing": args.listing, "item": args.item}
    elif args.command == "pause":
        output = {"result": "paused", "control_version": control.pause()}
    elif args.command == "resume":
        output = {"result": "resumed", "control_version": control.resume()}
    elif args.command == "takeover":
        output = {"result": "human", "control_version": control.takeover(args.chat, args.buyer)}
    elif args.command == "release":
        output = {"result": "auto", "control_version": control.release(args.chat, args.buyer)}
    else:
        output = control.status()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
