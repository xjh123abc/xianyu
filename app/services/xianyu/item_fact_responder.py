"""Deterministic answers from confirmed, seller-maintained item facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.services.intent_router import IntentMatch
from app.services.query_planner import QuestionPlan
from app.services.xianyu.experts.price_agent import PriceAgent
from app.services.xianyu.responses import clarification, handoff, reply


@dataclass(frozen=True)
class PlannedFactAnswer:
    """Facts and evidence gaps found while handling a multi-part item question."""

    answers: list[str]
    unresolved: list[str]
    has_conflict: bool = False


class ItemFactResponder:
    """Answer only facts explicitly supplied by the seller/MCP item record."""

    _TARGET_MATCHES: dict[str, IntentMatch] = {
        "availability.sale_status": IntentMatch("AVAILABILITY", ("sale_status",), "rule"),
        "history.repair_history": IntentMatch("REPAIR_HISTORY", ("history",), "rule"),
        "history.disassembly_history": IntentMatch("REPAIR_HISTORY", ("history",), "rule"),
        "history.drop_history": IntentMatch("REPAIR_HISTORY", ("history",), "rule"),
        "function.shutter": IntentMatch("FUNCTION", ("function",), "rule"),
        "function.overall": IntentMatch("FUNCTION", ("function",), "rule"),
        "condition.summary": IntentMatch("CONDITION", ("condition",), "rule"),
        "condition.scratches": IntentMatch("DEFECT", ("condition",), "rule"),
        "condition.dents": IntentMatch("DEFECT", ("condition",), "rule"),
        "condition.known_issues": IntentMatch("DEFECT", ("condition",), "rule"),
        "lens.details": IntentMatch("ACCESSORIES", ("accessories", "accessory_details"), "rule"),
        "lens.focal_length_mm": IntentMatch("ACCESSORIES", ("accessories", "accessory_details"), "rule"),
        "accessories.items": IntentMatch("ACCESSORIES", ("accessories", "accessory_details"), "rule"),
        "accessories.completeness": IntentMatch("ACCESSORIES", ("accessories", "accessory_details"), "rule"),
        "accessories.original": IntentMatch("ACCESSORIES", ("accessories", "accessory_details"), "rule"),
        "accessories.manual_or_packaging": IntentMatch("ACCESSORIES", ("accessories", "accessory_details"), "rule"),
        "identity.model": IntentMatch("PRODUCT_INFO", ("identity", "product_info"), "rule"),
        "product_info.production_year": IntentMatch("PRODUCT_INFO", ("identity", "product_info"), "rule"),
        "product_info.beginner_suitability": IntentMatch("PRODUCT_INFO", ("identity", "product_info"), "rule"),
        "product_info.usage": IntentMatch("PRODUCT_INFO", ("identity", "product_info"), "rule"),
        "product_info.sale_reason": IntentMatch("PRODUCT_INFO", ("identity", "product_info"), "rule"),
        "shipping.dispatch_time": IntentMatch("SHIPPING_TIME", ("shipping",), "rule"),
        "shipping.ship_from": IntentMatch("SHIPPING_TIME", ("shipping",), "rule"),
        "shipping.carrier": IntentMatch("SHIPPING_TIME", ("shipping",), "rule"),
        "shipping.fee": IntentMatch("SHIPPING_FEE", ("shipping",), "rule"),
        "after_sale.return_policy": IntentMatch("AFTER_SALE", ("after_sale",), "rule"),
        "after_sale.transaction_channel": IntentMatch("AFTER_SALE", ("after_sale",), "rule"),
        "after_sale.description_policy": IntentMatch("AFTER_SALE", ("after_sale",), "rule"),
        "after_sale.inspection_confirmation": IntentMatch("AFTER_SALE", ("after_sale",), "rule"),
    }

    def __init__(self, price_agent: PriceAgent | None = None) -> None:
        self._price_agent = price_agent or PriceAgent()

    def answer_target(
        self,
        question: str,
        item: Mapping[str, object],
        query_target: str,
    ) -> dict[str, object]:
        """Answer the planner-selected target without rerouting buyer text."""

        match = self._TARGET_MATCHES.get(query_target)
        if match is None:
            return handoff(question, "unsupported_query_target", item)
        return self.answer_intent(
            question,
            item,
            match,
            query_target=query_target,
        )

    @classmethod
    def required_fields_for_target(cls, query_target: str) -> tuple[str, ...]:
        match = cls._TARGET_MATCHES.get(query_target)
        return match.required_fields if match is not None else ()

    def answer_intent(
        self,
        query: str,
        item: Mapping[str, object],
        match: IntentMatch,
        *,
        query_target: str | None = None,
    ) -> dict[str, object]:
        """Answer a classified buyer intent from explicit seller facts only."""

        if match.intent in {"PRICE", "BARGAIN"}:
            decision = self._price_agent.decide(query, item, match)
            if decision.status == "answered":
                assert decision.answer is not None
                return reply(query, decision.answer, item)
            return handoff(query, decision.reason or "price_decision_unavailable", item)

        facts = self.structured_facts(item)
        if set(match.required_fields) & self.fact_conflicts(facts):
            return handoff(
                query,
                "item_fact_conflict",
                item,
            )

        def known(value: object) -> str | None:
            return (
                value.strip()
                if isinstance(value, str) and value.strip() != "unknown"
                else None
            )

        def mapping_value(name: str, key: str) -> str | None:
            raw = facts.get(name)
            return known(raw.get(key)) if isinstance(raw, Mapping) else None

        def answer(text: str) -> dict[str, object]:
            return reply(query, text, item)

        def needs_human(message: str) -> dict[str, object]:
            return handoff(query, message, item)

        intent = match.intent
        lowered = query.casefold()
        if intent == "AVAILABILITY":
            status = item.get("sale_status")
            if status == "listed":
                return answer("还在的，这台目前还没出。")
            if status == "sold":
                return answer("这件已经出掉了。")
            return needs_human("sale_status_unavailable")

        if intent == "CONDITION":
            key = "summary"
            label = "成色"
            if query_target == "condition.scratches" or (
                query_target is None and "划痕" in lowered
            ):
                key, label = "scratches", "划痕"
            elif query_target == "condition.dents" or (
                query_target is None and "磕碰" in lowered
            ):
                key, label = "dents", "磕碰"
            value = mapping_value("condition", key)
            if value:
                return answer(self._natural_fact_value(value))
            return needs_human(f"{label}_unavailable")

        if intent == "DEFECT":
            if query_target in {"condition.scratches", "condition.dents"} or (
                query_target is None and ("划痕" in lowered or "磕碰" in lowered)
            ):
                key = "scratches" if query_target == "condition.scratches" or "划痕" in lowered else "dents"
                label = "划痕" if key == "scratches" else "磕碰"
                value = mapping_value("condition", key)
                if value:
                    return answer(self._natural_fact_value(value))
                return needs_human(f"{label}_unavailable")
            issues = facts.get("condition")
            known_issues = issues.get("known_issues") if isinstance(issues, Mapping) else None
            if isinstance(known_issues, list) and all(
                isinstance(entry, str) for entry in known_issues
            ):
                if known_issues:
                    return answer(self._natural_fact_value("、".join(known_issues)))
                return answer("暂未发现已知问题。")
            return needs_human("known_issues_unavailable")

        if intent == "REPAIR_HISTORY":
            key = "repair_history"
            label = "维修历史"
            if query_target == "history.disassembly_history" or (
                query_target is None and "拆" in lowered
            ):
                key, label = "disassembly_history", "拆修记录"
            elif query_target == "history.drop_history" or (
                query_target is None and ("摔" in lowered or "跌" in lowered)
            ):
                key, label = "drop_history", "摔碰历史"
            value = mapping_value("history", key)
            if value:
                return answer(self._natural_fact_value(value))
            return needs_human(f"{label}_unavailable")

        if intent == "FUNCTION":
            key = "shutter" if query_target == "function.shutter" or (
                query_target is None and "快门" in lowered
            ) else "overall"
            value = mapping_value("function", key)
            if value == "working":
                return answer(
                    "快门功能正常。"
                    if key == "shutter"
                    else "目前记录的功能状态正常，可以使用。"
                )
            if value == "not_working":
                return answer("该功能目前记录为异常。")
            if value == "not_applicable":
                return answer("这件商品不适用快门功能。")
            return needs_human("function_status_unavailable")

        if intent == "ACCESSORIES":
            if query_target in {"lens.details", "lens.focal_length_mm"} or (
                query_target is None and "镜头" in lowered
            ):
                if "lens" in self.fact_conflicts(facts):
                    return needs_human("lens_fact_conflict")
                text, unresolved = self.lens_fact_answer(facts.get("lens"))
                return needs_human(text) if unresolved else answer(text)
            detail_key = None
            label = "配件"
            if query_target == "accessories.completeness" or (
                query_target is None and "齐全" in lowered
            ):
                detail_key, label = "completeness", "配件是否齐全"
            elif query_target == "accessories.original" or (
                query_target is None and "原装" in lowered
            ):
                detail_key, label = "original_accessories", "原装配件"
            elif query_target is None and "图片" in lowered:
                detail_key, label = "image_items", "图片中的物品"
            elif query_target == "accessories.manual_or_packaging" or (
                query_target is None and ("说明书" in lowered or "包装" in lowered)
            ):
                detail_key, label = "manual_or_packaging", "说明书或包装"
            if detail_key is not None:
                value = mapping_value("accessory_details", detail_key)
                if value:
                    return answer(self._natural_fact_value(value))
                return needs_human(f"{label}_unavailable")
            accessories = facts.get("accessories")
            if isinstance(accessories, list) and all(
                isinstance(entry, str) for entry in accessories
            ):
                return answer("一起出的有 " + "、".join(accessories) + "。")
            return needs_human("accessories_unavailable")

        if intent in {"SHIPPING_TIME", "SHIPPING_FEE"}:
            key, label = (
                ("shipping_fee", "运费")
                if intent == "SHIPPING_FEE"
                else ("dispatch_time", "发货时间")
            )
            if query_target == "shipping.ship_from" or (
                query_target is None and "从哪里" in lowered
            ):
                key, label = "ship_from", "发货地"
            elif query_target == "shipping.carrier" or (
                query_target is None
                and any(
                    term in lowered
                    for term in ("快递", "顺丰", "中通", "圆通", "韵达", "京东")
                )
            ):
                key, label = "carrier", "快递"
            value = mapping_value("shipping", key)
            if value:
                return answer(self._natural_fact_value(value))
            return needs_human(f"{label}_unavailable")

        if intent == "PRODUCT_INFO":
            if query_target == "identity.model" or (
                query_target is None and "型号" in lowered
            ):
                identity = facts.get("identity")
                model = known(identity.get("model")) if isinstance(identity, Mapping) else None
                brand = known(identity.get("brand")) if isinstance(identity, Mapping) else None
                if model:
                    return answer(
                        "这件的型号是："
                        + " ".join(part for part in (brand, model) if part)
                        + "。"
                    )
                return needs_human("model_unavailable")
            key, label = "production_year", "生产年份"
            if query_target == "product_info.beginner_suitability" or (
                query_target is None and "新手" in lowered
            ):
                key, label = "beginner_suitability", "是否适合新手"
            elif query_target == "product_info.usage" or (
                query_target is None and "怎么用" in lowered
            ):
                key, label = "usage", "使用方法"
            elif query_target == "product_info.sale_reason" or (
                query_target is None and ("为什么" in lowered or "卖" in lowered)
            ):
                key, label = "sale_reason", "出售原因"
            value = mapping_value("product_info", key)
            if value == "yes":
                return answer("适合新手；建议先阅读基础使用教程再开始使用。")
            if value == "no":
                return answer("这件不太适合新手，建议有相关使用经验后再考虑。")
            if value:
                return answer(self._natural_fact_value(value))
            return needs_human(f"{label}_unavailable")

        if intent == "AFTER_SALE":
            key, label = "return_policy", "退货和售后规则"
            if query_target == "after_sale.transaction_channel" or (
                query_target is None and "闲鱼交易" in lowered
            ):
                key, label = "transaction_channel", "交易方式"
            elif query_target == "after_sale.description_policy" or (
                query_target is None and "描述" in lowered
            ):
                key, label = "description_policy", "描述一致性"
            elif query_target == "after_sale.inspection_confirmation" or (
                query_target is None and ("验货" in lowered or "确认收货" in lowered)
            ):
                key, label = "inspection_confirmation", "验货和确认收货方式"
            value = mapping_value("after_sale", key)
            if key == "transaction_channel" and value == "xianyu":
                return answer("可以走闲鱼平台交易。")
            if key == "description_policy" and value == "seller_confirmed":
                return answer("商品按现有描述发布；下单前可把在意的细节再确认一次。")
            if value and value not in {"manual_review", "unknown"}:
                return answer(self._natural_fact_value(value))
            return needs_human(f"{label}_unavailable")

        return clarification(query, item_id=str(item["item_id"]))

    def answer_plan(
        self,
        query: str,
        item: Mapping[str, object],
        plan: QuestionPlan,
    ) -> PlannedFactAnswer:
        """Resolve fact portions of a multi-part question from the same rules."""

        fact_answers: list[str] = []
        unresolved_answers: list[str] = []
        facts = self.structured_facts(item)
        requested_structured_fields = {
            field
            for field in ("identity", "lens", "included_items", "condition", "history")
            if field in plan["item_fields"]
        }
        if requested_structured_fields & self.fact_conflicts(facts):
            return PlannedFactAnswer([], [], has_conflict=True)

        if "listed_price_cents" in plan["item_fields"]:
            cents = int(item["listed_price_cents"])
            fact_answers.append(f"这台标价 ¥{cents / 100:.2f}。")
        if "sale_status" in plan["item_fields"]:
            status = item["sale_status"]
            if status == "listed":
                fact_answers.append("还在的，这台目前还没出。")
            elif status == "sold":
                fact_answers.append("这台已经出掉了。")
            else:
                unresolved_answers.append("这台现在还在不在，我这边暂时没确认。")

        field_handlers = {
            "lens": self.lens_fact_answer,
            "included_items": self.included_items_fact_answer,
            "condition": self.condition_fact_answer,
            "identity": self.identity_fact_answer,
        }
        for field, handler in field_handlers.items():
            if field not in requested_structured_fields:
                continue
            text, unresolved = handler(facts.get(field))
            (unresolved_answers if unresolved else fact_answers).append(text)
        if "history" in requested_structured_fields:
            text, unresolved = self.history_fact_answer(query, facts.get("history"))
            (unresolved_answers if unresolved else fact_answers).append(text)
        return PlannedFactAnswer(fact_answers, unresolved_answers)

    @staticmethod
    def _natural_fact_value(value: str) -> str:
        """Render one confirmed seller value without exposing its storage label."""

        text = value.strip()
        return text if text.endswith(("。", "！", "？", "!", "?")) else f"{text}。"

    @classmethod
    def history_fact_answer(cls, query: str, value: object) -> tuple[str, bool]:
        """Answer the history facet actually asked, preserving explicit negatives."""

        if not isinstance(value, Mapping):
            return "这台的维修、拆修和摔碰历史这边还没确认。", True
        lowered = query.casefold()
        key = "repair_history"
        if "拆" in lowered:
            key = "disassembly_history"
        elif "摔" in lowered or "跌" in lowered:
            key = "drop_history"
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip() and raw.strip() != "unknown":
            return cls._natural_fact_value(raw), False
        return "这台的维修、拆修和摔碰历史这边还没确认。", True

    @staticmethod
    def structured_facts(item: Mapping[str, object]) -> Mapping[str, object]:
        """Return only validated-shaped structured facts, never inferred text."""

        facts = item.get("facts")
        return facts if isinstance(facts, Mapping) else {}

    @staticmethod
    def fact_conflicts(facts: Mapping[str, object]) -> set[str]:
        conflicts = facts.get("fact_conflicts")
        if not isinstance(conflicts, list):
            return set()
        return {entry for entry in conflicts if isinstance(entry, str)}

    @staticmethod
    def lens_fact_answer(value: object) -> tuple[str, bool]:
        if not isinstance(value, Mapping):
            return "镜头信息这边还没确认。", True
        included = value.get("included")
        if included is False:
            return "这台不带镜头。", False
        if included is not True:
            return "带不带镜头这边还没确认。", True
        details: list[str] = []
        model = value.get("model")
        focal_length = value.get("focal_length_mm")
        if isinstance(model, str) and model.strip():
            details.append(f"配的是 {model.strip()}")
        if (
            isinstance(focal_length, int)
            and not isinstance(focal_length, bool)
            and focal_length > 0
        ):
            details.append(f"{focal_length}mm 焦段")
        if not details:
            return "带镜头，具体型号和焦段这边还没确认。", False
        return "带的，" + "，".join(details) + "。", False

    @staticmethod
    def identity_fact_answer(value: object) -> tuple[str, bool]:
        if not isinstance(value, Mapping):
            return "品牌和型号这边还没确认。", True
        parts = [
            raw.strip()
            for raw in (value.get("brand"), value.get("model"), value.get("category"))
            if isinstance(raw, str) and raw.strip()
        ]
        if not parts:
            return "品牌和型号这边还没确认。", True
        return "这是 " + " ".join(parts) + "。", False

    @staticmethod
    def included_items_fact_answer(value: object) -> tuple[str, bool]:
        if value is None:
            return "配件这边还没确认。", True
        if not isinstance(value, list) or not all(
            isinstance(entry, str) and entry.strip() for entry in value
        ):
            return "配件这边还没确认。", True
        if not value:
            return "目前没有确认随附配件。", False
        return "一起出的有：" + "、".join(value) + "。", False

    @staticmethod
    def condition_fact_answer(value: object) -> tuple[str, bool]:
        if not isinstance(value, Mapping):
            return "成色和功能这边还没确认。", True
        details: list[str] = []
        appearance = value.get("appearance")
        function = value.get("function")
        known_issues = value.get("known_issues")
        if isinstance(appearance, str) and appearance.strip():
            details.append(appearance.strip())
        if isinstance(function, str) and function.strip():
            details.append(function.strip())
        if isinstance(known_issues, list) and all(
            isinstance(issue, str) for issue in known_issues
        ):
            if known_issues:
                details.append("已知问题有 " + "、".join(known_issues))
            else:
                details.append("暂时没有记录已知问题")
        if not details:
            return "成色和功能这边还没确认。", True
        return "这台" + "；".join(details) + "。", False

    @staticmethod
    def attach_intent_metadata(
        response: dict[str, object],
        match: IntentMatch,
        item: Mapping[str, object] | None,
    ) -> None:
        """Expose decision inputs needed for repeatable batch evaluation."""

        response["intent"] = match.intent
        response["required_fields"] = list(match.required_fields)
        selected: dict[str, object] = {}
        facts = item.get("facts") if isinstance(item, Mapping) else None
        for field in match.required_fields:
            if field in {"listed_price_cents", "sale_status"} and isinstance(item, Mapping):
                selected[field] = item.get(field)
            elif isinstance(facts, Mapping):
                selected[field] = facts.get(field)
        response["facts"] = selected
