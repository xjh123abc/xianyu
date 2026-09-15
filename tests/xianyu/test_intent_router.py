"""Acceptance coverage for the 50 buyer phrasings in the batch evaluation."""

from __future__ import annotations

import pytest

from app.services.intent_router import IntentRouter


CASES = (
    ("这个还在吗？", "AVAILABILITY"), ("还没出吧？", "AVAILABILITY"),
    ("现在还能拍吗？", "AVAILABILITY"), ("东西卖掉了吗？", "AVAILABILITY"),
    ("现在下单还有货吗？", "AVAILABILITY"), ("这个多少钱？", "PRICE"),
    ("现在什么价出？", "PRICE"), ("标价就是最终价格吗？", "PRICE"),
    ("这个价格包含全部东西吗？", "ACCESSORIES"), ("我拍的话是多少钱？", "PRICE"),
    ("能便宜点吗？", "BARGAIN"), ("最低多少？", "BARGAIN"),
    ("诚心要，能少一点不？", "BARGAIN"), ("还能刀吗？", "BARGAIN"),
    ("直接拍能优惠多少？", "BARGAIN"), ("成色怎么样？", "CONDITION"),
    ("外观看起来新不新？", "CONDITION"), ("使用痕迹明显吗？", "CONDITION"),
    ("有划痕吗？", "DEFECT"), ("有没有磕碰？", "DEFECT"),
    ("这个有什么毛病吗？", "DEFECT"), ("有什么隐藏问题吗？", "DEFECT"),
    ("以前摔过吗？", "REPAIR_HISTORY"), ("有没有维修过？", "REPAIR_HISTORY"),
    ("有拆修记录吗？", "REPAIR_HISTORY"), ("功能都正常吗？", "FUNCTION"),
    ("现在可以正常使用吗？", "FUNCTION"), ("有没有哪个功能是坏的？", "FUNCTION"),
    ("快门正常吗？", "FUNCTION"), ("买回去能直接用吗？", "FUNCTION"),
    ("都带什么东西？", "ACCESSORIES"), ("配件齐全吗？", "ACCESSORIES"),
    ("有原装配件吗？", "ACCESSORIES"), ("图片里的东西都一起给吗？", "ACCESSORIES"),
    ("有没有说明书或者包装？", "ACCESSORIES"), ("什么时候能发？", "SHIPPING_TIME"),
    ("今天买今天能发吗？", "SHIPPING_TIME"), ("从哪里发货？", "SHIPPING_TIME"),
    ("包邮吗？", "SHIPPING_FEE"), ("用什么快递？", "SHIPPING_TIME"),
    ("这个具体是什么型号？", "PRODUCT_INFO"), ("哪一年生产的？", "PRODUCT_INFO"),
    ("这个适合新手吗？", "PRODUCT_INFO"), ("这个东西怎么用？", "PRODUCT_INFO"),
    ("为什么要卖？", "PRODUCT_INFO"), ("可以走闲鱼交易吗？", "AFTER_SALE"),
    ("收到发现有问题怎么办？", "AFTER_SALE"), ("可以退吗？", "AFTER_SALE"),
    ("能保证和描述的一样吗？", "AFTER_SALE"), ("可以验货以后再确认收货吗？", "AFTER_SALE"),
)


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_batch_buyer_phrasings_use_rules_before_ai(query: str, expected: str) -> None:
    def unexpected_ai(_: str) -> str:
        raise AssertionError("the 50 fixed evaluation phrasings must not need AI")

    match = IntentRouter(classifier=unexpected_ai).route(query)

    assert match.intent == expected
    assert match.source == "rule"
    assert match.required_fields


def test_ai_fallback_accepts_only_known_classification_labels() -> None:
    assert IntentRouter(classifier=lambda _: "BARGAIN").route("能不能再商量一下呀").intent == "BARGAIN"
    assert IntentRouter(classifier=lambda _: "give_discount").route("能不能再商量一下呀").intent == "OTHER"
