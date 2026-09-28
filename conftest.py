"""Process-wide test configuration and third-party compatibility shims."""

import os
from pathlib import Path

# Keep automated chat tests independent of the seller's live item catalog.
_TEST_ITEMS_PATH = str(
    Path(__file__).resolve().parent / "eval" / "fixtures" / "core_alignment_test_items.json"
)
_TEST_NEGOTIATION_POLICY_PATH = str(
    Path(__file__).resolve().parent
    / "eval"
    / "fixtures"
    / "core_alignment_test_negotiation_policies.json"
)
_TEST_XIANYU_KNOWLEDGE_PATH = str(
    Path(__file__).resolve().parent / "eval" / "fixtures" / "core_alignment_knowledge"
)
os.environ["XIANYU_ITEMS_PATH"] = _TEST_ITEMS_PATH
os.environ["XIANYU_NEGOTIATION_POLICY_PATH"] = _TEST_NEGOTIATION_POLICY_PATH
os.environ["XIANYU_KNOWLEDGE_BASE_PATH"] = _TEST_XIANYU_KNOWLEDGE_PATH

# NumPy/OpenBLAS otherwise creates one worker per logical CPU during pytest
# collection.  On Windows that can exhaust the commit limit before the test
# process reaches an assertion, producing unrelated model-load or SQLite I/O
# failures.  These settings must precede imports that may load NumPy or Torch.
for _thread_setting in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_setting] = "1"

from config.settings import settings

settings.xianyu_items_path = _TEST_ITEMS_PATH
settings.xianyu_negotiation_policy_path = _TEST_NEGOTIATION_POLICY_PATH
settings.xianyu_knowledge_base_path = _TEST_XIANYU_KNOWLEDGE_PATH

import anyio.abc
from anyio.from_thread import BlockingPortal


# Starlette 1.6.0 still evaluates anyio.abc.BlockingPortal in a type hint.
# AnyIO keeps that name as a deprecated lazy alias, so expose the canonical
# class before Starlette's TestClient is imported.
if "BlockingPortal" not in vars(anyio.abc):
    anyio.abc.BlockingPortal = BlockingPortal
