"""Test-runner compatibility for Starlette 1.6.0 and AnyIO 4.15+."""

import anyio.abc
from anyio.from_thread import BlockingPortal


# Starlette 1.6.0 still evaluates anyio.abc.BlockingPortal in a type hint.
# AnyIO keeps that name as a deprecated lazy alias, so expose the canonical
# class before Starlette's TestClient is imported.
if "BlockingPortal" not in vars(anyio.abc):
    anyio.abc.BlockingPortal = BlockingPortal
