"""Process-wide test configuration and third-party compatibility shims."""

import os

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

import anyio.abc
from anyio.from_thread import BlockingPortal


# Starlette 1.6.0 still evaluates anyio.abc.BlockingPortal in a type hint.
# AnyIO keeps that name as a deprecated lazy alias, so expose the canonical
# class before Starlette's TestClient is imported.
if "BlockingPortal" not in vars(anyio.abc):
    anyio.abc.BlockingPortal = BlockingPortal
