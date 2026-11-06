"""Shared locations for CPU contracts; prepared source is always explicit."""

import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ENV = "VLLM_RTX_SPARK_SOURCE_DIR"
SOURCE = Path(os.environ[SOURCE_ENV]).resolve() if os.environ.get(SOURCE_ENV) else None
SOURCE_REQUIRED = f"Set {SOURCE_ENV} to a freshly prepared vLLM source tree"

if SOURCE is not None and not (SOURCE / "vllm").is_dir():
    raise RuntimeError(f"{SOURCE_ENV} is not a prepared vLLM source tree: {SOURCE}")


def source_path(relative):
    if SOURCE is None:
        raise unittest.SkipTest(SOURCE_REQUIRED)
    path = SOURCE / relative
    if not path.is_file():
        raise FileNotFoundError(f"Prepared source is missing {relative}: {SOURCE}")
    return path
