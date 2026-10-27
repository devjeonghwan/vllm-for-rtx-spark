"""Minimal Windows binding for the original MBX load-format import hook.

Not the original closed-source loader binary. No model runtime, graph owner,
weight requantization or normalization-dtype replacement is installed here.
"""

import importlib
import sys
import types
from pathlib import Path

NAME = "_mbx_windows_loader_candidate_v1"
HERE = Path(__file__).resolve().parent
if NAME not in sys.modules:
    package = types.ModuleType(NAME)
    package.__path__ = [str(HERE)]
    sys.modules[NAME] = package
elif list(sys.modules[NAME].__path__) != [str(HERE)]:
    raise RuntimeError("Native loader candidate package collision")

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

core = importlib.import_module(NAME + ".loader_core")
MbxModelLoader = core.make_loader_class(DefaultModelLoader)
MbxModelLoader.__module__ = __name__
register_model_loader("mbx")(MbxModelLoader)
