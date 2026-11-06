"""Minimal Windows binding for the original MBX load-format import hook.

Not the original closed-source loader binary. No model runtime, graph owner,
weight requantization or normalization-dtype replacement is installed here.
"""

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from . import loader_core

MbxModelLoader = loader_core.make_loader_class(DefaultModelLoader)
MbxModelLoader.__module__ = __name__
register_model_loader("mbx")(MbxModelLoader)
