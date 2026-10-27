"""Custom code reference resolver for dynamic function loading.

Provides loaders for different handler specification formats:

File-based (with .py extension):
- "model.py:predict_fn" → loads predict_fn() from model.py file
- "handler.py:MyClass.process" → loads MyClass.process method from handler.py
- "utils/preprocess.py:clean_data" → loads clean_data() from utils/preprocess.py

Module-based (no .py extension):
- "mypackage:handler_fn" → loads handler_fn() from installed mypackage module
- "model:predict" → loads predict() from customer's model.py (special handling)
- "sklearn.preprocessing:StandardScaler" → loads StandardScaler from sklearn

The FunctionLoader provides a unified interface that automatically detects
the specification format and uses the appropriate loader.
"""

from ..handler.spec import HandlerSpec, parse_handler_spec
from .file_loader import FileLoader
from .function_loader import FunctionLoader
from .module_loader import ModuleLoader

__all__ = [
    "FileLoader",
    "ModuleLoader",
    "FunctionLoader",
    "HandlerSpec",
    "parse_handler_spec",
]
