"""Load the checkout's private integration for CPU tools and tests."""
import importlib.util
from pathlib import Path
import sys


def load():
    if "_spark" not in sys.modules:
        source = Path(__file__).resolve().parents[1] / "src"
        spec = importlib.util.spec_from_file_location(
            "_spark", source / "__init__.py", submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["_spark"]
