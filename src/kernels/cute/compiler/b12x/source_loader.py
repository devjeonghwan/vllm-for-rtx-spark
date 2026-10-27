"""Load unchanged original CuTe definitions for build-only compilation.

No Torch/Triton host dispatcher or replacement math is loaded. Original AST
nodes and filenames are kept, including every device method and intrinsic.
"""

import ast
import os
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(os.environ["SPARK_SOURCE_ROOT"])
SITE = ROOT / "src/kernels/cute/device"


class SourceGraph:
    def __init__(self):
        self.units = {}
        self.loaded = {}

    def unit(self, module):
        if module not in self.units:
            path = SITE.joinpath(*module.split(".")).with_suffix(".py")
            raw = path.read_bytes()
            tree = ast.parse(raw, filename=str(path))
            definitions, imports = {}, {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    definitions[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for name in ast.walk(target):
                            if isinstance(name, ast.Name):
                                definitions[name.id] = node
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name.split(".")[0]] = (node, alias)
                elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (node, alias)
            self.units[module] = dict(path=path, raw=raw, tree=tree, definitions=definitions,
                                      imports=imports, selected=set(), references=set())
        return self.units[module]

    def add(self, module, names):
        unit = self.unit(module)
        missing = set(names) - unit["definitions"].keys()
        if missing:
            raise ValueError(f"Original definition missing in {module}: {missing}")
        unit["selected"].update(names)
        while True:
            references = {node.id for name in unit["selected"]
                          for node in ast.walk(unit["definitions"][name])
                          if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
            expanded = unit["selected"] | (references & unit["definitions"].keys())
            if expanded == unit["selected"]:
                break
            unit["selected"] = expanded
        unit["references"] = references
        for name in references & unit["imports"].keys():
            node, alias = unit["imports"][name]
            if isinstance(node, ast.ImportFrom) and node.module.startswith("b12x."):
                self.add(node.module, {alias.name})

    def load(self, module):
        if module in self.loaded:
            return self.loaded[module]
        unit = self.unit(module)
        name = "original_aot_" + module.replace(".", "_")
        loaded = ModuleType(name)
        loaded.__file__ = str(unit["path"])
        namespace = vars(loaded)
        sys.modules[name] = loaded
        self.loaded[module] = loaded
        for key in sorted(unit["references"] & unit["imports"].keys()):
            node, alias = unit["imports"][key]
            if key == "torch":
                # Only original host type annotations need these names. No
                # Torch runtime, Tensor operation, cuda object, or fake GPU API.
                namespace[key] = SimpleNamespace(dtype=object, Tensor=object, device=object)
            elif key in ("triton", "tl", "F"):
                raise RuntimeError("Unexpected non-CuTe runtime dependency: " + key)
            elif isinstance(node, ast.ImportFrom):
                imported = self.load(node.module) if node.module.startswith("b12x.") else importlib.import_module(node.module)
                namespace[key] = getattr(imported, alias.name)
            else:
                imported = importlib.import_module(alias.name)
                namespace[key] = imported if alias.asname else importlib.import_module(alias.name.split(".")[0])
        selected_ids = {id(unit["definitions"][key]) for key in unit["selected"]}
        body = [node for node in unit["tree"].body
                if id(node) in selected_ids or isinstance(node, ast.ImportFrom) and node.module == "__future__"]
        # Use the exact original nodes, locations and filename. No recurrence,
        # intrinsic, tile policy or launch method is rewritten by this loader.
        exec(compile(ast.Module(body=body, type_ignores=[]), str(unit["path"]), "exec"), namespace)
        return loaded

    def inventory(self):
        return {module: {"path": str(unit["path"]),
                         "sha256": hashlib.sha256(unit["raw"]).hexdigest(),
                         "definitions": sorted(unit["selected"]),
                         "original_AST_nodes_modified": False}
                for module, unit in sorted(self.units.items())}
