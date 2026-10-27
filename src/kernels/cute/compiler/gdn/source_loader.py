"""Load unchanged original FI CuTe definitions, excluding Torch host dispatch.

Torch names here are immutable dtype/type-annotation tags only. No Torch
module, Tensor constructor, CUDA context or replacement GPU math is provided.
"""
import ast
import os
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(os.environ["SPARK_SOURCE_ROOT"])
SITE = ROOT / "src/kernels/cute/device"


class SourceGraph:
    def __init__(self):
        self.units, self.loaded = {}, {}
        self.torch_types = SimpleNamespace(dtype=object, Tensor=object, device=object,
            int32="torch.int32", int64="torch.int64", float16="torch.float16",
            bfloat16="torch.bfloat16", float32="torch.float32",
            float8_e4m3fn="torch.float8_e4m3fn", float8_e5m2="torch.float8_e5m2")

    @staticmethod
    def import_name(module, node):
        if node.level:
            return importlib.util.resolve_name("." * node.level + (node.module or ""), module.rpartition(".")[0])
        return node.module

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
                            if isinstance(name, ast.Name): definitions[name.id] = node
                elif isinstance(node, ast.Import):
                    for alias in node.names: imports[alias.asname or alias.name.split(".")[0]] = (node, alias)
                elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                    for alias in node.names: imports[alias.asname or alias.name] = (node, alias)
            self.units[module] = dict(path=path, raw=raw, tree=tree, definitions=definitions, imports=imports, selected=set(), references=set())
        return self.units[module]

    def add(self, module, names):
        unit = self.unit(module)
        if set(names) - unit["definitions"].keys():
            raise ValueError("Missing original definition in " + module)
        unit["selected"].update(names)
        while True:
            refs = {node.id for name in unit["selected"] for node in ast.walk(unit["definitions"][name])
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
            expanded = unit["selected"] | (refs & unit["definitions"].keys())
            if expanded == unit["selected"]: break
            unit["selected"] = expanded
        unit["references"] = refs
        for key in refs & unit["imports"].keys():
            node, alias = unit["imports"][key]
            if isinstance(node, ast.ImportFrom):
                resolved = self.import_name(module, node)
                if resolved.startswith("flashinfer."): self.add(resolved, {alias.name})

    def load(self, module):
        if module in self.loaded: return self.loaded[module]
        unit = self.unit(module)
        name = "original_aot_" + module.replace(".", "_")
        result = ModuleType(name)
        result.__file__ = str(unit["path"])
        sys.modules[name] = result
        self.loaded[module] = result
        namespace = vars(result)
        for key in sorted(unit["references"] & unit["imports"].keys()):
            node, alias = unit["imports"][key]
            if key == "torch":
                namespace[key] = self.torch_types
            elif key in ("triton", "tl"):
                raise RuntimeError("Unexpected non-CuTe runtime dependency: " + key)
            elif isinstance(node, ast.ImportFrom):
                resolved = self.import_name(module, node)
                imported = self.load(resolved) if resolved.startswith("flashinfer.") else importlib.import_module(resolved)
                namespace[key] = getattr(imported, alias.name)
            else:
                imported = importlib.import_module(alias.name)
                namespace[key] = imported if alias.asname else importlib.import_module(alias.name.split(".")[0])
        selected = {id(unit["definitions"][key]) for key in unit["selected"]}
        body = [node for node in unit["tree"].body if id(node) in selected or
                isinstance(node, ast.ImportFrom) and node.module == "__future__"]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(unit["path"]), "exec"), namespace)
        return result

    def inventory(self):
        return {module: dict(path=str(unit["path"]), sha256=hashlib.sha256(unit["raw"]).hexdigest(),
                             definitions=sorted(unit["selected"]), original_AST_nodes_modified=False)
                for module, unit in sorted(self.units.items())}
