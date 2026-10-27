"""Locate the installed kernel pack; no compiler or CUDA import is needed."""
from functools import lru_cache
import json
from pathlib import Path, PureWindowsPath

from ...runtime import resources_root


def data_root():
    return resources_root() / "kernels"


def resolve(relative):
    name = Path(relative)
    if name.is_absolute() or PureWindowsPath(str(relative)).drive or ".." in name.parts:
        raise ValueError("Kernel assets require a package-relative path")
    target = (data_root() / name).resolve()
    if not target.is_relative_to(data_root().resolve()):
        raise ValueError("Kernel asset escapes its package")
    return target


def metadata(family):
    if family not in {"b12x", "hc", "skinny", "gdn", "manifest"}:
        raise ValueError("Unknown kernel metadata: " + family)
    return json.loads(resolve(family + ".json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def verify():
    manifest = metadata("manifest")
    if manifest["schema_version"] != 1 or manifest["target"] != {
        "cpu": "win_arm64", "gpu": "sm_121a", "sm_count": 48
    }:
        raise RuntimeError("Unsupported kernel pack target or schema")
    if not isinstance(manifest["files"], list):
        raise RuntimeError("Kernel pack requires a file list")
    for name in manifest["files"]:
        if not resolve(name).is_file():
            raise RuntimeError("Kernel asset is missing: " + name)
    return manifest
