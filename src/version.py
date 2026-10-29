"""One project release pins one vLLM source and ABI; no native imports."""

import json
from pathlib import Path


PROJECT_VERSION = "0.1.2"


def manifest_path():
    """Locate this release's manifest in a checkout or the installed wheel."""
    source = Path(__file__).resolve().parent
    installed = source / "manifest.json"
    if installed.is_file():
        return installed
    return source.parent / "patches/vllm/manifest.json"


def load_manifest():
    return json.loads(manifest_path().read_text(encoding="utf-8"))


def wheel_version():
    return f"{load_manifest()['vllm']['version']}+rtx_spark.{PROJECT_VERSION}"


def wheel_version_tuple():
    upstream = load_manifest()["vllm"]["version"]
    return tuple(int(part) for part in upstream.split(".")) + (f"rtx_spark.{PROJECT_VERSION}",)


UPSTREAM_VERSION = load_manifest()["vllm"]["version"]
WHEEL_VERSION = wheel_version()
WHEEL_VERSION_TUPLE = wheel_version_tuple()


def render_vllm_version(commit=None, *, upstream_only: bool = False) -> bytes:
    """Render the generated vLLM module, keeping source patches release-neutral."""
    pinned = load_manifest()["vllm"]
    upstream = pinned["version"]
    commit = pinned["commit"] if commit is None else commit
    base_tuple = tuple(int(part) for part in upstream.split("."))
    version = upstream if upstream_only else f"{upstream}+rtx_spark.{PROJECT_VERSION}"
    version_tuple = base_tuple if upstream_only else base_tuple + (f"rtx_spark.{PROJECT_VERSION}",)
    return (
        '"""Version of the pinned Windows ARM64 build."""\n'
        f"__version__ = version = {json.dumps(version)}\n"
        f"__version_tuple__ = version_tuple = {version_tuple!r}\n"
        f"__commit_id__ = commit_id = {json.dumps(commit)}\n"
    ).encode("utf-8")
