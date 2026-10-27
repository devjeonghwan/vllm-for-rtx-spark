"""Kernel pack build interface. Compiler-host details stay in this provider."""
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess

from . import assets

_PROVIDERS = {}


def register_provider(name, provider):
    """Register local compiler metadata; modules replay it in other processes."""
    if name in _PROVIDERS or not callable(provider):
        raise ValueError("Invalid or duplicate compiler provider: " + name)
    _PROVIDERS[name] = provider


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _verify_pack(directory):
    directory = Path(directory).resolve()
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("target") != {
        "cpu": "win_arm64", "gpu": "sm_121a", "sm_count": 48
    }:
        raise ValueError("Unsupported kernel pack schema or target")
    if not isinstance(manifest["files"], list):
        raise ValueError("Kernel pack requires a file list")
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts:
            raise ValueError("Unsafe kernel asset path")
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise ValueError("Built kernel pack is missing: " + name)
    for family in ("b12x", "hc", "skinny", "gdn"):
        metadata = _json(directory / (family + ".json"))
        if metadata["library"] not in manifest["files"]:
            raise ValueError("Built kernel library is not listed in its manifest")
    return manifest


def _install(pack, output, source_root):
    if source_root is None:
        raise ValueError("Installing a built pack requires its source checkout")
    root = Path(source_root).resolve()
    target = assets.data_root().resolve()
    expected = (root / "assets/kernels").resolve()
    backup = (output / "previous-kernels").resolve()
    if target != expected or not target.is_relative_to(root) or not backup.is_relative_to(root):
        raise ValueError("Kernel installation and backup must stay within the source checkout")
    _verify_pack(pack)
    if backup.exists():
        raise RuntimeError("Preserve the existing kernel pack backup")
    target.rename(backup)
    try:
        shutil.copytree(pack, target)
        _verify_pack(target)
    except BaseException:
        if target.exists():
            target.rename(output / "failed-kernel-install")
        backup.rename(target)
        raise
    assets.verify.cache_clear()
    return {"installed_to": str(target), "previous_pack": str(backup)}


def _manifest(destination, *, rebuilt):
    document = assets.metadata("manifest")
    document["files"] = [p.relative_to(destination).as_posix()
                         for p in sorted(destination.rglob("*"))
                         if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    document["prebuilt_reused"] = not rebuilt
    _write(destination / "manifest.json", document)
    return destination / "manifest.json"


def _prebuilt(output, source_root, settings):
    assets.verify()
    destination = output / "kernels"
    shutil.copytree(assets.data_root(), destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {"manifest": str(_manifest(destination, rebuilt=False)), "prebuilt_reused": True,
            "GPU_executed": False, "compiler_required": False}


def _run(argv, env, log):
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run([str(arg) for arg in argv], env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError("Kernel build failed; see " + str(log))


def _native_library(name, objects, root, output, configuration, env):
    request = {key: configuration[key] for key in ("cuda_root", "visual_studio", "llvm_nm")}
    request.update(output=str(output / ("host-" + name)), name=name, objects=[str(p) for p in objects])
    request_path = output / ("link-" + name + ".json")
    _write(request_path, request)
    _run(["pwsh", "-NoProfile", "-File", root / "src/kernels/cute/host/build_host.ps1", "-Request", request_path],
         env, output / ("link-" + name + ".log"))
    result = _json(output / ("host-" + name) / "result.json")
    return Path(result["library"])


def _entry(header):
    text = Path(header).read_text(encoding="utf-8")
    match = re.search(r"void (_mlir_\w+__mlir_ciface_\w+)\(void \*\*args, int32_t num_args\)", text)
    if not match:
        raise RuntimeError("Original generated C entry is missing: " + str(header))
    return match[1]


def _source(output, source_root, settings):
    if source_root is None or settings is None:
        raise ValueError("Source builds require source_root and a compiler settings JSON")
    root, settings_path = Path(source_root).resolve(), Path(settings).resolve()
    configuration = _json(settings_path)
    for name in ("python", "cuda_root", "cute_lib", "visual_studio", "llvm_nm"):
        if not Path(configuration[name]).exists():
            raise ValueError("Compiler configuration path is missing: " + name)
    work = output / "build"
    work.mkdir()
    environment = dict(os.environ)
    environment.update(SPARK_SOURCE_ROOT=str(root), SPARK_BUILD_OUTPUT=str(work),
        SPARK_CUDA_ROOT=configuration["cuda_root"], SPARK_CUTE_LIB=configuration["cute_lib"],
        SPARK_COMPILER_ROOT=str(Path(configuration["python"]).parent),
        CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    compiler = root / "src/kernels/cute/compiler"
    for name in ("b12x/compile_base.py", "b12x/compile_fast.py", "hc/compile.py", "skinny/compile.py", "gdn/compile.py"):
        _run([configuration["python"], "-I", compiler / "launch.py", settings_path, compiler / name, "--execute-build"],
             environment, work / (name.replace("/", "-") + ".log"))
    base, fast = _json(work / "b12x-base/result.json"), _json(work / "b12x-fast/result.json")
    hc, skinny, gdn = (_json(work / family / "result.json") for family in ("hc", "skinny", "gdn"))
    for result in (base, fast, hc, skinny, gdn):
        if not result["status"].startswith("passed_") or result.get("GPU_executed") or result.get("Torch_imported") or result.get("blocked_CUDA_calls"):
            raise RuntimeError("Compiler producer did not finish within the CPU-only build contract")
    destination = output / "kernels"
    shutil.copytree(assets.data_root(), destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    b12x_programs = [*base["programs"], *fast["programs"]]
    objects = [Path(program["artifacts"]["host.o"]["path"]) for program in b12x_programs]
    libraries = {"b12x": _native_library("b12x", objects, root, work, configuration, environment)}
    # Names are emitted by the original HC compiler; each IR sits beside its COFF.
    hc_objects = sorted((work / "hc").glob("*_host.o"))
    libraries["hc"] = _native_library("hc", hc_objects, root, work, configuration, environment)
    libraries["skinny"] = _native_library("skinny", [Path(p["artifacts"]["host.o"]["path"]) for p in skinny["programs"]], root, work, configuration, environment)
    libraries["gdn"] = _native_library("gdn", [Path(p["host_CO_FF"]) for p in gdn["variants"].values()], root, work, configuration, environment)
    for name, path in libraries.items():
        shutil.copyfile(path, destination / "lib" / (name + ".dll"))
    document = _json(destination / "b12x.json")
    document["programs"] = [{**{key: value for key, value in p.items() if key not in ("artifacts", "build_wall_seconds")},
                             "invoke_symbol": _entry(p["artifacts"]["host.h"]["path"])} for p in b12x_programs]
    document["ranges_alpha_one"] = fast["ranges"]
    document.pop("library_sha256", None)
    _write(destination / "b12x.json", document)
    for name in ("hc", "skinny", "gdn"):
        document = _json(destination / (name + ".json"))
        document.pop("library_sha256", None)
        if name == "hc":
            for variant, compiled in zip(document["variants"], hc["kernels"], strict=True):
                header = next(Path(compiled["lowered_IR"]).parent.glob(Path(compiled["lowered_IR"]).stem + "_host.h"))
                variant["invoke_symbol"] = _entry(header)
        elif name == "skinny":
            compiled_by_id = {p["id"]: p for p in skinny["programs"]}
            for variant in document["variants"]:
                identity = re.search(r"original_skinny_(\w+)_cuda_init", variant["init_symbol"])[1]
                variant["invoke_symbol"] = _entry(compiled_by_id[identity]["artifacts"]["host.h"]["path"])
        else:
            for key, compiled in gdn["variants"].items():
                role = "checkpoint" if key.endswith("checkpoint") else "plain"
                document["variants"][role]["invoke_symbol"] = _entry(compiled["exported_headers"][0])
        _write(destination / (name + ".json"), document)
    return {"manifest": str(_manifest(destination, rebuilt=True)), "prebuilt_reused": False,
            "GPU_executed": False, "compiler_provider": "windows-x64-cute-4.8.1"}


_PROVIDERS.update(prebuilt=_prebuilt, source=_source)


def build(output_directory, *, source_root=None, provider="prebuilt", settings=None, install=False):
    """Stage a verified pack, or compile a pack through the selected provider.

    The caller sees a kernel manifest and does not select a host architecture.
    Failed output directories are kept for inspection and never overwritten.
    """
    from ..providers import initialize_providers

    initialize_providers()
    if provider not in _PROVIDERS:
        raise ValueError("Unknown kernel compiler provider: " + provider)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = _PROVIDERS[provider](output, source_root, settings)
    pack = Path(result["manifest"]).parent
    _verify_pack(pack)
    if install:
        result.update(_install(pack, output, source_root))
    return result
