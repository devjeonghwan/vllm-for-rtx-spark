"""Maintainer export of the already-validated vLLM port into versioned patches.

Exported source and manifests contain no workspace paths. The source inputs
are maintainer-only; applying and building the exported patches needs upstream.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
MAX_CHUNK = 16 * 1024 * 1024
SKIP_SUFFIXES = {".pyd", ".so", ".dll", ".lib", ".exp", ".pdb", ".pyc"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def tracked(path):
    return set(subprocess.check_output(["git", "-C", str(path), "ls-files"], text=True).splitlines())


def spark_hooks(name, raw):
    if name == "vllm/__init__.py":
        needle = b"from .version import __version__, __version_tuple__  # isort:skip\n"
        addition = (b"\n# Configure the native platform before importing vLLM consumers.\n"
                    b"from vllm._spark.runtime.bootstrap import initialize as _spark_initialize\n"
                    b"\n_spark_initialize()\n")
        if raw.count(needle) != 1:
            raise RuntimeError("Unexpected vLLM bootstrap anchor")
        raw = raw.replace(needle, needle + addition)
    elif name == "vllm/engine/arg_utils.py":
        text = raw.decode("utf-8")
        tree = ast.parse(text)
        node, = [item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)
                 and item.name == "create_engine_config"]
        if not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[-1], ast.Return):
            raise RuntimeError("Unexpected engine-config construction anchors")
        if not isinstance(node.body[-1].value, ast.Name) or node.body[-1].value.id != "config":
            raise RuntimeError("Unexpected engine-config return")
        lines = text.splitlines(keepends=True)
        lines[node.body[-1].lineno - 1:node.body[-1].lineno - 1] = [
            "        from vllm._spark.runtime import configure_engine\n",
            "\n", "        configure_engine(config)\n", "\n"]
        lines[node.body[0].end_lineno:node.body[0].end_lineno] = [
            "        from vllm._spark.runtime import prepare_engine_args\n",
            "\n", "        prepare_engine_args(self)\n", "\n"]
        raw = "".join(lines).encode()
    elif name == "vllm/_version.py":
        raw = (b'"""Version of the pinned Windows ARM64 build."""\n'
               b'__version__ = version = "0.32.0+spark.1"\n'
               b'__version_tuple__ = version_tuple = (0, 32, 0, "spark.1")\n'
               b'__commit_id__ = commit_id = "5df63a7203cdb9328d824a27b682e77c7b9928be"\n')
    elif name == "vllm/v1/worker/gpu/windows_lifecycle.py":
        raw = raw.replace(b"from porting import recipe_original_ple_windows",
                          b"from vllm._spark.runtime import ple as recipe_original_ple_windows")
    elif name == "vllm/models/qwen4_exp/common/ngram_embedding.py":
        raw = raw.replace(b"from porting.recipe_original_ple_windows import",
                          b"from vllm._spark.runtime.ple import")
    return raw


def patch_text(name, before, after):
    old = before.decode("utf-8").splitlines(keepends=True) if before is not None else []
    new = after.decode("utf-8").splitlines(keepends=True)
    result = [f"diff --git a/{name} b/{name}\n"]
    if before is None:
        result.append("new file mode 100644\n")
    if not old and not new:
        return b""
    for line in difflib.unified_diff(old, new, fromfile="a/" + name if before is not None else "/dev/null",
                                     tofile="b/" + name, n=3):
        result.append(line)
        if not line.endswith("\n"):
            result.extend(("\n", "\\ No newline at end of file\n"))
    return "".join(result).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--native-source-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "patches/vllm/0.32.0")
    args = parser.parse_args()
    base = args.upstream_dir.resolve()
    native = args.native_source_dir.resolve()
    runtime = args.runtime_dir.resolve()
    head = subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise RuntimeError("Use the pinned upstream source as the preimage")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Patch output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    base_files = tracked(base)
    final = {}
    for file in sorted((runtime / "vllm").rglob("*")):
        if (not file.is_file() or "__pycache__" in file.parts or file.suffix in SKIP_SUFFIXES):
            continue
        name = file.relative_to(runtime).as_posix()
        final[name] = spark_hooks(name, normalized(file))
    # Core binaries were built from this native source tree. Do not include
    # generated template .cu files; upstream CMake generates those itself.
    candidates = {name for name in base_files if name == "CMakeLists.txt" or name.startswith(("csrc/", "cmake/"))}
    candidates.update(name for name in tracked(native) if name == "CMakeLists.txt" or name.startswith(("csrc/", "cmake/")))
    candidates.update(("cmake/windows_arm64_cuda.cmake", "fix_cuda_13_align.py", "fix_cutlass_msvc.py",
                       "cmake/patches/windows_cutlass_memory_sm75.patch", "cmake/patches/windows_flash_attn.patch"))
    for name in sorted(candidates):
        file = native / name
        if file.is_file() and file.suffix not in SKIP_SUFFIXES:
            final[name] = normalized(file)
    # _version.py is generated and not part of the upstream git archive.
    final["vllm/_version.py"] = spark_hooks("vllm/_version.py", b"")
    categories = {"native-build": [], "recipe-runtime": [], "bundled-kernels": []}
    files = []
    for name, after in sorted(final.items()):
        before = normalized(base / name) if name in base_files else None
        if before == after:
            continue
        category = ("bundled-kernels" if name.startswith(("vllm/third_party/", "vllm/vllm_flash_attn/cute/"))
                    else "recipe-runtime" if name.startswith("vllm/") else "native-build")
        try:
            diff = patch_text(name, before, after)
        except UnicodeDecodeError as error:
            raise RuntimeError("Unexpected binary source asset: " + name) from error
        home = str(Path.home()).encode()
        if home in after or home.replace(b"\\", b"/") in after:
            raise RuntimeError("Personal workspace dependency in exported source: " + name)
        files.append({"path": name, "category": category,
                      "before_sha256": None if before is None else sha(before),
                      "after_sha256": sha(after), "empty_added": before is None and not after})
        if diff:
            categories[category].append(diff)
    patches = []
    number = 1
    for category, chunks in categories.items():
        buffer = bytearray()
        part = 1
        for chunk in chunks + [b""]:
            if buffer and (len(buffer) + len(chunk) > MAX_CHUNK or not chunk):
                name = f"{number:04d}-{category}-{part}.patch"
                (args.output / name).write_bytes(buffer)
                patches.append({"path": name, "sha256": sha(buffer), "bytes": len(buffer), "category": category})
                buffer.clear()
                number += 1
                part += 1
            buffer.extend(chunk)
    result = {"upstream_repository": "https://github.com/vllm-project/vllm", "upstream_commit": COMMIT,
              "version": "0.32.0+spark.1", "recipe": "myllmbox/qwen38-flash-next-recipe@v5.3.1",
              "recipe_commit": "8d36258ed111619ec40186c28bbb22f1c2c8c23f",
              "Windows_base": "SystemPanic/vllm-windows@a5fd74a0dc5fbb4d3f9a9f27674d1c21fbb6f38c",
              "text_line_endings": "LF", "patches": patches, "files": files,
              "runtime_hooks": ["vllm._spark.runtime.bootstrap.initialize",
                                "vllm._spark.runtime.prepare_engine_args",
                                "vllm._spark.runtime.configure_engine"]}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"patch_count": len(patches), "changed_files": len(files),
                      "patch_bytes": sum(row["bytes"] for row in patches),
                      "categories": {key: sum(row["category"] == key for row in files) for key in categories}}, indent=2))


if __name__ == "__main__":
    main()
