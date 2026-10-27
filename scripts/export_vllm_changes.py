"""Export vLLM patches and copied sources into a repository-shaped staging tree."""
from __future__ import annotations

import argparse
import ast
import difflib
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tokenize

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = runpy.run_path(ROOT / "src/version.py")
MANIFEST = VERSIONS["load_manifest"]()
PATCH_ROOT = VERSIONS["manifest_path"]().parent
COMMIT = MANIFEST["vllm"]["commit"]
SKIP_SUFFIXES = {".pyd", ".so", ".dll", ".lib", ".exp", ".pdb", ".pyc"}
HELPER_PACKAGE = "vllm._rtx_spark.recipes.qwen3_8.helpers"
RECIPE_HELPERS = {
    "csf_codec", "loader_core", "mbx_depth", "mbx_draft_head2", "mbx_loader",
    "mbx_marlin_csf", "mbx_opt_quant", "mbx_ple_ckpt", "native_sampling",
    "raw_reader", "vllm_fp8_hybrid",
}


def normalized(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def tracked(path):
    return set(subprocess.check_output(["git", "-C", str(path), "ls-files"], text=True).splitlines())


def contained_path(root, name):
    relative = Path(name)
    root = Path(root).resolve()
    if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
        raise ValueError("Expected a relative export path: " + str(name))
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Export input or output escapes its root: " + str(name))
    return path


def _normalize_hook_tokens(raw):
    """Rename legacy identifiers without touching substrings or source trivia."""
    if b"vllm._spark" not in raw and b"_spark_initialize" not in raw:
        return raw
    text = raw.decode("utf-8")
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME:
            continue
        replacement = None
        if token.string == "_spark_initialize":
            replacement = "_rtx_spark_initialize"
        elif (token.string == "_spark" and index >= 2
              and tokens[index - 1].string == "." and tokens[index - 2].string == "vllm"):
            replacement = "_rtx_spark"
        if replacement is not None:
            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            edits.append((start, end, replacement))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text.encode("utf-8")


def _qualify_recipe_imports(raw):
    """Keep original recipe imports inside their package, including on export."""
    text = raw.decode("utf-8")
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(ast.parse(text)):
        replacement = None
        if isinstance(node, ast.Import) and any(alias.name in RECIPE_HELPERS for alias in node.names):
            if len(node.names) != 1:
                raise RuntimeError("Separate recipe helper imports before export")
            replacement = "from " + HELPER_PACKAGE + " import " + ast.unparse(node)[len("import "):]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in RECIPE_HELPERS:
            node.module = HELPER_PACKAGE + "." + node.module
            replacement = ast.unparse(node)
        if replacement is not None:
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            edits.append((start, end, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.encode("utf-8")


def _has_direct_hook(body, module, imported_name, arguments, *, canonical_name=None,
                     keyword_arguments=(), allow_missing_keywords=False):
    """Admit one complete direct import/call pair; reject partial hook inputs."""
    aliases = [(statement, alias.asname or alias.name) for statement in body
               if isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module == module
               for alias in statement.names if alias.name == imported_name]
    names = {name for _, name in aliases} | {canonical_name or imported_name}
    calls = [statement for statement in body if isinstance(statement, ast.Expr)
             and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name)
             and statement.value.func.id in names]
    if not aliases and not calls:
        return False
    if len(aliases) != 1 or len(calls) != 1:
        raise RuntimeError("Incomplete or duplicate direct hook: " + imported_name)
    call = calls[0].value
    keywords_match = (len(call.keywords) == len(keyword_arguments)
                      and all(item.arg == expected_name and isinstance(item.value, ast.Name)
                              and item.value.id == expected_value
                              for item, (expected_name, expected_value)
                              in zip(call.keywords, keyword_arguments)))
    if allow_missing_keywords and not call.keywords:
        keywords_match = True
    if (aliases[0][1] != call.func.id or not keywords_match
            or len(call.args) != len(arguments)
            or any(not isinstance(arg, ast.Name) or arg.id != expected
                   for arg, expected in zip(call.args, arguments))
            or aliases[0][0].lineno >= calls[0].lineno):
        raise RuntimeError("Unexpected direct hook arguments/order: " + imported_name)
    return True


def rtx_spark_hooks(name, raw):
    if name.endswith(".py"):
        raw = _normalize_hook_tokens(raw)
        raw = _qualify_recipe_imports(raw)
    if name == "vllm/__init__.py":
        tree = ast.parse(raw)
        if _has_direct_hook(tree.body, "vllm._rtx_spark.runtime.bootstrap", "initialize", (),
                            canonical_name="_rtx_spark_initialize"):
            return raw
        needle = b"from .version import __version__, __version_tuple__  # isort:skip\n"
        addition = (b"\n# Configure the native platform before importing vLLM consumers.\n"
                    b"from vllm._rtx_spark.runtime.bootstrap import initialize as _rtx_spark_initialize\n"
                    b"\n_rtx_spark_initialize()\n")
        if raw.count(needle) != 1:
            raise RuntimeError("Unexpected vLLM bootstrap anchor")
        raw = raw.replace(needle, needle + addition)
    elif name == "vllm/engine/arg_utils.py":
        text = raw.decode("utf-8")
        tree = ast.parse(text)
        node, = [item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)
                 and item.name == "create_engine_config"]
        prepared = _has_direct_hook(node.body, "vllm._rtx_spark.runtime", "prepare_engine_args", ("self",))
        configured = _has_direct_hook(
            node.body, "vllm._rtx_spark.runtime", "configure_engine", ("config",),
            keyword_arguments=(("engine_args", "self"),), allow_missing_keywords=True)
        if configured:
            alias, = [item.asname or item.name for statement in node.body
                      if isinstance(statement, ast.ImportFrom) and statement.level == 0
                      and statement.module == "vllm._rtx_spark.runtime"
                      for item in statement.names if item.name == "configure_engine"]
            call, = [statement.value for statement in node.body if isinstance(statement, ast.Expr)
                     and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name)
                     and statement.value.func.id == alias]
            if not call.keywords:
                offsets = [0]
                for line in text.splitlines(keepends=True):
                    offsets.append(offsets[-1] + len(line))
                start = offsets[call.lineno - 1] + call.col_offset
                end = offsets[call.end_lineno - 1] + call.end_col_offset
                call.keywords = [ast.keyword(arg="engine_args", value=ast.Name(id="self", ctx=ast.Load()))]
                text = text[:start] + ast.unparse(call) + text[end:]
                raw = text.encode("utf-8")
                node, = [item for item in ast.walk(ast.parse(text)) if isinstance(item, ast.FunctionDef)
                         and item.name == "create_engine_config"]
        if prepared and configured:
            return raw
        if ast.get_docstring(node) is None or not isinstance(node.body[-1], ast.Return):
            raise RuntimeError("Unexpected engine-config construction anchors")
        if not isinstance(node.body[-1].value, ast.Name) or node.body[-1].value.id != "config":
            raise RuntimeError("Unexpected engine-config return")
        lines = text.splitlines(keepends=True)
        if not configured:
            lines[node.body[-1].lineno - 1:node.body[-1].lineno - 1] = [
                "        from vllm._rtx_spark.runtime import configure_engine\n",
                "\n", "        configure_engine(config, engine_args=self)\n", "\n"]
        if not prepared:
            lines[node.body[0].end_lineno:node.body[0].end_lineno] = [
                "        from vllm._rtx_spark.runtime import prepare_engine_args\n",
                "\n", "        prepare_engine_args(self)\n", "\n"]
        raw = "".join(lines).encode()
    elif name == "vllm/_version.py":
        raw = VERSIONS["render_vllm_version"](upstream_only=True)
    elif name == "vllm/v1/worker/gpu/windows_lifecycle.py":
        raw = raw.replace(b"from porting import recipe_original_ple_windows",
                          b"from vllm._rtx_spark.runtime import ple as recipe_original_ple_windows")
    elif name == "vllm/models/qwen4_exp/common/ngram_embedding.py":
        raw = raw.replace(b"from porting.recipe_original_ple_windows import",
                          b"from vllm._rtx_spark.runtime.ple import")
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


def patch_owners():
    """Keep the existing four role boundaries without another file inventory."""
    owners = {}
    for name in MANIFEST["patches"]:
        for line in contained_path(PATCH_ROOT, name).read_text(encoding="utf-8").splitlines():
            if line.startswith("diff --git a/"):
                owners[line.split(" b/", 1)[1]] = name
    return owners


def copied_path(name):
    return any(name == rule["destination"] or name.startswith(rule["destination"] + "/")
               for rule in MANIFEST["copy"])


def is_patch_source(name):
    """Maintained private modules and declared copies are not patch payloads."""
    private = "vllm/_rtx_spark"
    return name != private and not name.startswith(private + "/") and not copied_path(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--native-source-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New staging root mirroring repository-relative paths")
    args = parser.parse_args()
    base, native, runtime = (value.resolve() for value in
                             (args.upstream_dir, args.native_source_dir, args.runtime_dir))
    output = args.output.resolve()
    head = subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise RuntimeError("Use this release's pinned upstream commit as the preimage")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Export output must be new or empty")
    base_files = tracked(base)
    final = {}
    for file in sorted((runtime / "vllm").rglob("*")):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix in SKIP_SUFFIXES:
            continue
        name = file.relative_to(runtime).as_posix()
        if is_patch_source(name):
            final[name] = rtx_spark_hooks(name, normalized(file))
    candidates = {name for name in base_files | tracked(native)
                  if name == "CMakeLists.txt" or name.startswith(("csrc/", "cmake/"))}
    candidates.update(("cmake/windows_arm64_cuda.cmake", "fix_cuda_13_align.py", "fix_cutlass_msvc.py",
                       "cmake/patches/windows_cutlass_memory_sm75.patch", "cmake/patches/windows_flash_attn.patch"))
    for name in sorted(candidates):
        file = native / name
        if file.is_file() and file.suffix not in SKIP_SUFFIXES and is_patch_source(name):
            final[name] = normalized(file)
    final["vllm/_version.py"] = rtx_spark_hooks("vllm/_version.py", b"")
    owners = patch_owners()
    groups = {name: [] for name in MANIFEST["patches"]}
    for name, after in sorted(final.items()):
        before = normalized(base / name) if name in base_files else None
        if before == after:
            continue
        owner = owners.get(name)
        if owner is None:
            role = "recipe-runtime" if name.startswith("vllm/") else "native-build"
            owner = next(patch for patch in groups if role in patch)
        home = str(Path.home()).encode()
        if home in after or home.replace(b"\\", b"/") in after:
            raise RuntimeError("Personal workspace dependency in exported source: " + name)
        groups[owner].append(patch_text(name, before, after))
    if any(not chunks for chunks in groups.values()):
        raise RuntimeError("A release patch role has no source changes; review the fixed patch list")
    copied_sources = []
    for rule in MANIFEST["copy"]:
        repository_source = contained_path(ROOT, rule["source"])
        original = contained_path(runtime, rule["destination"])
        if not original.exists():
            original = repository_source
        if not original.exists():
            raise FileNotFoundError(original)
        copied_sources.append((original, contained_path(output, rule["source"])))
    patch_output = contained_path(output, PATCH_ROOT.relative_to(ROOT))
    patch_output.mkdir(parents=True, exist_ok=True)
    for name, chunks in groups.items():
        contained_path(patch_output, name).write_bytes(b"".join(chunks))
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyd", "*.so", "*.so.*", "*.dll", "*.lib", "*.exp", "*.pdb")
    for original, destination in copied_sources:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(original, destination, ignore=ignore)
        else:
            shutil.copyfile(original, destination)
    (patch_output / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"patches": MANIFEST["patches"], "copy_rules": len(MANIFEST["copy"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
