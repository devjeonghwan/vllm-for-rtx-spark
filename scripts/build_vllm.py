"""Build the pinned vLLM source and package its native Windows ARM64 wheel."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
_VERSION = runpy.run_path(ROOT / "src/version.py")
_SOURCE_CHANGES = runpy.run_path(ROOT / "scripts/apply_source_changes.py")
MANIFEST = _VERSION["load_manifest"]()
PROJECT_VERSION = _VERSION["PROJECT_VERSION"]
UPSTREAM_VERSION = MANIFEST["vllm"]["version"]
VERSION = _VERSION["wheel_version"]()
render_vllm_version = _VERSION["render_vllm_version"]
PATCH_ROOT = _VERSION["manifest_path"]().parent
UPSTREAM = MANIFEST["vllm"]["repository"]
COMMIT = MANIFEST["vllm"]["commit"]
NATIVE_NAMES = tuple(MANIFEST["build"]["core_extensions"])
safe_relative = _SOURCE_CHANGES["safe_relative"]
payload_files = _SOURCE_CHANGES["payload_files"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def run(argv, **kwargs):
    subprocess.run([str(item) for item in argv], check=True, **kwargs)


def project_file(name):
    path = (ROOT / safe_relative(name)).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Project input escapes the checkout: " + name)
    return path


def copy_inputs(manifest=None, *, source_root=None):
    return _SOURCE_CHANGES["copy_inputs"](
        MANIFEST if manifest is None else manifest, source_root=ROOT if source_root is None else source_root)


def copy_sources(source: Path, manifest=None, *, source_root=None):
    return _SOURCE_CHANGES["copy_sources"](
        source, MANIFEST if manifest is None else manifest, source_root=ROOT if source_root is None else source_root)


def prepare(source: Path, upstream: Path | None = None) -> dict:
    manifest = MANIFEST
    commit, repository = manifest["vllm"]["commit"], manifest["vllm"]["repository"]
    copy_inputs(manifest)
    receipt = source / "rtx-spark-source.json"
    if source.exists():
        if not receipt.is_file():
            raise RuntimeError("Source directory already exists without a completed source receipt")
        verify_source(source)
        return read_json(receipt)
    source.parent.mkdir(parents=True, exist_ok=True)
    if upstream is None:
        run(["git", "init", str(source)])
        run(["git", "-C", str(source), "config", "--local", "core.autocrlf", "false"])
        run(["git", "-C", str(source), "config", "--local", "core.eol", "lf"])
        run(["git", "-C", str(source), "config", "--local", "core.longpaths", "true"])
        run(["git", "-C", str(source), "remote", "add", "origin", repository])
        run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
        run(["git", "-C", str(source), "diff", "--exit-code", "--name-only", "HEAD"])
    else:
        actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError("Local upstream checkout is not the pinned commit")
        # git archive applies checkout line-ending conversion. Keep patch inputs
        # independent of the user's Git autocrlf/eol settings.
        archive = subprocess.check_output([
            "git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
            "-C", str(upstream), "archive", "--format=tar", commit,
        ])
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                safe_relative(member.name)
            tar.extractall(source, filter="data")
        # Isolate git apply from a surrounding checkout. Without a local git
        # root Git can silently ignore patches from an ignored build folder.
        run(["git", "init", "--quiet", str(source)])
        run(["git", "-C", str(source), "config", "--local", "core.autocrlf", "false"])
        run(["git", "-C", str(source), "config", "--local", "core.eol", "lf"])
        run(["git", "-C", str(source), "config", "--local", "core.longpaths", "true"])
    changes = _SOURCE_CHANGES["apply_source_changes"](source, _VERSION["manifest_path"](), source_root=ROOT)
    copied = changes["copied_files"]
    record = {"upstream": repository, "commit": commit, "vllm_version": manifest["vllm"]["version"],
              "version": VERSION, "copy_rules": len(manifest["copy"]), "copied_files": copied,
              "source_prepared": True,
              "GPU_initialized": False}
    write_json(receipt, record)
    verify_source(source)
    return record


def verify_source(source: Path):
    """Check the fixed source identity and required paths, without a hash index."""
    receipt = read_json(source / "rtx-spark-source.json")
    if (receipt.get("source_prepared") is not True or receipt.get("commit") != COMMIT
            or receipt.get("vllm_version") != UPSTREAM_VERSION):
        raise RuntimeError("Source receipt belongs to another vLLM release")
    if (source / "vllm/_rtx_spark").exists():
        raise RuntimeError("Prepared source contains obsolete private copies; use a fresh --source-dir")
    for name in ("CMakeLists.txt", "vllm/__init__.py", "vllm/_version.py", "vllm/engine/arg_utils.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    for rule in MANIFEST["copy"]:
        original = project_file(rule["source"])
        destination = source / safe_relative(rule["destination"])
        if not destination.resolve().is_relative_to(source.resolve()):
            raise ValueError("Prepared source copy escapes its source tree")
        if not destination.exists():
            raise FileNotFoundError(destination)
        stale = "Prepared source copy is stale; use a fresh --source-dir: " + rule["destination"]
        if original.is_dir():
            if not destination.is_dir():
                raise RuntimeError(stale)
            expected = dict((relative, path) for path, relative in payload_files(original))
            actual = dict((relative, path) for path, relative in payload_files(destination))
            if expected.keys() != actual.keys():
                raise RuntimeError(stale)
            if any(path.read_bytes() != actual[relative].read_bytes() for relative, path in expected.items()):
                raise RuntimeError(stale)
        elif not original.is_file():
            raise FileNotFoundError(original)
        elif not destination.is_file() or original.read_bytes() != destination.read_bytes():
            raise RuntimeError(stale)


def check_native(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 64 or raw[:2] != b"MZ":
        raise ValueError("Native extension is not PE: " + path.name)
    offset = struct.unpack_from("<I", raw, 60)[0]
    if raw[offset:offset + 6] != b"PE\0\0\x64\xaa":
        raise ValueError("Native extension must target ARM64: " + path.name)
    return {"path": "vllm/" + path.name, "PE_machine": "AA64"}


def dependencies() -> list[str]:
    from packaging.requirements import Requirement

    manifest = MANIFEST
    result = []
    seen = set()
    for kind in ("runtime", "runtime_vendor"):
        for raw in project_file(manifest["requirements"][kind]).read_text(encoding="utf-8").splitlines():
            line = raw.partition(" #")[0].strip()
            if not line or line.startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.name.replace("_", "-").lower() == "vllm-rtx-spark":
                raise ValueError("The single vLLM wheel must not depend on a separate RTX Spark distribution")
            if line not in seen:
                result.append(line)
                seen.add(line)
    if "torch==" + manifest["platform"]["torch"] not in result:
        raise ValueError("The tested native PyTorch dependency is missing")
    return result


def build_native(source: Path, build: Path, jobs: int):
    manifest = MANIFEST
    if sys.platform != "win32" or platform.machine().upper() not in ("ARM64", "AARCH64"):
        raise RuntimeError("Native core builds require Windows ARM64 Python")
    if ".".join(map(str, sys.version_info[:2])) != manifest["platform"]["python"]:
        raise RuntimeError("Use this release's native Python " + manifest["platform"]["python"])
    import torch
    if torch.__version__ != manifest["platform"]["torch"]:
        raise RuntimeError("Use the tested NVIDIA ARM64 torch==" + manifest["platform"]["torch"])
    if shutil.which("cl") is None or shutil.which("nvcc") is None:
        raise RuntimeError("Enter the native ARM64 MSVC environment and install CUDA " + manifest["platform"]["cuda_toolkit"])
    environment = dict(os.environ, VLLM_TARGET_DEVICE="cuda",
                       TORCH_CUDA_ARCH_LIST=manifest["build"]["torch_cuda_arch_list"],
                       VLLM_BUILD_EXTERNAL_KERNELS="0")
    run(["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_COMPILER=cl", "-DCMAKE_CUDA_FLAGS=-target-dir=arm64",
         "-DCMAKE_CUDA_ARCHITECTURES=" + manifest["build"]["cuda_architectures"],
         "-DVLLM_PYTHON_EXECUTABLE=" + sys.executable,
         "-DVLLM_BUILD_EXTERNAL_KERNELS=OFF", "-DVLLM_BUILD_SM120_CUTLASS=OFF",
         "-DCMAKE_INSTALL_PREFIX=" + str(source), "-DNVCC_THREADS=2"], env=environment)
    # This equivalent alias avoids MSVC's dependent-alias lookup error.
    stride = build / "_deps/cutlass-src/include/cute/stride.hpp"
    if stride.exists():
        old = "    using Seq    = typename Lambda::template seq<Shape>;  // Seq or RSeq"
        new = "    using Seq = conditional_t<is_same<Major, LayoutLeft>::value, tuple_seq<Shape>, tuple_rseq<Shape>>;"
        text = stride.read_text(encoding="utf-8")
        if old in text:
            stride.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    for extension in manifest["build"]["core_extensions"]:
        component = Path(extension).stem
        run(["cmake", "--build", build, "--parallel", str(jobs), "--target", component], env=environment)
        run(["cmake", "--install", build, "--component", component], env=environment)
    return source / "vllm"


def wheel(source: Path, native: Path, output: Path) -> dict:
    manifest = MANIFEST
    verify_source(source)
    version = _VERSION["wheel_version"]()
    commit = manifest["vllm"]["commit"]
    native_names = manifest["build"]["core_extensions"]
    tag = "-".join(manifest["build"][name] for name in ("python_tag", "abi_tag", "platform_tag"))
    native_records = [check_native(native / name) for name in native_names]
    output.mkdir(parents=True, exist_ok=True)
    wheel_name = "vllm-" + version + "-" + tag + ".whl"
    destination = output / wheel_name
    if destination.exists():
        raise FileExistsError("Wheel already exists: " + str(destination))
    dist = "vllm-" + version + ".dist-info"
    records = []
    written = set()
    payload_counts = {}
    def add(archive, path: str, raw: bytes):
        if path in written:
            raise ValueError("Duplicate wheel member: " + path)
        written.add(path)
        info = zipfile.ZipInfo(path, (2026, 10, 27, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, raw)
        token = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        records.append((path, "sha256=" + token, str(len(raw))))
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted((source / "vllm").rglob("*")):
            relative = path.relative_to(source).as_posix()
            if (not path.is_file() or "__pycache__" in path.parts or path.suffix in
                    (".pyc", ".pyd", ".so", ".pdb", ".lib", ".exp")):
                continue
            raw = render_vllm_version(commit) if relative == "vllm/_version.py" else path.read_bytes()
            add(archive, relative, raw)
        for name in native_names:
            add(archive, "vllm/" + name, (native / name).read_bytes())
        for folder, prefix in (("src", "vllm/_rtx_spark"),
                               ("third_party/python", "vllm/_rtx_spark/vendor"),
                               ("assets", "vllm/_rtx_spark/assets")):
            paths = list(payload_files(ROOT / folder))
            if not paths:
                raise RuntimeError("Bundled wheel payload is missing: " + folder)
            for path, relative in paths:
                add(archive, prefix + "/" + relative, path.read_bytes())
            payload_counts[folder] = len(paths)
        add(archive, "vllm/_rtx_spark/manifest.json", _VERSION["manifest_path"]().read_bytes())
        major, minor = map(int, manifest["platform"]["python"].split("."))
        metadata = ["Metadata-Version: 2.4", "Name: vllm", "Version: " + version,
                    "Summary: vLLM " + manifest["vllm"]["version"] + " with RTX Spark Windows ARM64 optimizations",
                    f"Requires-Python: >={major}.{minor},<{major}.{minor + 1}", "License-Expression: Apache-2.0",
                    "License-File: LICENSE", "License-File: THIRD_PARTY_NOTICES.md",
                    "Project-URL: Repository, https://github.com/devjeonghwan/vllm-for-rtx-spark"]
        metadata.extend("Requires-Dist: " + item for item in dependencies())
        add(archive, dist + "/METADATA", ("\n".join(metadata) + "\n\n").encode())
        add(archive, dist + "/WHEEL", ("Wheel-Version: 1.0\nGenerator: vllm-rtx-spark\nRoot-Is-Purelib: false\nTag: " + tag + "\n").encode())
        add(archive, dist + "/entry_points.txt", (
            "[console_scripts]\nvllm = vllm.entrypoints.cli.main:main\n\n"
            "[vllm.general_plugins]\nrtx_spark = vllm._rtx_spark:register\n").encode())
        add(archive, dist + "/top_level.txt", b"vllm\n")
        add(archive, dist + "/licenses/LICENSE", (source / "LICENSE").read_bytes())
        add(archive, dist + "/licenses/THIRD_PARTY_NOTICES.md", (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes())
        record_buffer = io.StringIO(newline="")
        csv.writer(record_buffer, lineterminator="\n").writerows(records + [(dist + "/RECORD", "", "")])
        record_info = zipfile.ZipInfo(dist + "/RECORD", (2026, 10, 27, 0, 0, 0))
        record_info.compress_type = zipfile.ZIP_DEFLATED
        record_info.external_attr = 0o644 << 16
        archive.writestr(record_info, record_buffer.getvalue().encode())
    result = {"wheel": wheel_name, "version": version, "tag": tag, "upstream_commit": commit,
              "native_extensions": native_records, "native_extensions_recompiled_by_this_command": False,
              "bundled_payload_files": payload_counts, "standalone_rtx_spark_distribution_required": False,
              "GPU_initialized": False, "file_count": len(records) + 1}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "wheel"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--upstream-dir", type=Path, help="Optional local checkout of the pinned upstream commit")
    parser.add_argument("--native-dir", type=Path, help="Directory containing both completed ARM64 extensions for wheel mode")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 9))
    args = parser.parse_args()
    source = (args.source_dir or ROOT / ".build" / ("vllm-" + UPSTREAM_VERSION)).resolve()
    if args.command in ("prepare", "build"):
        result = prepare(source, args.upstream_dir)
    if args.command == "build":
        build = (args.build_dir or ROOT / ".build" / ("vllm-native-" + UPSTREAM_VERSION)).resolve()
        native = build_native(source, build, args.jobs)
        result = wheel(source, native, args.output.resolve())
        result["native_extensions_recompiled_by_this_command"] = True
    elif args.command == "wheel":
        if args.native_dir is None:
            parser.error("wheel mode requires --native-dir; it never substitutes precompiled code silently")
        result = wheel(source, args.native_dir.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
