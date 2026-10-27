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
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "vllm" / "0.32.0"
UPSTREAM = "https://github.com/vllm-project/vllm.git"
COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
VERSION = "0.32.0+spark.1"
NATIVE_NAMES = ("_C_stable_libtorch.pyd", "_moe_C_stable_libtorch.pyd")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def run(argv, **kwargs):
    subprocess.run([str(item) for item in argv], check=True, **kwargs)


def safe_relative(name: str) -> Path:
    value = Path(name)
    if value.is_absolute() or value.drive or ".." in value.parts:
        raise ValueError("Unsafe path in source manifest: " + name)
    return value


def normalized(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def prepare(source: Path, upstream: Path | None = None) -> dict:
    manifest = read_json(PATCH_ROOT / "manifest.json")
    if manifest["upstream_commit"] != COMMIT:
        raise RuntimeError("Source manifest targets another vLLM revision")
    receipt = source / "spark-source.json"
    if source.exists():
        if not receipt.is_file():
            raise RuntimeError("Source directory already exists without a completed source receipt")
        verify_source(source, manifest)
        return read_json(receipt)
    source.parent.mkdir(parents=True, exist_ok=True)
    if upstream is None:
        run(["git", "init", str(source)])
        run(["git", "-C", str(source), "config", "--local", "core.autocrlf", "false"])
        run(["git", "-C", str(source), "config", "--local", "core.eol", "lf"])
        run(["git", "-C", str(source), "config", "--local", "core.longpaths", "true"])
        run(["git", "-C", str(source), "remote", "add", "origin", UPSTREAM])
        run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", COMMIT])
        run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
        run(["git", "-C", str(source), "diff", "--exit-code", "--name-only", "HEAD"])
    else:
        actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT:
            raise RuntimeError("Local upstream checkout is not the pinned commit")
        archive = subprocess.check_output(["git", "-C", str(upstream), "archive", "--format=tar", COMMIT])
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
    for row in manifest["files"]:
        path = source / safe_relative(row["path"])
        previous = row["before_sha256"]
        if previous is None:
            if path.exists():
                raise RuntimeError("Unexpected existing added file: " + row["path"])
        else:
            if not path.is_file() or sha(normalized(path)) != previous:
                raise RuntimeError("Upstream preimage differs: " + row["path"])
            path.write_bytes(normalized(path))
    for row in manifest["patches"]:
        patch = PATCH_ROOT / safe_relative(row["path"])
        if sha(patch.read_bytes()) != row["sha256"]:
            raise RuntimeError("Patch hash differs: " + row["path"])
        run(["git", "apply", "--check", "--whitespace=nowarn", str(patch)], cwd=source)
        run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=source)
    for row in manifest["files"]:
        if row.get("empty_added"):
            path = source / safe_relative(row["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=False)
    verify_source(source, manifest)
    record = {"upstream": UPSTREAM, "commit": COMMIT, "version": VERSION,
              "patch_manifest_sha256": sha((PATCH_ROOT / "manifest.json").read_bytes()),
              "changed_files": len(manifest["files"]), "source_verified": True,
              "GPU_initialized": False}
    write_json(receipt, record)
    return record


def verify_source(source: Path, manifest: dict | None = None):
    manifest = manifest or read_json(PATCH_ROOT / "manifest.json")
    for row in manifest["files"]:
        path = source / safe_relative(row["path"])
        if not path.is_file() or sha(normalized(path)) != row["after_sha256"]:
            raise RuntimeError("Patched source differs: " + row["path"])


def check_native(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 64 or raw[:2] != b"MZ":
        raise ValueError("Native extension is not PE: " + path.name)
    offset = struct.unpack_from("<I", raw, 60)[0]
    if raw[offset:offset + 6] != b"PE\0\0\x64\xaa":
        raise ValueError("Native extension must target ARM64: " + path.name)
    return {"path": "vllm/" + path.name, "bytes": len(raw), "sha256": sha(raw), "PE_machine": "AA64"}


def dependencies() -> list[str]:
    from packaging.requirements import Requirement

    result = []
    seen = set()
    for name in ("runtime.txt", "runtime-vendor.txt"):
        for raw in (ROOT / "requirements" / name).read_text(encoding="utf-8").splitlines():
            line = raw.partition(" #")[0].strip()
            if not line or line.startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.name.replace("_", "-").lower() == "vllm-rtx-spark":
                raise ValueError("The single vLLM wheel must not depend on a separate Spark distribution")
            if line not in seen:
                result.append(line)
                seen.add(line)
    if "torch==2.14.0+cu134" not in result:
        raise ValueError("The tested native PyTorch dependency is missing")
    return result


def payload_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or path.suffix in (".pyc", ".pyo")
                or any(part in ("__pycache__", ".git", ".pytest_cache", ".ruff_cache")
                       or part.endswith(".egg-info") for part in relative.parts)):
            continue
        yield path, relative.as_posix()


def build_native(source: Path, build: Path, jobs: int):
    if sys.platform != "win32" or platform.machine().upper() not in ("ARM64", "AARCH64"):
        raise RuntimeError("Native core builds require Windows ARM64 Python")
    import torch
    if torch.__version__ != "2.14.0+cu134":
        raise RuntimeError("Use the tested NVIDIA ARM64 torch==2.14.0+cu134")
    if shutil.which("cl") is None or shutil.which("nvcc") is None:
        raise RuntimeError("Enter the native ARM64 MSVC environment and install CUDA 13.4 first")
    environment = dict(os.environ, VLLM_TARGET_DEVICE="cuda", TORCH_CUDA_ARCH_LIST="12.1a",
                       VLLM_BUILD_EXTERNAL_KERNELS="0")
    run(["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CXX_COMPILER=cl", "-DCMAKE_CUDA_FLAGS=-target-dir=arm64",
         "-DCMAKE_CUDA_ARCHITECTURES=121a-real", "-DVLLM_PYTHON_EXECUTABLE=" + sys.executable,
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
    for target in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
        run(["cmake", "--build", build, "--parallel", str(jobs), "--target", target], env=environment)
        run(["cmake", "--install", build, "--component", target], env=environment)
    return source / "vllm"


def wheel(source: Path, native: Path, output: Path) -> dict:
    verify_source(source)
    native_records = [check_native(native / name) for name in NATIVE_NAMES]
    output.mkdir(parents=True, exist_ok=True)
    wheel_name = "vllm-" + VERSION + "-cp313-cp313-win_arm64.whl"
    destination = output / wheel_name
    if destination.exists():
        raise FileExistsError("Wheel already exists: " + str(destination))
    dist = "vllm-" + VERSION + ".dist-info"
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
            add(archive, relative, path.read_bytes())
        for name in NATIVE_NAMES:
            add(archive, "vllm/" + name, (native / name).read_bytes())
        for folder, prefix in (("src", "vllm/_spark"),
                               ("third_party/python", "vllm/_spark/vendor"),
                               ("assets", "vllm/_spark/assets")):
            paths = list(payload_files(ROOT / folder))
            if not paths:
                raise RuntimeError("Bundled wheel payload is missing: " + folder)
            for path, relative in paths:
                add(archive, prefix + "/" + relative, path.read_bytes())
            payload_counts[folder] = len(paths)
        metadata = ["Metadata-Version: 2.4", "Name: vllm", "Version: " + VERSION,
                    "Summary: vLLM 0.32 with RTX Spark Windows ARM64 optimizations",
                    "Requires-Python: >=3.13,<3.14", "License-Expression: Apache-2.0",
                    "License-File: LICENSE", "License-File: THIRD_PARTY_NOTICES.md",
                    "Project-URL: Repository, https://github.com/devjeonghwan/vllm-for-rtx-spark"]
        metadata.extend("Requires-Dist: " + item for item in dependencies())
        add(archive, dist + "/METADATA", ("\n".join(metadata) + "\n\n").encode())
        add(archive, dist + "/WHEEL", b"Wheel-Version: 1.0\nGenerator: vllm-rtx-spark\nRoot-Is-Purelib: false\nTag: cp313-cp313-win_arm64\n")
        add(archive, dist + "/entry_points.txt", b"[console_scripts]\nvllm = vllm.entrypoints.cli.main:main\n")
        add(archive, dist + "/top_level.txt", b"vllm\n")
        add(archive, dist + "/licenses/LICENSE", (source / "LICENSE").read_bytes())
        add(archive, dist + "/licenses/THIRD_PARTY_NOTICES.md", (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes())
        record_buffer = io.StringIO(newline="")
        csv.writer(record_buffer, lineterminator="\n").writerows(records + [(dist + "/RECORD", "", "")])
        record_info = zipfile.ZipInfo(dist + "/RECORD", (2026, 10, 27, 0, 0, 0))
        record_info.compress_type = zipfile.ZIP_DEFLATED
        record_info.external_attr = 0o644 << 16
        archive.writestr(record_info, record_buffer.getvalue().encode())
    result = {"wheel": wheel_name, "sha256": sha(destination.read_bytes()), "bytes": destination.stat().st_size,
              "version": VERSION, "tag": "cp313-cp313-win_arm64", "upstream_commit": COMMIT,
              "patch_manifest_sha256": sha((PATCH_ROOT / "manifest.json").read_bytes()),
              "native_extensions": native_records, "native_extensions_recompiled_by_this_command": False,
              "bundled_payload_files": payload_counts, "standalone_spark_distribution_required": False,
              "GPU_initialized": False, "file_count": len(records) + 1}
    write_json(output / (wheel_name + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "wheel"))
    parser.add_argument("--source-dir", type=Path, default=ROOT / ".build/vllm")
    parser.add_argument("--upstream-dir", type=Path, help="Optional local checkout of the pinned upstream commit")
    parser.add_argument("--native-dir", type=Path, help="Directory containing both completed ARM64 extensions for wheel mode")
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".build/vllm-native")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 9))
    args = parser.parse_args()
    source = args.source_dir.resolve()
    if args.command in ("prepare", "build"):
        result = prepare(source, args.upstream_dir)
    if args.command == "build":
        native = build_native(source, args.build_dir.resolve(), args.jobs)
        result = wheel(source, native, args.output.resolve())
        result["native_extensions_recompiled_by_this_command"] = True
        write_json(args.output.resolve() / (result["wheel"] + ".json"), result)
    elif args.command == "wheel":
        if args.native_dir is None:
            parser.error("wheel mode requires --native-dir; it never substitutes precompiled code silently")
        result = wheel(source, args.native_dir.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
