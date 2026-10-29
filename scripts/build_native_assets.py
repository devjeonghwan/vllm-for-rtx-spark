"""Build the native Windows ARM64 host libraries and FlashInfer sampler."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_path(name):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError("Build dependency is missing: " + name)
    return Path(spec.origin).resolve().parent


def native_pe(path):
    raw = Path(path).read_bytes()
    if raw[:2] != b"MZ":
        raise RuntimeError("Expected a Windows DLL: " + str(path))
    offset = struct.unpack_from("<I", raw, 0x3C)[0]
    if raw[offset:offset + 6] != b"PE\0\0\x64\xaa":
        raise RuntimeError("Expected native ARM64 machine code: " + str(path))


def normalize_llguidance_metadata(root):
    """Replace Cargo checkout URIs with package identities, then refresh RECORD."""
    root = Path(root)
    infos = list(root.glob("llguidance-*.dist-info"))
    if len(infos) != 1:
        raise RuntimeError("Expected one llguidance metadata directory")
    info = infos[0]
    sbom = info / "sboms/llguidance_py.cyclonedx.json"
    document = json.loads(sbom.read_text(encoding="utf-8"))
    replacements = {}
    members = {"python_ext": "llguidance_py", "parser": "llguidance",
               "toktrie": "toktrie", "toktrie_hf_tokenizers": "toktrie_hf_tokenizers",
               "toktrie_tiktoken": "toktrie_tiktoken"}
    pattern = re.compile(r"path\+file:///[^#]+/(python_ext|parser|toktrie|toktrie_hf_tokenizers|toktrie_tiktoken)#([^\s#]+)")

    def identity(match):
        member, value = match.groups()
        crate, version = value.split("@", 1) if "@" in value else (members[member], value)
        if crate != members[member]:
            raise RuntimeError("Unexpected llguidance workspace crate")
        result = "pkg:cargo/" + crate + "@" + version
        replacements[member] = result
        return result

    def rewrite(value):
        if isinstance(value, str):
            value = pattern.sub(identity, value)
            return re.sub(r"\?download_url=file://[^#]*", "", value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    clean = json.dumps(rewrite(document), indent=2, ensure_ascii=False) + "\n"
    if "path+file:///" in clean:
        raise RuntimeError("An unrecognized local checkout URI remains in the SBOM")
    sbom.write_text(clean, encoding="utf-8", newline="\n")
    record = info / "RECORD"
    with record.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        path = root / row[0]
        if path == record:
            row[1:] = ["", ""]
            continue
        raw = path.read_bytes()
        token = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        row[1:] = ["sha256=" + token, str(len(raw))]
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return replacements


def build(args):
    if os.name != "nt" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Use native Windows ARM64 Python")
    compiler = shutil.which("cl")
    if not compiler:
        raise RuntimeError("Run scripts/build-native-assets.ps1 to enter the ARM64 build environment")
    output = (args.output or ROOT / "build" / ("native-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(command, label, *, cwd=None):
        commands.append([str(value) for value in command])
        with (output / (label + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=cwd or output)

    produced = {}
    chosen = set(args.components.split(","))
    unknown = chosen - {"ple", "depth", "csf", "sampling", "hc_upmix", "tiktoken", "llguidance"}
    if unknown:
        raise ValueError("Unknown components: " + ",".join(sorted(unknown)))
    for component, source, file, key in (
        ("ple", "ple/native/nvme.cpp", "mbx_ple_nvme.dll", "ple_nvme"),
        ("ple", "ple/native/async.cpp", "mbx_ple_async.dll", "ple_async"),
        ("depth", "depth/mbx_mtp.cpp", "mbx_mtp.dll", "depth"),
    ):
        if component not in chosen:
            continue
        destination = output / file
        command = [compiler, "/nologo", "/std:c++20", "/EHsc", "/O2", "/W4", "/LD",
                   "/Fo" + str(output / (key + ".obj")), str(ROOT / "src/runtime" / source),
                   "/link", "/MACHINE:ARM64", "/INCREMENTAL:NO", "/OUT:" + str(destination)]
        if component == "depth":
            command.insert(6, "/fp:strict")
        run(command, key)
        produced[key] = destination
    cuda = args.cuda_root.resolve()
    if "hc_upmix" in chosen:
        source = ROOT / "src/kernels/hc_upmix/native/hc_upmix.cu"
        obj = output / "hc_upmix.obj"
        flags = ["-std=c++17", "-O3", "-use_fast_math", "-gencode=arch=compute_121,code=sm_121",
                 "-Xcompiler", "/MD,/EHsc,/bigobj,/Zc:__cplusplus,/Zc:preprocessor",
                 "-ccbin", str(Path(compiler).parent)]
        run([str(cuda / "bin/nvcc.exe"), *flags, "-c", str(source), "-o", str(obj)], "hc-upmix-compile")
        destination = output / "rtx_hc_upmix.dll"
        run([shutil.which("link"), "/nologo", "/DLL", "/MACHINE:ARM64", "/INCREMENTAL:NO",
             "/OUT:" + str(destination), "/IMPLIB:" + str(output / "hc_upmix.lib"), str(obj),
             str(cuda / "lib/arm64/cudart.lib")], "hc-upmix-link")
        run([shutil.which("dumpbin"), "/exports", str(destination)], "hc-upmix-exports")
        exports = (output / "hc-upmix-exports.log").read_text(encoding="utf-8")
        names = set(re.findall(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(rtx_hc_\S+)", exports, re.M))
        if names != {"rtx_hc_upmix_abi", "rtx_hc_upmix_bf16"}:
            raise RuntimeError("Unexpected HC native ABI exports")
        produced["hc_upmix"] = destination
    if "csf" in chosen:
        target = output / "csf"
        torch = package_path("torch")
        run(["cmake", "-S", str(ROOT / "src/kernels/csf"), "-B", str(target), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=" + compiler,
             "-DTORCH_ROOT=" + str(torch), "-DCUDA_ROOT=" + str(cuda),
             "-DCUDA_ARM64_LIB=" + str(cuda / "lib/arm64")], "csf-configure")
        run(["cmake", "--build", str(target), "--parallel", str(args.jobs)], "csf-build")
        produced["csf_host"] = target / "mbx_original_csf_native_host.dll"
    if "sampling" in chosen:
        ffi = package_path("tvm_ffi")
        source = ROOT / "third_party/flashinfer-native"
        includes = [source / "cccl/cub", source / "cccl/thrust", source / "cccl/libcudacxx/include",
                    source / "native-include", source / "vendor/csrc", source / "vendor/cutlass/include",
                    source / "vendor/cutlass/tools/util/include", source / "vendor/spdlog/include",
                    ffi / "include", ffi / "3rdparty/dlpack/include", cuda / "include"]
        flags = ["-std=c++17", "-O3", "-DNDEBUG", "-use_fast_math", "-Xfatbin=-compress-all",
                 "--compress-mode=size", "--expt-relaxed-constexpr", "-static-global-template-stub=false",
                 "-gencode=arch=compute_121a,code=sm_121a", "-DFLASHINFER_ENABLE_F16",
                 "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2",
                 "-DFLASHINFER_ENABLE_FP8_E8M0", "-DFLASHINFER_ENABLE_FP4_E2M1", "-Xcompiler",
                 "/MD,/EHsc,/bigobj,/Zc:__cplusplus,/Zc:preprocessor", "-ccbin", str(Path(compiler).parent)]
        flags.extend("-I" + str(path) for path in includes)
        objects = []
        for name in ("sampling.cu", "renorm.cu", "flashinfer_sampling_binding.cu"):
            obj = output / (name + ".obj")
            run([str(cuda / "bin/nvcc.exe"), *flags, "-c", str(source / "vendor/csrc" / name),
                 "-o", str(obj)], "sampling-" + name)
            objects.append(obj)
        destination = output / "flashinfer_sampling.dll"
        run([shutil.which("link"), "/nologo", "/DLL", "/MACHINE:ARM64", "/OUT:" + str(destination),
             *map(str, objects), str(ffi / "lib/tvm_ffi.lib"), str(cuda / "lib/arm64/cudart.lib"),
             str(cuda / "lib/arm64/cuda.lib")], "sampling-link")
        produced["sampling"] = destination
    receipt = {"target": "win_arm64", "GPU_executed": False, "commands": commands, "files": {}}
    for name, path in produced.items():
        native_pe(path)
        receipt["files"][name] = {"file": path.name, "path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    if "tiktoken" in chosen:
        if not shutil.which("cargo"):
            raise RuntimeError("Install native ARM64 Rust/Cargo to build tiktoken")
        wheelhouse = output / "tokenizer"
        wheelhouse.mkdir()
        run([sys.executable, "-m", "pip", "wheel", str(ROOT / "third_party/tiktoken"),
             "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheelhouse)], "tiktoken-build")
        wheels = list(wheelhouse.glob("tiktoken-*-win_arm64.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one native ARM64 tiktoken wheel")
        receipt["tokenizer_wheel"] = {"path": str(wheels[0]), "sha256": sha(wheels[0])}
        if args.install:
            target = ROOT / "third_party/python"
            with zipfile.ZipFile(wheels[0]) as archive:
                for member in archive.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe wheel path")
                    if not member.is_dir():
                        path = target / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(archive.read(member))
    if "llguidance" in chosen:
        if not shutil.which("cargo"):
            raise RuntimeError("Install native ARM64 Rust/Cargo to build llguidance")
        if importlib.util.find_spec("maturin") is None:
            raise RuntimeError("Install maturin>=1,<2 in the build environment")
        wheelhouse = output / "llguidance"
        wheelhouse.mkdir()
        source = ROOT / "third_party/llguidance"
        run([sys.executable, "-m", "maturin", "build", "--manifest-path", str(source / "python_ext/Cargo.toml"),
             "--release", "--locked", "--target", "aarch64-pc-windows-msvc", "--jobs", str(args.jobs),
             "--out", str(wheelhouse), "--interpreter", sys.executable], "llguidance-build", cwd=source)
        wheels = list(wheelhouse.glob("llguidance-*-win_arm64.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one native ARM64 llguidance wheel")
        receipt["llguidance_wheel"] = {"path": str(wheels[0]), "sha256": sha(wheels[0])}
        if args.install:
            target = ROOT / "third_party/python"
            with zipfile.ZipFile(wheels[0]) as archive:
                for member in archive.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe wheel path")
                    if not member.is_dir():
                        path = target / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(archive.read(member))
            normalize_llguidance_metadata(target)
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if args.install:
        target = ROOT / "assets/native"
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        for name, path in produced.items():
            shutil.copyfile(path, target / path.name)
            manifest["files"][name].pop("sha256", None)
            manifest["files"][name].pop("bytes", None)
            manifest["files"][name].update(file=path.name,
                                            origin="Built from this repository's src and third_party trees")
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output / "build.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, default=Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--components", default="ple,depth,csf,sampling,hc_upmix,tiktoken,llguidance")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--install", action="store_true", help="Copy new DLLs into package resources and update its manifest")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
