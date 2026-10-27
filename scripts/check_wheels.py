"""Check the standalone vLLM release wheel without loading native code."""
import argparse
import ast
import base64
import configparser
import csv
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import runpy
import stat
import struct
import zipfile

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
_VERSION = runpy.run_path(ROOT / "src/version.py")
MANIFEST = _VERSION["load_manifest"]()
EXPECTED_TAG = "-".join(MANIFEST["build"][name] for name in ("python_tag", "abi_tag", "platform_tag"))
EXPECTED_TORCH = "torch==" + MANIFEST["platform"]["torch"]
RTX_SPARK = "vllm/_rtx_spark/"
PLUGIN = "vllm._rtx_spark:register"
ADAPTER = "vllm._rtx_spark.runtime.vllm.adapter"
CORE_REQUIRED = {
    "vllm/__init__.py", "vllm/entrypoints/cli/main.py",
    "vllm/entrypoints/__init__.py", "vllm/entrypoints/cli/__init__.py",
    "vllm/engine/__init__.py", "vllm/engine/arg_utils.py",
    "vllm/_C_stable_libtorch.pyd", "vllm/_moe_C_stable_libtorch.pyd",
}
RTX_SPARK_REQUIRED = {
    RTX_SPARK + name for name in (
        "__init__.py", "version.py", "manifest.json", "plugin.py", "optimizations/__init__.py",
        "runtime/__init__.py", "runtime/bootstrap.py", "runtime/check.py", "runtime/vllm/__init__.py",
        "runtime/ple/__init__.py", "kernels/__init__.py", "kernels/providers.py", "kernels/csf/__init__.py",
        "kernels/cute/__init__.py", "kernels/cute/assets.py", "kernels/cute/b12x_aot.py",
        "kernels/cute/coff_runtime.py", "kernels/cute/hc_aot.py", "kernels/cute/builder.py",
        "kernels/cute/skinny_aot.py", "kernels/cute/gdn_aot.py",
        "kernels/cute/device/b12x/__init__.py", "kernels/cute/device/b12x/_lib/__init__.py",
        "kernels/cute/device/b12x/policy/__init__.py",
        "kernels/cute/device/b12x/policy/_profiles/__init__.py",
        "kernels/cute/device/b12x/policy/_profiles/data/nvidia.gb10.48sm.json.gz",
        "assets/native/manifest.json", "assets/kernels/manifest.json",
        "recipes/__init__.py", "recipes/qwen3_8/__init__.py", "recipes/qwen3_8/helpers/__init__.py",
        "recipes/qwen3_8/helpers/mbx_marlin_csf.py", "recipes/qwen3_8/helpers/mbx_loader.py",
        "recipes/qwen3_8/helpers/mbx_depth.py", "recipes/qwen3_8/helpers/native_sampling.py",
        "recipes/qwen3_8/helpers/mbx_draft_head2.py", "recipes/qwen3_8/helpers/mbx_ple_ckpt.py",
        "recipes/qwen3_8/helpers/mbx_opt_quant.py", "recipes/qwen3_8/helpers/loader_core.py",
        "recipes/qwen3_8/helpers/csf_codec.py", "recipes/qwen3_8/helpers/vllm_fp8_hybrid.py",
        "recipes/qwen3_8/helpers/raw_reader.py", "assets/recipes/qwen3_8/mtp-depth.json",
        "assets/recipes/qwen3_8/qwen3.8-head2.basis.pt",
        "vendor/openai_harmony/__init__.py", "vendor/openai_harmony/openai_harmony.pyd",
        "vendor/llguidance/__init__.py", "vendor/llguidance/_lib.pyd",
    )
}
NATIVE_REQUIRED = {"csf_host", "csf_cubin", "ple_nvme", "ple_async", "depth", "sampling"}


class WheelError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise WheelError(message)


def safe_name(name):
    path = PurePosixPath(name)
    check(bool(name) and name.rstrip("/") == path.as_posix()
          and not path.is_absolute() and not PureWindowsPath(name).drive
          and ".." not in path.parts and "\\" not in name, "Unsafe wheel path: " + name)
    return name


def _native_architecture(archive, info):
    with archive.open(info) as handle:
        header = handle.read(64)
        if info.filename.lower().endswith((".dll", ".pyd")):
            check(len(header) >= 64 and header[:2] == b"MZ", "Invalid native PE: " + info.filename)
            offset = struct.unpack_from("<I", header, 60)[0]
            check(offset + 6 <= info.file_size, "Invalid PE offset: " + info.filename)
            handle.seek(offset)
            check(handle.read(6) == b"PE\0\0\x64\xaa", "Native binary is not ARM64: " + info.filename)
            return "ARM64"
        if header.startswith(b"\x7fELF"):
            check(len(header) >= 20 and header[5] == 1, "Invalid ELF header: " + info.filename)
            machine = struct.unpack_from("<H", header, 18)[0]
            check(machine == 190, "Linux host ELF in wheel: " + info.filename)
            return "CUDA"
    return None


def _asset(archive, names, prefix, name):
    path = safe_name(prefix + safe_name(name))
    check(path in names, "Missing manifest asset: " + path)
    return archive.read(path)


def _asset_manifests(archive, names):
    native_prefix, kernel_prefix = RTX_SPARK + "assets/native/", RTX_SPARK + "assets/kernels/"
    native = json.loads(archive.read(native_prefix + "manifest.json"))
    check(native["format_version"] == 1 and native["host"] == "win_arm64", "Wrong native manifest target")
    check(isinstance(native["files"], dict), "Invalid native file mapping")
    check(NATIVE_REQUIRED <= native["files"].keys(), "Missing native CSF/PLE/MTP/sampling component")
    for key, record in native["files"].items():
        _asset(archive, names, native_prefix, record["file"])
        info = archive.getinfo(native_prefix + record["file"])
        architecture = _native_architecture(archive, info)
        check(architecture == ("CUDA" if key == "csf_cubin" else "ARM64"),
              "Invalid native component architecture: " + key)
    kernel = json.loads(archive.read(kernel_prefix + "manifest.json"))
    check(kernel["schema_version"] == 1 and kernel["target"]["cpu"] == "win_arm64", "Wrong kernel manifest target")
    check(isinstance(kernel["files"], list) and all(isinstance(name, str) for name in kernel["files"]),
          "Kernel manifest files must be a list")
    check(len(kernel["files"]) == len(set(kernel["files"])), "Duplicate kernel manifest path")
    check(not any(name.startswith("sources/") for name in kernel["files"]),
          "Kernel sources belong in the private package, not duplicated assets")
    for name in kernel["files"]:
        _asset(archive, names, kernel_prefix, name)
    for family in ("b12x", "hc", "skinny", "gdn"):
        name = family + ".json"
        check(name in kernel["files"], "Missing kernel family metadata: " + family)
        metadata = json.loads(archive.read(kernel_prefix + name))
        check(metadata["library"] in kernel["files"], "Kernel library is not listed in its manifest")
        _asset(archive, names, kernel_prefix, metadata["library"])
    return {"native_assets": len(native["files"]), "kernel_assets": len(kernel["files"])}


def _called_import(archive, member, module, function):
    tree = ast.parse(archive.read(member), filename=member)
    aliases = {alias.asname or alias.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module == module and node.level == 0
               for alias in node.names if alias.name == function}
    check(aliases and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id in aliases for node in ast.walk(tree)),
          f"Missing RTX Spark hook {module}.{function} in {member}")


def _integration(archive, names, prefix, requirements, expected_torch):
    check(not any(name.startswith("vllm/_spark/") for name in names),
          "Legacy internal namespace in standalone wheel")
    check(not any(re.search(r"(?:^|/)vllm[-_]rtx[-_]spark(?:/|[-.])", name, re.IGNORECASE)
                  for name in names), "Legacy companion package path in standalone wheel")
    for requirement in requirements:
        dependency = re.match(r"\s*([a-zA-Z0-9_.-]+)", requirement)
        check(dependency is not None, "Invalid Requires-Dist entry")
        check(re.sub(r"[-_.]+", "-", dependency[1].lower()) != "vllm-rtx-spark",
              "Standalone vLLM must not depend on the legacy companion package")
    check(expected_torch in {item.replace(" ", "") for item in requirements},
          "Native wheel does not pin the tested Torch/CUDA ABI: vllm")
    entrypoint_path = prefix + "entry_points.txt"
    check(entrypoint_path in names, "Missing vllm console entrypoint")
    entries = configparser.ConfigParser(interpolation=None)
    entries.read_string(archive.read(entrypoint_path).decode("utf-8"))
    check(entries.get("console_scripts", "vllm", fallback="").strip()
          == "vllm.entrypoints.cli.main:main", "Missing or incorrect vllm console entrypoint")
    check(entries.get("vllm.general_plugins", "rtx_spark", fallback="").strip() == PLUGIN,
          "Missing or incorrect RTX Spark plugin registration")
    for section in entries.sections():
        for key, value in entries.items(section):
            check(not re.search(r"vllm[-_]rtx[-_]spark", value, re.IGNORECASE)
                  and "vllm._spark" not in value,
                  "Legacy companion plugin or entrypoint in standalone wheel")
    _called_import(archive, "vllm/__init__.py", "vllm._rtx_spark.runtime.bootstrap", "initialize")
    for function in ("prepare_engine_args", "configure_engine"):
        _called_import(archive, "vllm/engine/arg_utils.py", "vllm._rtx_spark.runtime", function)


def _release_manifest(archive, names, version):
    member = RTX_SPARK + "manifest.json"
    check(member in names, "Missing private release manifest")
    manifest = json.loads(archive.read(member))
    check(manifest == MANIFEST, "Private release manifest differs from this release")
    check(Version(version) == Version(_VERSION["wheel_version"]()),
          "Wheel version does not match this release")
    adapter = ADAPTER.replace(".", "/") + ".py"
    check(adapter in names, "Missing vLLM adapter: " + adapter)
    return manifest


def validate_wheel(path, *, expected_tag=None):
    """Verify container integrity, installed layout, native ABI and required files."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            check(len(names) == len(set(names)), "Duplicate ZIP member")
            for info in infos:
                safe_name(info.filename)
                check(not stat.S_ISLNK(info.external_attr >> 16), "Wheel contains a symlink")
            check(archive.testzip() is None, "Wheel ZIP CRC check failed")
            metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
            check(len(metadata_paths) == 1, "Expected exactly one distribution metadata directory")
            prefix = metadata_paths[0].rsplit("/", 1)[0] + "/"
            check(prefix + "WHEEL" in names and prefix + "RECORD" in names, "Wheel metadata is incomplete")
            metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
            wheel = Parser().parsestr(archive.read(prefix + "WHEEL").decode("utf-8"))
            check(all(metadata[key] for key in ("Name", "Version", "Requires-Python")), "Incomplete distribution metadata")
            name = re.sub(r"[-_.]+", "-", metadata["Name"].lower())
            check(name == "vllm", "Expected the standalone vllm distribution: " + name)
            manifest = _release_manifest(archive, names, metadata["Version"])
            python = manifest["platform"]["python"]
            python_tag = manifest["build"]["python_tag"]
            expected_tag = expected_tag or "-".join(
                manifest["build"][key] for key in ("python_tag", "abi_tag", "platform_tag"))
            filename_parts = path.stem.split("-")
            check(len(filename_parts) in (5, 6) and path.suffix == ".whl", "Invalid wheel filename")
            check(re.sub(r"[-_.]+", "-", filename_parts[0].lower()) == name
                  and filename_parts[1] == metadata["Version"], "Filename and distribution metadata differ")
            check(set(wheel.get_all("Tag", [])) == {expected_tag}, "Wrong wheel ABI/platform tag")
            check("-".join(path.stem.split("-")[-3:]) == expected_tag, "Filename and wheel ABI/platform differ")
            check(wheel["Root-Is-Purelib"] == "false", "Native wheel is incorrectly marked pure Python")
            python_requires = {part.strip() for part in metadata["Requires-Python"].split(",")}
            major, minor = map(int, python.split("."))
            check(python_requires == {">=" + python, f"<{major}.{minor + 1}"},
                  "Expected the target CPython range")
            rows = list(csv.reader(io.StringIO(archive.read(prefix + "RECORD").decode("utf-8"))))
            records = {}
            for row in rows:
                check(len(row) == 3, "Invalid RECORD row")
                member, digest, size = row
                safe_name(member)
                check(member not in records, "Duplicate RECORD member: " + member)
                records[member] = (digest, size)
            signatures = {prefix + "RECORD.jws", prefix + "RECORD.p7s"}
            check(set(records) <= set(names), "RECORD names a missing ZIP member")
            check(set(names) - signatures <= records.keys(), "ZIP member is absent from RECORD")
            check(records.get(prefix + "RECORD") == ("", ""), "RECORD must contain its unhashed self entry")
            native = []
            for info in infos:
                member = info.filename
                if info.is_dir():
                    continue
                check(not member.lower().endswith((".pyc", ".pyo")) and "__pycache__" not in PurePosixPath(member).parts,
                      "Bytecode cache in wheel: " + member)
                check(not re.search(r"\.so(?:\.\d+)*$", member, flags=re.IGNORECASE), "Linux shared object in wheel: " + member)
                extension_tag = re.search(r"\.(cp\d+)-(win_[^.]+)\.pyd$", member, flags=re.IGNORECASE)
                if extension_tag:
                    check(extension_tag.groups() == (python_tag, "win_arm64"), "Native extension filename has the wrong Python/platform tag: " + member)
                algorithm, value = records.get(member, ("", ""))[0].partition("=")[::2]
                if member == prefix + "RECORD" or member in signatures:
                    algorithm = "sha256"
                else:
                    check(algorithm in {"sha256", "sha384", "sha512"} and bool(value), "Missing or weak RECORD digest: " + member)
                digest, actual_size = hashlib.new(algorithm), 0
                with archive.open(info) as handle:
                    for block in iter(lambda: handle.read(1048576), b""):
                        digest.update(block)
                        actual_size += len(block)
                if member != prefix + "RECORD" and member not in signatures:
                    actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
                    check(actual == value, "RECORD digest mismatch: " + member)
                    check(records[member][1] == str(actual_size), "RECORD size mismatch: " + member)
                architecture = _native_architecture(archive, info)
                if architecture:
                    native.append({"file": member, "architecture": architecture})
            required = CORE_REQUIRED | RTX_SPARK_REQUIRED
            check(required <= set(names), "Missing required module or asset: " + ", ".join(sorted(required - set(names))))
            requirements = metadata.get_all("Requires-Dist", [])
            _integration(archive, names, prefix, requirements, "torch==" + manifest["platform"]["torch"])
            asset_counts = _asset_manifests(archive, names)
            return {"distribution": name, "version": metadata["Version"], "file": path.name,
                    "tag": expected_tag, "ZIP_CRC_valid": True,
                    "RECORD_valid": True, "files": len(names), "native": native,
                    "asset_manifests": asset_counts, "requirements": requirements,
                    "vllm_commit": manifest["vllm"]["commit"], "adapter": ADAPTER}
    except (zipfile.BadZipFile, UnicodeError, KeyError, TypeError, ValueError,
            SyntaxError, configparser.Error) as issue:
        if isinstance(issue, WheelError):
            raise
        raise WheelError(f"{path.name}: {issue}") from issue


def validate_release(paths):
    check(len(paths) == 1, "Provide exactly one standalone vllm wheel")
    results = [validate_wheel(path) for path in paths]
    return {"status": "passed_windows_arm64_release_wheel", "wheels": results,
            "packages_installed": False, "native_libraries_loaded": False, "GPU_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="*", type=Path)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    parser.add_argument("--json", type=Path, help="Write a new validation receipt")
    args = parser.parse_args()
    try:
        report = validate_release(args.wheels or sorted(args.directory.glob("*.whl")))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(report, indent=2) + "\n")
        for wheel in report["wheels"]:
            print(f"PASS {wheel['file']}: {wheel['files']} files, CRC/RECORD, ARM64, required assets")
    except (WheelError, OSError) as issue:
        print("FAIL " + str(issue))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
