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
import stat
import struct
import zipfile

EXPECTED_TAG = "cp313-cp313-win_arm64"
EXPECTED_TORCH = "torch==2.14.0+cu134"
SPARK = "vllm/_spark/"
CORE_REQUIRED = {
    "vllm/__init__.py", "vllm/entrypoints/cli/main.py",
    "vllm/entrypoints/__init__.py", "vllm/entrypoints/cli/__init__.py",
    "vllm/engine/__init__.py", "vllm/engine/arg_utils.py",
    "vllm/_C_stable_libtorch.pyd", "vllm/_moe_C_stable_libtorch.pyd",
}
SPARK_REQUIRED = {
    SPARK + name for name in (
        "__init__.py", "runtime/__init__.py", "runtime/bootstrap.py", "runtime/check.py",
        "runtime/ple/__init__.py", "kernels/__init__.py", "kernels/csf/__init__.py",
        "kernels/cute/__init__.py", "kernels/cute/assets.py", "kernels/cute/b12x_aot.py",
        "kernels/cute/coff_runtime.py", "kernels/cute/hc_aot.py", "kernels/cute/builder.py",
        "kernels/cute/skinny_aot.py", "kernels/cute/gdn_aot.py",
        "kernels/cute/device/b12x/__init__.py", "kernels/cute/device/b12x/_lib/__init__.py",
        "kernels/cute/device/b12x/policy/__init__.py",
        "kernels/cute/device/b12x/policy/_profiles/__init__.py",
        "kernels/cute/device/b12x/policy/_profiles/data/nvidia.gb10.48sm.json.gz",
        "assets/native/manifest.json", "assets/kernels/manifest.json",
        "recipes/__init__.py", "recipes/qwen3_8/__init__.py",
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


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _asset(archive, hashes, prefix, name, digest, size=None):
    path = safe_name(prefix + safe_name(name))
    check(path in hashes, "Missing manifest asset: " + path)
    check(hashes[path][0] == digest, "Manifest SHA256 mismatch: " + path)
    if size is not None:
        check(hashes[path][1] == size, "Manifest size mismatch: " + path)
    return archive.read(path)


def _asset_manifests(archive, hashes):
    native_prefix, kernel_prefix = SPARK + "assets/native/", SPARK + "assets/kernels/"
    native = json.loads(archive.read(native_prefix + "manifest.json"))
    check(native["format_version"] == 1 and native["host"] == "win_arm64", "Wrong native manifest target")
    check(NATIVE_REQUIRED <= native["files"].keys(), "Missing native CSF/PLE/MTP/sampling component")
    for record in native["files"].values():
        _asset(archive, hashes, native_prefix, record["file"], record["sha256"], record["bytes"])
    kernel = json.loads(archive.read(kernel_prefix + "manifest.json"))
    check(kernel["schema_version"] == 1 and kernel["target"]["cpu"] == "win_arm64", "Wrong kernel manifest target")
    check(not any(name.startswith("sources/") for name in kernel["files"]),
          "Kernel sources belong in the private package, not duplicated assets")
    for name, digest in kernel["files"].items():
        _asset(archive, hashes, kernel_prefix, name, digest)
    for family in ("b12x", "hc", "skinny", "gdn"):
        name = family + ".json"
        check(name in kernel["files"], "Missing kernel family metadata: " + family)
        metadata = json.loads(archive.read(kernel_prefix + name))
        check(metadata["library"] in kernel["files"], "Kernel library is not pinned in its manifest")
        _asset(archive, hashes, kernel_prefix, metadata["library"], metadata["library_sha256"])
    return {"native_assets": len(native["files"]), "kernel_assets": len(kernel["files"])}


def _called_import(archive, member, module, function):
    tree = ast.parse(archive.read(member), filename=member)
    aliases = {alias.asname or alias.name for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module == module and node.level == 0
               for alias in node.names if alias.name == function}
    check(aliases and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id in aliases for node in ast.walk(tree)),
          f"Missing Spark hook {module}.{function} in {member}")


def _integration(archive, names, prefix, requirements):
    check(not any(re.search(r"(?:^|/)vllm[-_]rtx[-_]spark(?:/|[-.])", name, re.IGNORECASE)
                  for name in names), "Legacy companion package path in standalone wheel")
    for requirement in requirements:
        dependency = re.match(r"\s*([a-zA-Z0-9_.-]+)", requirement)
        check(dependency is not None, "Invalid Requires-Dist entry")
        check(re.sub(r"[-_.]+", "-", dependency[1].lower()) != "vllm-rtx-spark",
              "Standalone vLLM must not depend on the legacy companion package")
    check(EXPECTED_TORCH in {item.replace(" ", "") for item in requirements},
          "Native wheel does not pin the tested Torch/CUDA ABI: vllm")
    entrypoint_path = prefix + "entry_points.txt"
    check(entrypoint_path in names, "Missing vllm console entrypoint")
    entries = configparser.ConfigParser(interpolation=None)
    entries.read_string(archive.read(entrypoint_path).decode("utf-8"))
    check(entries.get("console_scripts", "vllm", fallback="").strip()
          == "vllm.entrypoints.cli.main:main", "Missing or incorrect vllm console entrypoint")
    for section in entries.sections():
        for key, value in entries.items(section):
            check(not re.search(r"vllm[-_]rtx[-_]spark", value, re.IGNORECASE)
                  and not (section == "vllm.general_plugins" and key.replace("-", "_") == "rtx_spark"),
                  "Legacy companion plugin or entrypoint in standalone wheel")
    _called_import(archive, "vllm/__init__.py", "vllm._spark.runtime.bootstrap", "initialize")
    for function in ("prepare_engine_args", "configure_engine"):
        _called_import(archive, "vllm/engine/arg_utils.py", "vllm._spark.runtime", function)


def validate_wheel(path, *, expected_tag=EXPECTED_TAG):
    """Verify container integrity, installed layout, native targets and asset pins."""
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
            filename_parts = path.stem.split("-")
            check(len(filename_parts) in (5, 6) and path.suffix == ".whl", "Invalid wheel filename")
            check(re.sub(r"[-_.]+", "-", filename_parts[0].lower()) == name
                  and filename_parts[1] == metadata["Version"], "Filename and distribution metadata differ")
            check(set(wheel.get_all("Tag", [])) == {expected_tag}, "Wrong wheel ABI/platform tag")
            check("-".join(path.stem.split("-")[-3:]) == expected_tag, "Filename and wheel ABI/platform differ")
            check(wheel["Root-Is-Purelib"] == "false", "Native wheel is incorrectly marked pure Python")
            python_requires = {part.strip() for part in metadata["Requires-Python"].split(",")}
            check(python_requires == {">=3.13", "<3.14"}, "Expected the tested CPython 3.13 range")
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
            hashes, native = {}, []
            for info in infos:
                member = info.filename
                if info.is_dir():
                    continue
                check(not member.lower().endswith((".pyc", ".pyo")) and "__pycache__" not in PurePosixPath(member).parts,
                      "Bytecode cache in wheel: " + member)
                check(not re.search(r"\.so(?:\.\d+)*$", member, flags=re.IGNORECASE), "Linux shared object in wheel: " + member)
                extension_tag = re.search(r"\.(cp\d+)-(win_[^.]+)\.pyd$", member, flags=re.IGNORECASE)
                if extension_tag:
                    check(extension_tag.groups() == ("cp313", "win_arm64"), "Native extension filename has the wrong Python/platform tag: " + member)
                algorithm, value = records.get(member, ("", ""))[0].partition("=")[::2]
                if member == prefix + "RECORD" or member in signatures:
                    algorithm = "sha256"
                else:
                    check(algorithm in {"sha256", "sha384", "sha512"} and bool(value), "Missing or weak RECORD digest: " + member)
                digest, sha256, actual_size = hashlib.new(algorithm), hashlib.sha256(), 0
                with archive.open(info) as handle:
                    for block in iter(lambda: handle.read(1048576), b""):
                        digest.update(block)
                        sha256.update(block)
                        actual_size += len(block)
                hashes[member] = (sha256.hexdigest(), actual_size)
                if member != prefix + "RECORD" and member not in signatures:
                    actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
                    check(actual == value, "RECORD digest mismatch: " + member)
                    check(records[member][1] == str(actual_size), "RECORD size mismatch: " + member)
                architecture = _native_architecture(archive, info)
                if architecture:
                    native.append({"file": member, "architecture": architecture})
            required = CORE_REQUIRED | SPARK_REQUIRED
            check(required <= set(names), "Missing required module or asset: " + ", ".join(sorted(required - set(names))))
            requirements = metadata.get_all("Requires-Dist", [])
            _integration(archive, names, prefix, requirements)
            asset_counts = _asset_manifests(archive, hashes)
            return {"distribution": name, "version": metadata["Version"], "file": path.name,
                    "sha256": sha_file(path), "tag": expected_tag, "ZIP_CRC_valid": True,
                    "RECORD_valid": True, "files": len(names), "native": native,
                    "asset_manifests": asset_counts, "requirements": requirements}
    except (zipfile.BadZipFile, UnicodeError, KeyError, TypeError, ValueError,
            SyntaxError, configparser.Error) as issue:
        if isinstance(issue, WheelError):
            raise
        raise WheelError(f"{path.name}: {issue}") from issue


def validate_release(paths):
    check(len(paths) == 1, "Provide exactly one standalone vllm wheel")
    results = [validate_wheel(path) for path in paths]
    return {"status": "passed_windows_arm64_release_wheel", "wheels": results,
            "validator_sha256": sha_file(__file__),
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
            print(f"PASS {wheel['file']}: {wheel['files']} files, CRC/RECORD, ARM64, asset hashes")
    except (WheelError, OSError) as issue:
        print("FAIL " + str(issue))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
