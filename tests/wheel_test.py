"""Synthetic standalone wheels test release integrity without loading native code."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import struct
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WHEEL_VERSION = runpy.run_path(str(ROOT / "src/version.py"))["WHEEL_VERSION"]
SPEC = importlib.util.spec_from_file_location("release_wheel_check", ROOT / "scripts/check_wheels.py")
wheel_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wheel_check)

CONSOLE_ENTRYPOINT = "[console_scripts]\nvllm = vllm.entrypoints.cli.main:main\n"
ENTRYPOINTS = CONSOLE_ENTRYPOINT + "\n[vllm.general_plugins]\nrtx_spark = vllm._rtx_spark:register\n"


def pe(machine=0xAA64):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    return bytes(data)


def elf(machine=190):
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, machine)
    return bytes(data)


def payload():
    required = wheel_check.CORE_REQUIRED | wheel_check.RTX_SPARK_REQUIRED
    files = {name: pe() if name.endswith(".pyd") else b"# synthetic CPU fixture\n" for name in required}
    files["vllm/__init__.py"] = (
        b"from vllm._rtx_spark.runtime.bootstrap import initialize as _rtx_spark_initialize\n_rtx_spark_initialize()\n")
    files["vllm/engine/arg_utils.py"] = (
        b"from vllm._rtx_spark.runtime import prepare_engine_args, configure_engine\n"
        b"def create_config(args):\n    prepare_engine_args(args)\n    configure_engine(args)\n")
    files[wheel_check.RTX_SPARK + "manifest.json"] = json.dumps(wheel_check.MANIFEST).encode()
    files[wheel_check.ADAPTER.replace(".", "/") + ".py"] = b"# release adapter fixture\n"
    prefix = wheel_check.RTX_SPARK + "assets/native/"
    native = {"format_version": 1, "host": "win_arm64", "files": {}}
    for key in wheel_check.NATIVE_REQUIRED:
        name = key + (".cubin" if key == "csf_cubin" else ".dll")
        data = elf() if key == "csf_cubin" else pe()
        files[prefix + name] = data
        native["files"][key] = {"file": name, "origin": "CPU fixture"}
    files[prefix + "manifest.json"] = json.dumps(native).encode()
    prefix = wheel_check.RTX_SPARK + "assets/kernels/"
    kernel = {"schema_version": 1, "target": {"cpu": "win_arm64"}, "files": []}
    for family in ("b12x", "hc", "skinny", "gdn"):
        data, name = pe(), "lib/" + family + ".dll"
        files[prefix + name] = data
        metadata = {"library": name}
        files[prefix + family + ".json"] = json.dumps(metadata).encode()
    for path, data in files.items():
        if path.startswith(prefix) and path != prefix + "manifest.json":
            kernel["files"].append(path[len(prefix):])
    files[prefix + "manifest.json"] = json.dumps(kernel).encode()
    return files


def wheel(directory, files=None, *, distribution="vllm", tag=wheel_check.EXPECTED_TAG,
          requirements=None, entrypoints=ENTRYPOINTS, bad_record=None, unrecorded=None):
    name, version = distribution.replace("-", "_"), WHEEL_VERSION
    prefix = f"{name}-{version}.dist-info/"
    files = dict(files if files is not None else payload())
    metadata = f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\nRequires-Python: >=3.13,<3.14\n"
    for requirement in requirements if requirements is not None else (wheel_check.EXPECTED_TORCH,):
        metadata += "Requires-Dist: " + requirement + "\n"
    files[prefix + "METADATA"] = metadata.encode()
    files[prefix + "WHEEL"] = f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()
    if entrypoints is not None:
        files[prefix + "entry_points.txt"] = entrypoints.encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for path, data in files.items():
        if path == unrecorded:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, "sha256=" + ("wrong" if path == bad_record else digest), len(data)))
    writer.writerow((prefix + "RECORD", "", ""))
    files[prefix + "RECORD"] = record.getvalue().encode()
    path = directory / f"{name}-{version}-{wheel_check.EXPECTED_TAG}.whl"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member, data in sorted(files.items()):
            archive.writestr(member, data)
    return path


class WheelTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / ".build/wheel-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_standalone_release(self):
        report = wheel_check.validate_release([wheel(self.directory)])
        self.assertEqual(report["status"], "passed_windows_arm64_release_wheel")
        self.assertFalse(report["GPU_executed"])
        self.assertFalse(report["packages_installed"])
        self.assertEqual(len(report["wheels"]), 1)
        result = report["wheels"][0]
        self.assertEqual(result["distribution"], "vllm")
        self.assertEqual(result["asset_manifests"], {"native_assets": 6, "kernel_assets": 8})
        self.assertEqual(sum(item["architecture"] == "ARM64" for item in result["native"]), 13)
        self.assertEqual(sum(item["architecture"] == "CUDA" for item in result["native"]), 1)

    def test_rejects_old_two_wheel_invocation(self):
        for paths in ([], ["old-core.whl", "old-runtime.whl"]):
            with self.subTest(paths=paths), self.assertRaisesRegex(wheel_check.WheelError, "exactly one standalone"):
                wheel_check.validate_release(paths)

    def test_rejects_companion_distribution(self):
        with self.assertRaisesRegex(wheel_check.WheelError, "standalone vllm distribution"):
            wheel_check.validate_wheel(wheel(self.directory, distribution="vllm-rtx-spark"))

    def test_rejects_missing_private_package_source_or_vendor(self):
        for member in ("runtime/__init__.py", "kernels/__init__.py", "kernels/cute/__init__.py",
                       "kernels/csf/__init__.py", "recipes/__init__.py", "recipes/qwen3_8/__init__.py",
                       "vendor/openai_harmony/openai_harmony.pyd",
                       "kernels/cute/device/b12x/policy/_profiles/data/nvidia.gb10.48sm.json.gz"):
            with self.subTest(member=member):
                files = payload()
                del files[wheel_check.RTX_SPARK + member]
                with self.assertRaisesRegex(wheel_check.WheelError, "Missing required module"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_legacy_companion_paths(self):
        for member in ("vllm_rtx_spark/__init__.py", "vllm/_rtx_spark/vendor/vllm_rtx_spark/__init__.py",
                       "vllm/_rtx_spark/vendor/vllm_rtx_spark-0.1.0.dist-info/WHEEL"):
            with self.subTest(member=member):
                files = payload()
                files[member] = b"# obsolete package fixture\n"
                with self.assertRaisesRegex(wheel_check.WheelError, "Legacy companion package path"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_legacy_companion_dependency(self):
        for requirement in ("vllm-rtx-spark==0.1.0", "VLLM_RTX_SPARK[extra] >= 0.1.0"):
            with self.subTest(requirement=requirement), self.assertRaisesRegex(wheel_check.WheelError, "legacy companion"):
                wheel_check.validate_wheel(wheel(self.directory, requirements=(wheel_check.EXPECTED_TORCH, requirement)))

    def test_rejects_legacy_companion_plugin(self):
        with self.assertRaisesRegex(wheel_check.WheelError, "Legacy companion plugin"):
            wheel_check.validate_wheel(wheel(
                self.directory, entrypoints=ENTRYPOINTS + "anything = vllm_rtx_spark:register\n"))

    def test_rejects_missing_or_wrong_rtx_spark_plugin(self):
        for plugin in ("", "rtx_spark = vllm._spark:register",
                       "rtx_spark = vllm_rtx_spark:register", "rtx_spark = vllm._rtx_spark:wrong"):
            entries = CONSOLE_ENTRYPOINT + ("\n[vllm.general_plugins]\n" + plugin if plugin else "")
            with self.subTest(plugin=plugin), self.assertRaisesRegex(wheel_check.WheelError, "RTX Spark plugin registration"):
                wheel_check.validate_wheel(wheel(self.directory, entrypoints=entries))

    def test_rejects_old_internal_namespace(self):
        files = payload()
        files["vllm/_spark/__init__.py"] = b"# previous internal namespace\n"
        with self.assertRaisesRegex(wheel_check.WheelError, "Legacy internal namespace"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_missing_or_changed_release_manifest(self):
        member = wheel_check.RTX_SPARK + "manifest.json"
        for field in (None, "version", "commit"):
            with self.subTest(field=field):
                files = payload()
                if field is None:
                    del files[member]
                else:
                    manifest = json.loads(files[member])
                    manifest["vllm"][field] = "0.0.0" if field == "version" else "0" * 40
                    files[member] = json.dumps(manifest).encode()
                with self.assertRaisesRegex(wheel_check.WheelError, "release manifest"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_missing_adapter(self):
        files = payload()
        del files[wheel_check.ADAPTER.replace(".", "/") + ".py"]
        with self.assertRaisesRegex(wheel_check.WheelError, "vLLM adapter"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_missing_or_wrong_vllm_console_entrypoint(self):
        for entrypoints in (None, "[console_scripts]\nvllm = obsolete.launcher:main\n"):
            with self.subTest(entrypoints=entrypoints), self.assertRaisesRegex(wheel_check.WheelError, "console entrypoint"):
                wheel_check.validate_wheel(wheel(self.directory, entrypoints=entrypoints))

    def test_rejects_bootstrap_text_without_executed_hook(self):
        files = payload()
        files["vllm/__init__.py"] = b"# from vllm._rtx_spark.runtime.bootstrap import initialize\n# initialize()\n"
        with self.assertRaisesRegex(wheel_check.WheelError, "Missing RTX Spark hook"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_old_engine_hook_import(self):
        files = payload()
        files["vllm/engine/arg_utils.py"] = files["vllm/engine/arg_utils.py"].replace(
            b"vllm._rtx_spark.runtime", b"vllm_rtx_spark.runtime")
        with self.assertRaisesRegex(wheel_check.WheelError, "Missing RTX Spark hook"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_missing_or_wrong_torch_pin(self):
        for requirements in ((), ("torch>=2.14",), ("torch==2.14.0+cu130",)):
            with self.subTest(requirements=requirements), self.assertRaisesRegex(wheel_check.WheelError, "Torch/CUDA ABI"):
                wheel_check.validate_wheel(wheel(self.directory, requirements=requirements))

    def test_rejects_wrong_native_architecture(self):
        for member in ("vllm/_C_stable_libtorch.pyd", wheel_check.RTX_SPARK + "vendor/llguidance/_lib.pyd"):
            with self.subTest(member=member):
                files = payload()
                files[member] = pe(0x8664)
                with self.assertRaisesRegex(wheel_check.WheelError, "not ARM64"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_wrong_wheel_tag(self):
        with self.assertRaisesRegex(wheel_check.WheelError, "ABI/platform tag"):
            wheel_check.validate_wheel(wheel(self.directory, tag="cp313-cp313-win_amd64"))

    def test_rejects_wrong_extension_filename_tag(self):
        files = payload()
        files["extra.cp312-win_arm64.pyd"] = pe()
        with self.assertRaisesRegex(wheel_check.WheelError, "extension filename"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_record_mismatch(self):
        with self.assertRaisesRegex(wheel_check.WheelError, "RECORD digest mismatch"):
            wheel_check.validate_wheel(wheel(self.directory, bad_record="vllm/__init__.py"))

    def test_rejects_unrecorded_file(self):
        files = payload()
        files["extra.py"] = b"# fixture\n"
        with self.assertRaisesRegex(wheel_check.WheelError, "absent from RECORD"):
            wheel_check.validate_wheel(wheel(self.directory, files, unrecorded="extra.py"))

    def test_rejects_missing_asset_even_with_valid_record(self):
        for directory in ("native", "kernels"):
            with self.subTest(directory=directory):
                files = payload()
                path = wheel_check.RTX_SPARK + "assets/" + directory + "/manifest.json"
                manifest = json.loads(files[path])
                if directory == "native":
                    manifest["files"]["csf_cubin"]["file"] = "missing.cubin"
                else:
                    manifest["files"].append("lib/missing.dll")
                files[path] = json.dumps(manifest).encode()
                with self.assertRaisesRegex(wheel_check.WheelError, "Missing manifest asset"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_invalid_manifest_asset_paths(self):
        for directory in ("native", "kernels"):
            with self.subTest(directory=directory):
                files = payload()
                path = wheel_check.RTX_SPARK + "assets/" + directory + "/manifest.json"
                manifest = json.loads(files[path])
                if directory == "native":
                    manifest["files"]["csf_host"]["file"] = "../outside.dll"
                else:
                    manifest["files"].append("../outside.dll")
                files[path] = json.dumps(manifest).encode()
                with self.assertRaisesRegex(wheel_check.WheelError, "Unsafe wheel path"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_missing_native_component(self):
        files = payload()
        path = wheel_check.RTX_SPARK + "assets/native/manifest.json"
        manifest = json.loads(files[path])
        del manifest["files"]["csf_cubin"]
        files[path] = json.dumps(manifest).encode()
        with self.assertRaisesRegex(wheel_check.WheelError, "Missing native CSF/PLE/MTP/sampling component"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_old_kernel_source_asset_layout(self):
        files = payload()
        prefix = wheel_check.RTX_SPARK + "assets/kernels/"
        path = prefix + "manifest.json"
        manifest = json.loads(files[path])
        data = b"# duplicated source fixture\n"
        files[prefix + "sources/b12x/__init__.py"] = data
        manifest["files"].append("sources/b12x/__init__.py")
        files[path] = json.dumps(manifest).encode()
        with self.assertRaisesRegex(wheel_check.WheelError, "not duplicated assets"):
            wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_unsafe_zip_paths(self):
        for member in ("../outside.py", "C:/outside.py", "vllm\\outside.py", "/outside.py"):
            with self.subTest(member=member):
                files = payload()
                files[member] = b"fixture"
                with self.assertRaisesRegex(wheel_check.WheelError, "Unsafe wheel path"):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_linux_host_and_bytecode(self):
        for name, data in (("extra.so", elf(183)), ("extra.bin", elf(183)), ("extra.pyc", b"fixture")):
            with self.subTest(name=name):
                files = payload()
                files[name] = data
                with self.assertRaises(wheel_check.WheelError):
                    wheel_check.validate_wheel(wheel(self.directory, files))

    def test_rejects_crc_corruption(self):
        path = wheel(self.directory)
        with zipfile.ZipFile(path) as archive:
            entry = archive.getinfo("vllm/__init__.py")
        with path.open("r+b") as handle:
            handle.seek(entry.header_offset)
            header = handle.read(30)
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            offset = entry.header_offset + 30 + name_length + extra_length
            handle.seek(offset)
            handle.write(b"!")
        with self.assertRaisesRegex(wheel_check.WheelError, "CRC"):
            wheel_check.validate_wheel(path)


if __name__ == "__main__":
    unittest.main()

