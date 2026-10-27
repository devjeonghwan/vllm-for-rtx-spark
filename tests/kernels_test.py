"""CPU tests for the packaged source, policy and native-host contract."""
import ctypes
import json
from pathlib import Path
import runpy
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()
from _rtx_spark.kernels import cute  # noqa: E402
from _rtx_spark.kernels.cute import assets  # noqa: E402
from _rtx_spark.kernels.cute.builder import build  # noqa: E402
from _rtx_spark.kernels.cute import builder  # noqa: E402


class KernelPackTests(unittest.TestCase):
    def temporary_directory(self):
        root = ROOT / ".build" / "kernel-tests"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    @staticmethod
    def minimal_pack(directory, content):
        directory.mkdir(parents=True)
        (directory / "lib").mkdir()
        files = []
        payload = bytearray(128)
        payload[:2] = b"MZ"
        struct.pack_into("<I", payload, 60, 64)
        payload[64:70] = b"PE\0\0\x64\xaa"
        payload.extend(content)
        for name in ("b12x", "hc", "skinny", "gdn"):
            relative = "lib/" + name + ".dll"
            (directory / relative).write_bytes(payload)
            (directory / (name + ".json")).write_text(json.dumps({
                "library": relative}))
        for path in directory.rglob("*"):
            if path.is_file():
                files.append(path.relative_to(directory).as_posix())
        (directory / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "target": {"cpu": "win_arm64", "gpu": "sm_121a", "sm_count": 48}, "files": files}))

    def test_registration_is_cpu_only(self):
        before = {name for name in ("torch", "cutlass", "triton") if name in sys.modules}
        cute.install()
        cute.configure("auto")
        self.assertEqual(before, {name for name in ("torch", "cutlass", "triton") if name in sys.modules})
        self.assertIs(sys.modules["original_cute_coff_runtime"], sys.modules["_rtx_spark.kernels.cute.coff_runtime"])
        with self.assertRaisesRegex(RuntimeError, "provider is not installed"):
            cute.configure("native")

    def test_pack_identity_and_paths(self):
        manifest = assets.verify()
        self.assertTrue({name + ".json" for name in ("b12x", "hc", "skinny", "gdn")} <= set(manifest["files"]))
        self.assertFalse(any(name.startswith("sources/") for name in manifest["files"]))
        self.assertTrue((ROOT / "src/kernels/cute/device/b12x/_lib/dense_gemm.py").is_file())
        for name in manifest["files"]:
            self.assertNotIn("\\", name)
            self.assertTrue(assets.resolve(name).is_file())
        for name in ("../bad", "C:/bad", "/bad"):
            with self.assertRaises(ValueError):
                assets.resolve(name)
        for family in ("b12x", "hc", "skinny", "gdn"):
            self.assertFalse("devpa" in json.dumps(assets.metadata(family)), family)

    def test_native_cpu_architecture(self):
        for family in ("b12x", "hc", "skinny", "gdn"):
            payload = assets.resolve(assets.metadata(family)["library"]).read_bytes()
            offset = struct.unpack_from("<I", payload, 0x3C)[0]
            self.assertEqual(payload[:2], b"MZ")
            self.assertEqual(payload[offset:offset + 6], b"PE\0\0\x64\xaa")

    def test_original_dense_policy_is_complete(self):
        cute.install()
        import original_b12x_aot as backend
        backend._prepare_archive()
        self.assertEqual(len(backend._programs), 89)
        self.assertEqual(len(backend._ranges), 8)
        for table in (backend._ranges, backend._ranges_alpha_one):
            for ranges in table.values():
                self.assertEqual(ranges[0]["first_M"], 1)
                self.assertGreaterEqual(ranges[-1]["last_M"], 8192)
                for left, right in zip(ranges, ranges[1:]):
                    self.assertEqual(left["last_M"] + 1, right["first_M"])
                self.assertTrue(all(row["program"] in backend._programs for row in ranges))

    def test_generated_abi_structures(self):
        from _rtx_spark.kernels.cute.hc_aot import Tensor1Stride, Tensor2Stride
        from _rtx_spark.kernels.cute.skinny_aot import TensorStride, TensorTwoShapeStride
        self.assertEqual(ctypes.sizeof(Tensor1Stride), 24)
        self.assertEqual(ctypes.sizeof(Tensor2Stride), 24)
        self.assertEqual(ctypes.sizeof(TensorStride), 16)
        self.assertEqual(ctypes.sizeof(TensorTwoShapeStride), 24)

    def test_prebuilt_provider_is_compiler_free(self):
        with self.temporary_directory() as directory:
            result = build(Path(directory) / "pack")
            self.assertTrue(result["prebuilt_reused"])
            self.assertFalse(result["compiler_required"])
            self.assertFalse(result["GPU_executed"])
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_install_preserves_previous_assets(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            target = root / "assets/kernels"
            pack = root / "output/kernels"
            self.minimal_pack(target, b"old")
            self.minimal_pack(pack, b"new")
            with patch.object(assets, "data_root", return_value=target):
                result = builder._install(pack, root / "output", root)
            self.assertTrue((target / "lib/hc.dll").read_bytes().endswith(b"new"))
            self.assertTrue((Path(result["previous_pack"]) / "lib/hc.dll").read_bytes().endswith(b"old"))

    def test_install_failure_restores_previous_assets(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            target = root / "assets/kernels"
            pack = root / "output/kernels"
            self.minimal_pack(target, b"old")
            self.minimal_pack(pack, b"new")
            def partial_copy(source, destination):
                destination.mkdir()
                (destination / "partial").write_bytes(b"incomplete")
                raise OSError("simulated interrupted copy")
            with patch.object(assets, "data_root", return_value=target), patch.object(builder.shutil, "copytree", side_effect=partial_copy):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    builder._install(pack, root / "output", root)
            self.assertTrue((target / "lib/hc.dll").read_bytes().endswith(b"old"))
            self.assertTrue((root / "output/failed-kernel-install/partial").is_file())

    def test_runtime_resolution_uses_torch_without_sdk(self):
        from types import SimpleNamespace
        from _rtx_spark.kernels.cute.coff_runtime import _runtime_directory
        with self.temporary_directory() as directory:
            package = Path(directory) / "torch"
            library = package / "lib"
            library.mkdir(parents=True)
            payload = bytearray(128)
            payload[:2] = b"MZ"
            struct.pack_into("<I", payload, 0x3C, 64)
            payload[64:70] = b"PE\0\0\x64\xaa"
            (library / "cudart64_13.dll").write_bytes(payload)
            torch = SimpleNamespace(__file__=str(package / "__init__.py"))
            with patch.dict("os.environ", {"CUDA_PATH": str(Path(directory) / "no-sdk")}):
                self.assertEqual(_runtime_directory(torch), library)

    def test_cuda_runtime_version_compatibility(self):
        from _rtx_spark.kernels.cute.coff_runtime import runtime_version_supported
        self.assertTrue(runtime_version_supported(13040))
        self.assertTrue(runtime_version_supported(13050))
        self.assertFalse(runtime_version_supported(13030))
        self.assertFalse(runtime_version_supported(14000))
        self.assertFalse(runtime_version_supported("13050"))


if __name__ == "__main__":
    unittest.main()
