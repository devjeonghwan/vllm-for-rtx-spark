"""BUILD ONLY: source-identical original SM121 Qwen skinny FMA kernels."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["SPARK_SOURCE_ROOT"])
HELPER = Path(os.environ["SPARK_COMPILER_ROOT"])
KERNEL = ROOT / "src/kernels/cute/device/skinny/_skinny_gemm.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-build", action="store_true")
    args = parser.parse_args()
    if not args.execute_build:
        print(json.dumps({"status": "plan_only", "inventory": str(HERE / "inventory.json"), "GPU_executed": False}))
        return
    output = Path(os.environ["SPARK_BUILD_OUTPUT"]) / "skinny"
    output.mkdir(exist_ok=False)
    report = {"status": "started_original_skinny_compile_only", "Torch_imported": False,
              "GPU_executed": False, "blocked_CUDA_calls": [], "programs": [],
              "original_kernel_sha256": sha(KERNEL), "original_policy_inventory_sha256": sha(HERE / "inventory.json")}
    os.environ.update(CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_121a", CUTE_DSL_DISABLE_FILE_CACHING="1",
                      CUDA_PATH=os.environ["SPARK_CUDA_ROOT"])
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    os.environ["TEMP"] = os.environ["TMP"] = str(output / "temp")
    Path(os.environ["TEMP"]).mkdir()
    dirs = [os.add_dll_directory(str(path)) for path in
            (Path(os.environ["SPARK_CUTE_LIB"]), Path(os.environ["CUDA_PATH"]) / "bin/x64")]
    try:
        runpy.run_path(str(HERE.parent / "process_arch.py"))["require_windows_x64_process"]()
        import cuda.bindings.driver as driver
        import cuda.bindings.runtime as runtime
        def reject(name):
            def call(*a, **kw):
                report["blocked_CUDA_calls"].append(name)
                raise RuntimeError("BUILD ONLY: " + name)
            return call
        for module, names in ((driver, ("cuInit", "cuDeviceGet", "cuDeviceGetCount", "cuCtxCreate", "cuDevicePrimaryCtxRetain", "cuModuleLoadData", "cuLaunchKernel", "cuLaunchKernelEx")),
                              (runtime, ("cudaGetDevice", "cudaGetDeviceCount", "cudaMalloc", "cudaSetDevice", "cudaLaunchKernel", "cudaLaunchKernelEx", "cudaLibraryLoadData"))):
            for name in names:
                if hasattr(module, name):
                    setattr(module, name, reject(name))
        import cutlass
        import cutlass.cute as cute
        from cutlass.base_dsl.compiler import DumpDir, HostTarget, PtxasOptions
        kernel = load("original_skinny_kernel_unchanged", KERNEL)
        fake_path = ROOT / "src/kernels/cute/device/skinny/quack_compile_utils.py"
        if sha(fake_path) != "9d708639188592d22ab1fc98370d66ebd9ebf08cba88b6673fcfb1d07b214321":
            raise ValueError("Original image fake-tensor helper changed")
        make_fake_tensor = load("original_quack_fake_tensor_helper", fake_path).make_fake_tensor
        inventory = json.loads((HERE / "inventory.json").read_text())
        for ordinal, item in enumerate(inventory["configs"]):
            config, prefix = item["config"], "original_skinny_" + item["id"]
            folder = output / item["id"]
            folder.mkdir()
            n = cute.sym_int(divisibility=config["outputs_per_block"])
            k = config["static_k"] if config["static_k"] is not None else cute.sym_int(divisibility=config["block_size"] * config["vector_width"])
            m = config["num_rows"]
            tensors = (make_fake_tensor(cutlass.BFloat16, (m, k), divisibility=config["vector_width"]),
                       make_fake_tensor(cutlass.BFloat16, (n, k), divisibility=config["vector_width"]),
                       make_fake_tensor(cutlass.BFloat16, (m, n), divisibility=1),
                       make_fake_tensor(cutlass.BFloat16, (m, n), divisibility=1))
            launch = kernel.CuteSkinnyGemm(element_type=cutlass.BFloat16,
                num_rows=m, block_size=config["block_size"], outputs_per_block=config["outputs_per_block"],
                vector_width=config["vector_width"], k_unroll=config["k_unroll"],
                has_residual=False, use_pdl=True, static_k=config["static_k"])
            options = (cute.GPUArch("sm_121a"), cute.KeepPTX(True), cute.KeepCUBIN(True), DumpDir(str(folder)),
                       HostTarget("llvm -mtriple=aarch64-pc-windows-msvc"), PtxasOptions("-maxrregcount=64"))
            print(f"[{ordinal+1}/{len(inventory['configs'])}] original SM121 skinny {config}", flush=True)
            compiled = cute.compile[options](launch, *tensors, driver.CUstream(0), no_jit_engine=True)
            if compiled.engine is not None:
                raise RuntimeError("Unexpected CPU JIT execution engine")
            compiled.export_to_c(str(folder), "host", prefix)
            (folder / "device.cubin").write_bytes(compiled.__cubin__)
            (folder / "device.ptx").write_text(compiled.__ptx__)
            program = {**item, "prefix": prefix, "use_pdl": True, "has_residual": False,
                       "artifacts": {name: {"path": str(folder / name), "sha256": sha(folder / name),
                                               "bytes": (folder / name).stat().st_size}
                                     for name in ("host.o", "host.h", "device.cubin", "device.ptx")}}
            if (folder / "host.o").read_bytes()[:2] != b"\x64\xaa":
                raise RuntimeError("Original generated host is not ARM64 COFF")
            report["programs"].append(program)
            (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["blocked_CUDA_calls"] or "torch" in sys.modules:
            raise RuntimeError("Original kernel compilation exceeded CPU-only scope")
        report["status"] = "passed_original_SM121_skinny_GPU_and_ARM64_host_AOT_compile_only"
    except BaseException as error:
        report.update(status="failed_original_skinny_AOT_compile", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["Torch_imported"] = "torch" in sys.modules
        report["source_sha256"] = sha(__file__)
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "directory": str(output), "error": report.get("error")}), flush=True)
        dirs.clear()
    raise SystemExit(1 if report["status"].startswith("failed") else 0)


if __name__ == "__main__":
    main()
