"""Build-only export of original Qwen4 HC down+SiLU kernels; no GPU execution."""
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
ORIGINAL = ROOT / "src/kernels/cute/device/hc"


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
        print(json.dumps({"status": "plan_only", "GPU_executed": False}))
        return
    out = Path(os.environ["SPARK_BUILD_OUTPUT"]) / "hc"
    out.mkdir(exist_ok=False)
    report = {"status": "started_original_HC_CPU_compile", "pid": os.getpid(), "GPU_executed": False,
              "Torch_imported": False, "blocked_CUDA_calls": [], "source_AST_modified": False,
              "rank": 320, "HC": 4, "K": 10240, "logical_N": 324, "physical_output_stride": 336,
              "source_sha256": {name: sha(ORIGINAL / name) for name in ("hc_down_silu.py", "_hc_down_silu_fma.py", "_hc_down_silu_mma.py")}, "kernels": []}
    os.environ.update(CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_121a", CUTE_DSL_DISABLE_FILE_CACHING="1",
                      CUDA_PATH=os.environ["SPARK_CUDA_ROOT"])
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    os.environ["TEMP"] = os.environ["TMP"] = str(out / "temp")
    Path(os.environ["TEMP"]).mkdir()
    handles = []
    for folder in (Path(os.environ["SPARK_CUTE_LIB"]), Path(os.environ["CUDA_PATH"]) / "bin/x64"):
        if folder.is_dir():
            handles.append(os.add_dll_directory(str(folder)))
    try:
        runpy.run_path(str(HERE.parent / "process_arch.py"))["require_windows_x64_process"]()
        import cuda.bindings.driver as driver
        import cuda.bindings.runtime as runtime
        def reject(name):
            def deny(*a, **kw):
                report["blocked_CUDA_calls"].append(name)
                raise RuntimeError("No CUDA execution in build helper: " + name)
            return deny
        for module, names in ((driver, ("cuInit", "cuDeviceGet", "cuDeviceGetCount", "cuCtxCreate", "cuCtxCreate_v2", "cuCtxCreate_v3", "cuDevicePrimaryCtxRetain", "cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx", "cuLaunchKernel", "cuLaunchKernelEx")),
                              (runtime, ("cudaGetDevice", "cudaGetDeviceCount", "cudaSetDevice", "cudaMalloc", "cudaMallocAsync", "cudaLaunchKernel", "cudaLaunchKernelEx", "cudaLibraryLoadData", "cudaStreamCreate"))):
            for name in names:
                if hasattr(module, name):
                    setattr(module, name, reject(name))
        import cutlass
        import cutlass.cute as cute
        from cutlass.base_dsl.compiler import DumpDir, HostTarget, PtxasOptions
        from cutlass.cute.runtime import make_fake_tensor
        exporter = load("kernel_export", HERE.parent / "export.py")
        fma = load("original_HC_FMA_unchanged", ORIGINAL / "_hc_down_silu_fma.py")
        mma = load("original_HC_MMA_unchanged", ORIGINAL / "_hc_down_silu_mma.py")
        options = (cute.GPUArch("sm_121a"), cute.KeepPTX(True), cute.KeepCUBIN(True), DumpDir(str(out)),
                   HostTarget("llvm -mtriple=aarch64-pc-windows-msvc"))
        for m in range(1, 5):
            # Original policy uses these four static-M/K FMA kernels, prefetch only M1.
            gemm = fma.HcDownSiluFma(k=10240, threadblock_size=128, prefetch_pdl_weights=m == 1, rank=320, hc=4)
            n = cute.sym_int32()
            # The official C exporter represents dynamic strides as int64.
            output_stride = cute.sym_int64()
            tensors = (make_fake_tensor(cutlass.BFloat16, (m, 10240), (10240, 1), assumed_align=16),
                       make_fake_tensor(cutlass.BFloat16, (n, 10240), (10240, 1), assumed_align=16),
                       make_fake_tensor(cutlass.BFloat16, (m, n), (output_stride, 1), assumed_align=2))
            compiled = cute.compile[options + (PtxasOptions("-maxrregcount=64"),)](
                gemm, *tensors, m, 10240, 1, driver.CUstream(0), no_jit_engine=True)
            compiled.export_to_c(str(out), "original_hc_fma_m" + str(m) + "_host", "original_hc_fma_m" + str(m))
            item = {"backend": "fma", "M": m, "original_dispatch_key": ["fma", m, 10240, 128],
                    "prefetch_pdl_weights": m == 1, **exporter.export(compiled, out, "original_hc_fma_m" + str(m))}
            report["kernels"].append(item)
            (out / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Exported original HC FMA M" + str(m), flush=True)
        for stages, tile_n, m_range in ((5, 8, [5, 16]), (5, 16, [17, 32]), (4, 16, [33, 48])):
            gemm = mma.HcDownSiluMma(tile_n=tile_n, num_stages=stages, split_k=6, rank=320, hc=4)
            m, n, k = cute.sym_int32(), cute.sym_int32(), cute.sym_int32(divisibility=8)
            k_stride, output_stride = cute.sym_int64(divisibility=8), cute.sym_int64()
            tensors = (make_fake_tensor(cutlass.BFloat16, (m, k), (k_stride, 1), assumed_align=16),
                       make_fake_tensor(cutlass.BFloat16, (n, k), (k_stride, 1), assumed_align=16),
                       make_fake_tensor(cutlass.BFloat16, (m, n), (output_stride, 1), assumed_align=2))
            compiled = cute.compile[options](gemm, *tensors, driver.CUstream(0), no_jit_engine=True)
            name = "original_hc_mma_stages" + str(stages) + "_n" + str(tile_n)
            compiled.export_to_c(str(out), name + "_host", name)
            item = {"backend": "mma", "M_range": m_range, "original_dispatch_key": ["mma", 6, stages, tile_n],
                    **exporter.export(compiled, out, name)}
            report["kernels"].append(item)
            (out / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Exported " + name, flush=True)
        if "torch" in sys.modules or report["blocked_CUDA_calls"]:
            raise RuntimeError("CPU-only build contract violated")
        report["status"] = "passed_original_HC_7_kernels_CPU_compile_export_only"
    except BaseException as error:
        report.update(status="failed_original_HC_CPU_compile_export", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report["Torch_imported"] = "torch" in sys.modules
        report["ARM64_host_artifacts"] = [{"path": str(path), "bytes": path.stat().st_size,
                                           "sha256": sha(path), "COFF_machine": path.read_bytes()[:2].hex() if path.suffix == ".o" else None}
                                          for path in out.glob("*_host.*")]
        report["executed_script_sha256"] = sha(__file__)
        (out / "executed-build.py").write_bytes(Path(__file__).read_bytes())
        (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"], "result": str(out / "result.json"), "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
