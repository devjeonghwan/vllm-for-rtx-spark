"""Approved build-only x64 compile of ORIGINAL FI SM120 GDN to SM121a+ARM64 host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["SPARK_SOURCE_ROOT"])
HELPER = Path(os.environ["SPARK_COMPILER_ROOT"])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(compiled, directory, name):
    if getattr(compiled, "engine", None) is not None:
        raise RuntimeError("CPU build unexpectedly created a JIT execution engine")
    (directory / (name + ".mlir")).write_text(str(compiled.ir_module), encoding="utf-8")
    artifacts = {}
    for kind, value in (("ptx", compiled.__ptx__), ("cubin", compiled.__cubin__)):
        path = directory / (name + "." + kind)
        if isinstance(value, str) and kind == "ptx" and ("\n" in value or ".version" in value): path.write_text(value, encoding="utf-8")
        elif isinstance(value, str) and Path(value).is_file(): shutil.copyfile(value, path)
        elif isinstance(value, bytes): path.write_bytes(value)
        else: raise RuntimeError("Compiler did not export original " + kind)
        artifacts[kind] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    raw = Path(artifacts["cubin"]["path"]).read_bytes()
    if raw[:4] != b"\x7fELF" or struct.unpack_from("<H", raw, 18)[0] != 190:
        raise RuntimeError("Expected genuine CUDA device ELF")
    ptx = Path(artifacts["ptx"]["path"]).read_text(encoding="utf-8")
    if not re.search(r"\.target\s+sm_121a\b", ptx): raise RuntimeError("Expected original target SM121a")
    # Original FI compiler checks this exact unsupported instruction after JIT.
    if "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint" in ptx:
        raise RuntimeError("Original FI SM12x PTX check rejected cluster-scoped TMA load")
    compiled.export_to_c(str(directory), name + "_host", name)
    host = directory / (name + "_host.o")
    if host.read_bytes()[:2] != b"\x64\xaa": raise RuntimeError("Cross-produced host is not ARM64 COFF")
    return dict(artifacts=artifacts, host_CO_FF=str(host), host_machine="AA64", host_sha256=sha(host),
        exported_headers=[str(p) for p in directory.glob(name + "_host*.h")],
        lowered_IR=str(directory / (name + ".mlir")), JIT_engine_created=False,
        host_signature=str(compiled.execution_args.signature),
        kernel_info={str(k): [{"name": str(getattr(a, "name", "")), "value": str(getattr(a, "value", a))} for a in v]
                     for k, v in compiled.kernel_info.items()},
        PTX_entry_parameters=re.findall(r"\.visible\s+\.entry\s+[^\{]+", ptx))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-build", action="store_true")
    parser.add_argument("--checkpoint", choices=("off", "on", "both"), default="both")
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    if (HERE / "build-deps-x64").is_dir():
        sys.path.insert(0, str(HERE / "build-deps-x64"))
    from source_loader import SourceGraph
    graph = SourceGraph()
    module_name = "flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120"
    graph.add(module_name, {"_FullyFusedDeltaRuleSm120"})
    if not args.execute_build:
        print(json.dumps(dict(status="source_plan_only", Torch_imported=False, GPU_executed=False,
            interpreter=sys.executable, original_sources=graph.inventory(),
            selected_variants=args.checkpoint, actual_model=dict(Q_heads=16, K_heads=16, V_heads=48, D=128),
            CP_auto_scope="Original TP1 model H48 on48SM never matches parallelism<SM/2 orSM/3; CP forced mode not claimed"), indent=2))
        return
    attempt = Path(os.environ["SPARK_BUILD_OUTPUT"]) / "gdn"
    attempt.mkdir(exist_ok=False)
    report = dict(status="started_CPU_GDN_compile", pid=os.getpid(), machine=platform.machine(), Torch_imported=False,
        GPU_executed=False, blocked_CUDA_calls=[], original_sources=graph.inventory(), variants={},
        host_interop_adaptation="EnableTVMFFI(False) and typed grid_x Int32 forwarding for official C AOT exporter; original Python JIT/class remains unchanged",
        original_GPU_math_AST_modified=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_121a", CUTE_DSL_DISABLE_FILE_CACHING="1",
                      CUDA_PATH=os.environ["SPARK_CUDA_ROOT"], OMP_NUM_THREADS="2")
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    os.environ["TEMP"] = os.environ["TMP"] = str(attempt / "temp")
    Path(os.environ["TEMP"]).mkdir()
    handles = []
    for folder in (Path(os.environ["SPARK_CUTE_LIB"]), Path(os.environ["CUDA_PATH"]) / "bin/x64",
                   HERE / "build-deps-x64/apache_tvm_ffi.libs", HERE / "build-deps-x64/tvm_ffi/lib"):
        if folder.is_dir(): handles.append(os.add_dll_directory(str(folder)))
    try:
        runpy.run_path(str(HERE.parent / "process_arch.py"))["require_windows_x64_process"]()
        import cuda.bindings.driver as driver
        import cuda.bindings.runtime as runtime
        def reject(name):
            def deny(*a, **kw):
                report["blocked_CUDA_calls"].append(name)
                raise RuntimeError("Build-only CUDA call rejected: " + name)
            return deny
        for module, names in ((driver, ("cuInit", "cuDeviceGet", "cuDeviceGetCount", "cuCtxCreate", "cuCtxCreate_v2", "cuCtxCreate_v3", "cuDevicePrimaryCtxRetain", "cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx", "cuLaunchKernel", "cuLaunchKernelEx")),
                              (runtime, ("cudaGetDevice", "cudaGetDeviceCount", "cudaSetDevice", "cudaMalloc", "cudaMallocAsync", "cudaLaunchKernel", "cudaLaunchKernelEx", "cudaLibraryLoadData", "cudaStreamCreate"))):
            for name in names:
                if hasattr(module, name): setattr(module, name, reject(name))
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_tensor
        from cutlass.base_dsl.compiler import DumpDir, HostTarget
        from host_entry import c_abi_entry
        original = graph.load(module_name)
        report["original_sources"] = graph.inventory()
        def dynamic(name): return cute.sym_int64(symbol=name)
        def tensor(dtype, rank, leading, name, align):
            shape = tuple(dynamic(name + "_size" + str(i)) for i in range(rank))
            stride = tuple(1 if i == leading else dynamic(name + "_stride" + str(i)) for i in range(rank))
            return make_fake_tensor(dtype, shape, stride, assumed_align=align)
        variants = (False, True) if args.checkpoint == "both" else (args.checkpoint == "on",)
        for checkpoint in variants:
            name = "original_FI_GDN_sm121a_" + ("checkpoint" if checkpoint else "plain")
            directory = attempt / name
            directory.mkdir()
            kernel = original._FullyFusedDeltaRuleSm120(True, True, True, checkpoint,
                dtype=cutlass.BFloat16, initial_state_dtype=cutlass.Float32, state_dtype=cutlass.Float32,
                checkpoint_state_dtype=cutlass.Float32, use_state_indices=False,
                cu_seqlens_dtype=graph.torch_types.int64, state_indices_dtype=None,
                checkpoint_cu_starts_dtype=graph.torch_types.int64 if checkpoint else None)
            kernel_args = (
                tensor(cutlass.BFloat16, 3, 1, "q", 16), tensor(cutlass.BFloat16, 3, 0, "k", 16),
                tensor(cutlass.BFloat16, 3, 0, "v", 16), tensor(cutlass.BFloat16, 3, 0, "o", 16),
                tensor(cutlass.Float32, 1, 0, "alpha", 16), tensor(cutlass.Float32, 1, 0, "beta", 16),
                tensor(cutlass.Float32, 4, 3, "state", 16), tensor(cutlass.Float32, 4, 3, "init", 16), None,
                tensor(cutlass.Float32, 1, 0, "checkpoints", 16) if checkpoint else None,
                tensor(cutlass.Int64, 1, 0, "checkpoint_starts", 8) if checkpoint else None,
                tensor(cutlass.Uint8, 1, 0, "tensormaps", 128), tensor(cutlass.Int64, 1, 0, "cu_seqlens", 8),
                cutlass.Float32(128 ** -0.5), cutlass.Int32(16), cutlass.Int32(16), cutlass.Int32(48),
                cutlass.Int32(48), cutlass.Int32(1), cutlass.Int32(128 if checkpoint else 1),
                cutlass.Int32(64 if checkpoint else 0), cutlass.Int32(48), driver.CUstream(0))
            options = (cute.EnableTVMFFI(False), cute.GPUArch("sm_121a"), cute.KeepPTX(True), cute.KeepCUBIN(True),
                       DumpDir(str(directory)), HostTarget("llvm -mtriple=aarch64-pc-windows-msvc"))
            report["status"] = "compiling_" + name
            (attempt / "progress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(report["status"], flush=True)
            compiled = cute.compile[options](c_abi_entry, kernel, *kernel_args, no_jit_engine=True)
            report["variants"][name] = export(compiled, directory, name)
        if "torch" in sys.modules or "triton" in sys.modules or report["blocked_CUDA_calls"]:
            raise RuntimeError("Build imported a forbidden runtime or attempted CUDA execution")
        report.update(status="passed_original_FI_GDN_CPU_SM121a_compile_and_ARM64_host_export_only",
                      numerical_accuracy_tested=False, ARM64_runtime_implemented=False)
    except BaseException as error:
        report.update(status="failed_original_FI_GDN_CPU_compile_or_export", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["Torch_imported"] = "torch" in sys.modules
        report["source_sha256"] = {str(path): sha(path) for path in (Path(__file__), HERE / "source_loader.py", HERE / "host_entry.py")}
        (attempt / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(status=report["status"], result=str(attempt / "result.json"), error=report.get("error"))), flush=True)
    if report["status"].startswith("failed"): raise SystemExit(1)


if __name__ == "__main__": main()
