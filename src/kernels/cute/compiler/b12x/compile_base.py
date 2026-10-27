"""BUILD ONLY: original b12x auto-policy programs and ARM64 host COFF archive.

Runs only in the explicitly approved task-local x64 compiler environment.
Every device function and launch class is original AST. Windows ARM64 receives
the original generated host launch; no CuTe module or compiler is needed there.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["SPARK_SOURCE_ROOT"])
SOURCE = HERE
HELPER = Path(os.environ["SPARK_COMPILER_ROOT"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    directory = Path(os.environ["SPARK_BUILD_OUTPUT"]) / "b12x-base"
    directory.mkdir(exist_ok=False)
    report = {"status": "started_original_model_archive", "GPU_executed": False,
              "Torch_imported": False, "blocked_CUDA_calls": [], "programs": [], "ranges": {}}
    os.environ.update(CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_121a", CUTE_DSL_DISABLE_FILE_CACHING="1",
                      CUDA_PATH=os.environ["SPARK_CUDA_ROOT"])
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    os.environ["TEMP"] = os.environ["TMP"] = str(directory / "temp")
    Path(os.environ["TEMP"]).mkdir()
    dll_dirs = [os.add_dll_directory(str(folder)) for folder in
                (Path(os.environ["SPARK_CUTE_LIB"]), Path(os.environ["CUDA_PATH"]) / "bin/x64")]
    try:
        runpy.run_path(str(HERE.parent / "process_arch.py"))["require_windows_x64_process"]()
        import cuda.bindings.driver as driver
        import cuda.bindings.runtime as runtime

        def deny(name):
            def call(*a, **kw):
                report["blocked_CUDA_calls"].append(name)
                raise RuntimeError("BUILD ONLY: " + name)
            return call

        for module, names in (
            (driver, ("cuInit", "cuDeviceGet", "cuDeviceGetCount", "cuCtxCreate", "cuCtxCreate_v2", "cuCtxCreate_v3",
                      "cuDevicePrimaryCtxRetain", "cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx",
                      "cuLaunchKernel", "cuLaunchKernelEx")),
            (runtime, ("cudaGetDevice", "cudaGetDeviceCount", "cudaSetDevice", "cudaMalloc", "cudaMallocAsync",
                       "cudaLaunchKernel", "cudaLaunchKernelEx", "cudaLibraryLoadData", "cudaStreamCreate")),
        ):
            for name in names:
                if hasattr(module, name):
                    setattr(module, name, deny(name))
        import cutlass
        import cutlass.cute as cute
        from cutlass.base_dsl.compiler import DumpDir, HostTarget

        sys.path.insert(0, str(SOURCE))
        from source_loader import SourceGraph
        graph = SourceGraph()
        graph.add("b12x._lib.dense_gemm", {"_DenseGemmLaunch", "DenseGemmKernel", "_DenseGemmPolicy", "_DenseSplitKReduce",
                                         "_select_default_dense_gemm_plan", "_select_mxfp8_tile_k",
                                         "_dense_gemm_policy_for", "_dense_gemm_target_occupancy"})
        graph.add("b12x._lib.utils", {"make_ptr"})
        dense, utils = graph.load("b12x._lib.dense_gemm"), graph.load("b12x._lib.utils")
        model = json.loads((HERE / "model-shapes.json").read_text())
        report["model_shapes_sha256"] = sha(HERE / "model-shapes.json")
        report["original_sources"] = graph.inventory()
        programs = {}
        launches = {}
        for geometry in model["unique_N_K"]:
            n, k = geometry["N"], geometry["K"]
            shape_key = f"{n}x{k}"
            ranges = []
            for m in range(1, 8193):
                if m <= 8:
                    config = dict(route="a16", N=n, K=k, tile=[16, 128], BK=64, mma_k=16,
                                  policy=asdict(dense._DenseGemmPolicy(True, True, False, 4, False, False)),
                                  sm_version="sm_121a", target_occupancy=1, alpha_is_one=True)
                    launch = None
                else:
                    plan = dense._select_default_dense_gemm_plan(m, n, k, 48, is_mxfp8=True, expected_m=m)
                    if plan.swap_ab:
                        raise RuntimeError("Unexpected original swapped MXFP8 model geometry")
                    bk = dense._select_mxfp8_tile_k(m, n, k, m, 48)
                    if plan.mma_tiler_mn[0] != 128:
                        bk = 128
                    policy = dense._dense_gemm_policy_for(
                        m=m, n=n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
                        mma_tiler_mn=plan.mma_tiler_mn, cluster_shape_mn=(1, 1), sm_count=48,
                        tile_k=bk, expected_m=m, generalize_mxfp8_split_k=True)
                    occupancy = dense._dense_gemm_target_occupancy(
                        n=n, k=k, l=1, ab_dtype=cutlass.Float8E4M3FN, c_dtype=cutlass.BFloat16,
                        tile_k=bk, mma_tiler_mn=plan.mma_tiler_mn, cluster_shape_mn=(1, 1), sm_count=48,
                        load_path="tma", swap_ab=False, b_tile_major=False)
                    config = dict(route="quantized", N=n, K=k, tile=list(plan.mma_tiler_mn), BK=bk, mma_k=32,
                                  policy=asdict(policy), sm_version="sm_120", target_occupancy=occupancy,
                                  alpha_is_one=False)
                    launch = None
                key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
                if ranges and ranges[-1]["program"] == key:
                    ranges[-1]["last_M"] = m
                else:
                    ranges.append({"first_M": m, "last_M": m, "program": key})
                if key not in programs:
                    programs[key] = dict(config, id=key, representative_M=m)
            report["ranges"][shape_key] = ranges
        for n in sorted({row["N"] for row in model["unique_N_K"]}):
            config = {"route": "reduce", "N": n, "slices": 4}
            key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
            programs[key] = dict(config, id=key, representative_M=1)
        report["programs"] = list(programs.values())
        report["program_count"] = len(programs)
        print(json.dumps({"phase": "original_policy_resolved", "programs": len(programs),
                          "geometries": len(report["ranges"]), "execute_build": args.execute_build}), flush=True)
        (directory / "policy-plan.json").write_text(json.dumps(report, indent=2) + "\n")
        if not args.execute_build:
            report["status"] = "passed_original_policy_compile_plan_only"
            return directory, report
        for ordinal, program in enumerate(report["programs"]):
            key, n, k = program["id"], program["N"], program.get("K")
            folder = directory / key
            folder.mkdir()
            options = (cute.GPUArch("sm_121a"), cute.KeepPTX(True), cute.KeepCUBIN(True), DumpDir(str(folder)),
                       HostTarget("llvm -mtriple=aarch64-pc-windows-msvc"))
            if program["route"] == "reduce":
                print(f"[{ordinal+1}/{len(programs)}] original split4 reduction N{n}", flush=True)
                compiled = cute.compile[options](
                    dense._DenseSplitKReduce(n, 4),
                    utils.make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                    utils.make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
                    1, driver.CUstream(0), no_jit_engine=True)
                compiled.export_to_c(str(folder), "host", "b12x_" + key)
                (folder / "device.cubin").write_bytes(compiled.__cubin__)
                ptx = compiled.__ptx__
                (folder / "device.ptx").write_text(ptx if isinstance(ptx, str) else ptx.decode())
                program["prefix"] = "b12x_" + key
                program["artifacts"] = {name: {"path": str(folder / name), "sha256": sha(folder / name),
                                                "bytes": (folder / name).stat().st_size}
                                          for name in ("host.o", "host.h", "device.cubin", "device.ptx")}
                continue
            is_a16 = program["route"] == "a16"
            policy = dense._DenseGemmPolicy(**program["policy"])
            launch = dense._DenseGemmLaunch(
                n, k, 1, 4 if is_a16 else 1, "k", "k", "n",
                cutlass.BFloat16 if is_a16 else cutlass.Float8E4M3FN,
                cutlass.Uint8 if is_a16 else cutlass.Float8E8M0FNU,
                cutlass.Float32 if is_a16 else cutlass.BFloat16,
                cutlass.Float32, 32, program["mma_k"], program["BK"], tuple(program["tile"]), (1, 1),
                policy, 48, program["sm_version"], "tma", False, False,
                alpha_is_one=program["alpha_is_one"], target_occupancy=program["target_occupancy"],
                weight_only="mxfp8" if is_a16 else None, input_k=k)
            types = (cutlass.BFloat16 if is_a16 else cutlass.Float8E4M3FN,
                     cutlass.Uint8 if is_a16 else cutlass.Float8E4M3FN,
                     cutlass.Uint8 if is_a16 else cutlass.Float8E8M0FNU,
                     cutlass.Uint8 if is_a16 else cutlass.Float8E8M0FNU,
                     cutlass.Float32 if is_a16 else cutlass.BFloat16,
                     cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, cutlass.Float8E8M0FNU, cutlass.Float32)
            pointers = [utils.make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for dtype in types]
            started = time.perf_counter()
            print(f"[{ordinal+1}/{len(programs)}] original {program['route']} N{n} K{k} tile{program['tile']} BK{program['BK']}", flush=True)
            compiled = cute.compile[options](launch, *pointers, 1, driver.CUstream(0), no_jit_engine=True)
            if compiled.engine is not None:
                raise RuntimeError("Unexpected JIT execution engine")
            compiled.export_to_c(str(folder), "host", "b12x_" + key)
            (folder / "device.cubin").write_bytes(compiled.__cubin__)
            ptx = compiled.__ptx__
            (folder / "device.ptx").write_text(ptx if isinstance(ptx, str) else ptx.decode())
            program["prefix"] = "b12x_" + key
            program["build_wall_seconds"] = time.perf_counter() - started
            program["artifacts"] = {name: {"path": str(folder / name), "sha256": sha(folder / name),
                                            "bytes": (folder / name).stat().st_size}
                                      for name in ("host.o", "host.h", "device.cubin", "device.ptx")}
            if (folder / "host.o").read_bytes()[:2] != b"\x64\xaa":
                raise RuntimeError("Original generated host object is not native ARM64 COFF")
            (directory / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed_original_model_auto_dense_GPU_and_ARM64_host_artifact_archive_only"
        return directory, report
    except BaseException as error:
        report.update(status="failed_original_model_archive", error=repr(error), traceback=traceback.format_exc())
        return directory, report
    finally:
        report["Torch_imported"] = "torch" in sys.modules
        report["source_sha256"] = sha(__file__)
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        dll_dirs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-build", action="store_true")
    args = parser.parse_args()
    folder, result = build(args)
    print(json.dumps({"status": result["status"], "directory": str(folder), "error": result.get("error")}), flush=True)
    raise SystemExit(1 if result["status"].startswith("failed") else 0)
