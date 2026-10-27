"""Windows ARM64 import/host-loading boundary for ORIGINAL b12x dense code.

Original Torch packing, precision policy, Triton activation quantizer, scratch
layout, opaque Torch operators and wrappers execute their original AST. Only
the unavailable CuTe host factory is replaced by its own AOT-exported ARM64
COFF functions. No fake cutlass module, new GPU kernel or alternate arithmetic.
"""

import ast
import builtins
import ctypes as C
import hashlib
import importlib
import importlib.util
import inspect
import math
import os
import platform
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from . import assets

ORIGINAL = Path(__file__).with_name("device") / "b12x"
NATIVE_DLL = assets.resolve("lib/b12x.dll")
NATIVE_LOADER = Path(__file__).with_name("coff_runtime.py")
_api = None
_programs = {}
_prepared = {}
_ranges = {}
_ranges_alpha_one = {}
_reductions = {}
_quantizers = {}
_loaded_functions = {}
_source_records = {}
_alpha_one = None
_empty_output = None
_original_dense_signature = None


def _dense_packet(lhs, rhs, out=None, *, alpha=None, expected_m=None, stream=None, **kwargs):
    """Bind the complete ORIGINAL schema; reject uncompiled host selections.

    This is admission for a bounded original compiled archive, not a new
    selector. All original selectors ran in the artifact producer. Explicit
    caller modifiers must agree with that actual full compile key.
    """
    global _original_dense_signature
    if _original_dense_signature is None:
        path = ORIGINAL / '_lib/dense_gemm.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dense_gemm')
        node.body = [ast.Pass()]
        node.decorator_list = []
        namespace = {}
        schema = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(schema), str(path), 'exec'), namespace)  # noqa: S102 - original schema only
        _original_dense_signature = inspect.signature(namespace['dense_gemm'])
    bound = _original_dense_signature.bind(lhs, rhs, out, alpha=alpha, expected_m=expected_m, stream=stream, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    a, _ = lhs
    b, _ = rhs
    m, k, groups = a.shape
    n = b.shape[0]
    if groups != 1 or tuple(b.shape[1:]) != (k, 1) or expected_m != m:
        raise ValueError('Original archive requires group1 and the actual original live-M hint')
    program = _program(n, k, m, 'quantized', alpha_one=alpha is None)
    required = {'ab_dtype': 'float8_e4m3fn', 'sf_dtype': 'float8_e8m0fnu', 'c_dtype': 'bfloat16', 'sf_vec_size': 32,
                'cluster_shape_mn': (1, 1), 'sfb_k_replicated': False, 'rhs_values_tiled': None, '_quantized_c': None,
                'a_preexpanded': False, 'b_preexpanded': False, 'b_packed': False, 'a_fmt': None, 'b_fmt': None,
                'x_bf16': None, 'w_gscale': None, 'plain_fp8': False, 'row_scale': None, 'block_fp8': False}
    for name, expected in required.items():
        actual = values[name]
        if actual != expected:
            raise ValueError(f'Original dense archive has no specialization for {name}={actual!r}')
    choices = {'sm_count': (None, 48), 'mma_tiler_mn': (None, tuple(program['tile'])), 'load_path': (None, 'tma'),
               'swap_ab': (None, False), 'alpha_dtype': (None, 'float32'), '_tile_k_override': (None, program['BK']),
               '_split_k_slices_override': (None, program['policy']['split_k_slices']),
               '_large_m_unroll_override': (None, program['policy']['large_m_unroll']),
               '_target_occupancy_override': (None, program['target_occupancy'])}
    for name, allowed in choices.items():
        if values[name] not in allowed:
            raise ValueError(f'Original dense caller {name}={values[name]!r} differs from compiled key {allowed!r}')
    # The original host consults this caller buffer only when slices > 1.
    # Every archived quantized specialization has exactly one slice.
    if program['policy']['split_k_slices'] != 1:
        raise ValueError('Quantized split-K workspace must not be silently discarded')
    return program


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    """Source selection is inspectable without importing Torch or CuTe."""
    return {
        "gemm/_shared/wo_mxfp8.py": ["MXFP8Rows", "_check_gpu_tensor", "_check_mxfp8_k", "pack_mxfp8_scales_for_dense_gemm",
                                    "empty_mxfp8_rows_bases", "mxfp8_rows_from_bases"],
        "gemm/blockscaled/_linear.py": ["MXFP8LinearWeight", "TensorFP8LinearWeight", "_align_up", "_pad_k", "_scale_rows_to_u8",
                                       "_pad_scale_rows_k", "pack_mxfp8_linear_weight", "pack_weight", "_validate_bias", "blockscaled_mm", "prewarm",
                                       "_source_2d", "_output_dtype_name", "mxfp8_linear", "_packed_mxfp8_op", "_packed_mxfp8_fake"],
        "gemm/blockscaled/_a16.py": "ALL original class/function definitions",
        "gemm/blockscaled/_ops.py": "ALL original function definitions, including custom-op/fake registration",
        "_lib/intrinsics.py": ["align_up", "as_grouped_scale_view", "as_grouped_scale_view_mx"],
        "_lib/utils.py": ["cuda_stream_to_int"],
        "gemm/blockscaled/_policy.py": "Original module imported unchanged",
        "gemm/blockscaled/_quantize.py": "Original module imported unchanged",
        "gemm/_shared/block_fp8.py": ["_align_up", "_physical_mxfp8_k", "_run_block_fp8_quant_kernel",
                                      "_quantize_block_fp8_linear_input_for_immediate_gemm"],
        "_lib/quant/mxfp8_rows.py": ["quantize_mxfp8_rows_cute"],
    }


def _prepare_archive():
    global _ranges, _ranges_alpha_one
    assets.verify()
    archive = assets.metadata("b12x")
    for program in archive["programs"]:
        _programs[program["id"]] = program
        if program["route"] == "reduce":
            _reductions[program["N"]] = program["id"]
        elif program["route"] == "quantize_rows":
            _quantizers[program["K"]] = program["id"]
    _ranges = archive["ranges"]
    _ranges_alpha_one = archive["ranges_alpha_one"]


def _preload_shape(n, k, device):
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Original AOT code libraries must be preloaded while preparing weights")
    properties = torch.cuda.get_device_properties(device)
    if (int(properties.major), int(properties.minor), int(properties.multi_processor_count)) != (12, 1, 48):
        raise RuntimeError('Original AOT archive was compiled for actual SM121 / 48 SM only')
    shape = f"{n}x{k}"
    if shape not in _ranges:
        raise ValueError("Original checkpoint geometry is absent from this compiled archive: " + shape)
    _alpha_one(torch.device("cuda", device))
    if device in _prepared and shape in _prepared[device]:
        return
    spec = importlib.util.spec_from_file_location("original_cute_coff_runtime", NATIVE_LOADER)
    runtime = sys.modules.get(spec.name)
    if runtime is None:
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        spec.loader.exec_module(runtime)
    stream = torch.cuda.current_stream(device)
    keys = {record["program"] for record in [*_ranges[shape], *_ranges_alpha_one[shape]]}
    keys.add(_reductions[n])
    keys.add(_quantizers[k])
    for key in sorted(keys):
        if (device, key) in _loaded_functions:
            continue
        program = _programs[key]
        prefix = program["prefix"]
        module = runtime.load_module(
            NATIVE_DLL,
            init_symbol=prefix + "_cuda_init", load_symbol=prefix + "_cuda_load",
            stream=stream, invoke_symbols={program["invoke_symbol"]:
                                          4 if program["route"] == "reduce" else 8 if program["route"] == "quantize_rows" else 11})
        _loaded_functions[device, key] = module
    _prepared.setdefault(device, set()).add(shape)


def _program(n, k, m, route, *, alpha_one=False):
    shape = f"{n}x{k}"
    for record in (_ranges_alpha_one if alpha_one else _ranges).get(shape, ()):
        if record["first_M"] <= m <= record["last_M"]:
            program = _programs[record["program"]]
            if program["route"] != route:
                raise RuntimeError("Original runtime precision policy and compiled archive disagree")
            return program
    raise RuntimeError(f"Original AOT archive has no policy entry for M={m}, N={n}, K={k}")


def _invoke(program, values, device):
    module = _loaded_functions.get((device, program["id"]))
    if module is None:
        raise RuntimeError("Cold original native program was not preloaded before serving")
    module.invoke(program["invoke_symbol"], [C.addressof(value) for value in values])


def _dense_a16(source, weight, scales, alpha, out, *, recipe, input_k, config, reciprocal=False, stream=None):
    if recipe != "mxfp8" or reciprocal or tuple(config) != (128, 64, 4):
        raise RuntimeError("This original archive does not contain that explicit A16/profile configuration")
    import torch
    n, k = weight.shape
    if k != input_k:
        raise ValueError("Original model has no padded A16 input geometry")
    m = source.numel() // k
    program = _program(n, k, m, "a16")
    raw_stream = torch.cuda.current_stream(source.device).cuda_stream if stream is None else int(getattr(stream, "cuda_stream", stream))
    alpha_ptr = _alpha_one(source.device).data_ptr() if alpha is None else alpha.data_ptr()
    pointers = [source.data_ptr(), weight.data_ptr(), scales.data_ptr(), scales.data_ptr(), out.data_ptr(),
                16, 16, 16, alpha_ptr]
    values = [*(C.c_void_p(ptr) for ptr in pointers), C.c_int32(m), C.c_void_p(raw_stream)]
    _invoke(program, values, source.device.index)
    return out


def _dense_reduce(partials, out, *, n, m, slices, stream=None):
    import torch
    if slices != 4:
        raise ValueError("Original archive contains four-way A16 reduction")
    program = _programs[_reductions[n]]
    raw_stream = torch.cuda.current_stream(out.device).cuda_stream if stream is None else int(getattr(stream, "cuda_stream", stream))
    values = [C.c_void_p(partials.data_ptr()), C.c_void_p(out.data_ptr()), C.c_int32(m), C.c_void_p(raw_stream)]
    _invoke(program, values, out.device.index)


def _dense_quantized(lhs, rhs, out=None, *, alpha=None, expected_m=None, stream=None, **kwargs):
    import torch
    a, sfa = lhs
    b, sfb = rhs
    m, k, groups = a.shape
    n = b.shape[0]
    if groups != 1 or expected_m not in (None, m):
        raise ValueError("Native original model archive expects one group and original live-M hint")
    program = _dense_packet(lhs, rhs, out, alpha=alpha, expected_m=expected_m, stream=stream, **kwargs)
    if out is None:
        out = _empty_output(m, n, groups, dtype=torch.bfloat16, device=a.device)
    if alpha is None:
        alpha = _alpha_one(a.device)
    raw_stream = torch.cuda.current_stream(a.device).cuda_stream if stream is None else int(getattr(stream, "cuda_stream", stream))
    pointers = [a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(), out.data_ptr(),
                16, 16, 16, alpha.data_ptr()]
    values = [*(C.c_void_p(ptr) for ptr in pointers), C.c_int32(m), C.c_void_p(raw_stream)]
    _invoke(program, values, a.device.index)
    return out


@cache
def _get_native_rows_quant(k, source_dtype, subgroup_width, threads, value_order, min_amax=0.0):
    import torch
    if (source_dtype != torch.bfloat16 or subgroup_width != 4 or threads != 256
            or value_order != "linear" or min_amax != 0.0):
        raise RuntimeError("Original model archive does not include that rows quantizer specialization")
    program = _programs[_quantizers[k]]
    def launch_tensors(source, values, scale_rows, scale_mma):
        # Identical original nested launch_tensors grid calculation; only
        # original make_ptr/CuTe invocation becomes generated COFF C ABI.
        groups_per_warp = 32 // subgroup_width
        total_tasks = int(source.shape[0]) * ((k // 32 + groups_per_warp - 1) // groups_per_warp)
        warps_per_cta = threads // 32
        natural_grid = max(1, (total_tasks + warps_per_cta - 1) // warps_per_cta)
        sm_count = torch.cuda.get_device_properties(source.device).multi_processor_count
        grid_x = min(natural_grid, sm_count * 4)
        stream = torch.cuda.current_stream(source.device).cuda_stream
        arguments = [*(C.c_void_p(t.data_ptr()) for t in (source, values, scale_rows, scale_mma)),
                     C.c_int32(source.shape[0]), C.c_int32(source.shape[1]), C.c_int32(grid_x), C.c_void_p(stream)]
        _invoke(program, arguments, source.device.index)
    return launch_tensors


def _load_original_host(torch):
    """Execute original host definitions; redirect ONLY their import boundary."""
    global _alpha_one, _empty_output
    import triton
    if ("b12x.gemm.blockscaled._ops" in sys.modules
            or hasattr(torch.ops.b12x, "blockscaled_bf16") or hasattr(torch.ops.b12x, "blockscaled_bf16_out")):
        raise RuntimeError("Another b12x custom-op owner was initialized before the Windows original-host boundary")
    modules = {}
    native_dense = ModuleType("original_b12x_native_dense_host")
    native_dense.dense_gemm_a16 = _dense_a16
    native_dense.dense_gemm_a16_reduce = _dense_reduce
    native_dense.dense_gemm = _dense_quantized
    path = ORIGINAL / "_lib/dense_gemm.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    alpha_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_cached_alpha_one", "_empty_dense_gemm_output")]
    alpha_body = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *alpha_nodes], type_ignores=[])
    native_dense.__dict__.update(torch=torch, _ALPHA_ONE_CACHE={})
    exec(compile(ast.fix_missing_locations(alpha_body), str(path), "exec"), native_dense.__dict__)  # noqa: S102 - original pinned AST
    _alpha_one = native_dense._cached_alpha_one
    _empty_output = native_dense._empty_dense_gemm_output
    modules["b12x._lib.dense_gemm"] = native_dense
    selections = source_inventory()
    paths = {
        "b12x.gemm._shared.wo_mxfp8": "gemm/_shared/wo_mxfp8.py",
        "b12x.gemm.blockscaled._linear": "gemm/blockscaled/_linear.py",
        "b12x.gemm.blockscaled._a16": "gemm/blockscaled/_a16.py",
        "b12x.gemm.blockscaled._ops": "gemm/blockscaled/_ops.py",
        "b12x._lib.intrinsics": "_lib/intrinsics.py", "b12x._lib.utils": "_lib/utils.py",
        "b12x.gemm._shared.block_fp8": "gemm/_shared/block_fp8.py",
        "b12x._lib.quant.mxfp8_rows": "_lib/quant/mxfp8_rows.py",
    }
    for fullname, relative in paths.items():
        module = ModuleType("original_host_ast_" + fullname.replace(".", "_"))
        module.__file__ = str(ORIGINAL / relative)
        module.__package__ = fullname.rsplit(".", 1)[0]
        sys.modules[module.__name__] = module
        if fullname == "b12x.gemm.blockscaled._ops":
            # All original _ops definitions are executed below. Preserve that
            # exact single registration owner if a later normal import occurs.
            sys.modules[fullname] = module
        modules[fullname] = module

    original_import = builtins.__import__
    def original_host_import(name, globals=None, locals=None, fromlist=(), level=0):
        fullname = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
        if fullname in modules:
            if not fromlist:
                raise ImportError("Unexpected unqualified original host import: " + fullname)
            return modules[fullname]
        return original_import(name, globals, locals, fromlist, level)

    common = {"torch": torch, "triton": triton, "dataclass": dataclass, "math": math, "nullcontext": nullcontext,
              "MXFP8_SCALE_VEC_SIZE": 32, "MXFP8_SCALE_ROW_TILE": 128, "MXFP8_SCALE_K_TILE": 4, "SF_VEC_SIZE": 16}
    for fullname in paths:
        namespace = modules[fullname].__dict__
        namespace.update(common)
        namespace["__builtins__"] = {**vars(builtins), "__import__": original_host_import}
    wo = modules["b12x.gemm._shared.wo_mxfp8"]
    linear = modules["b12x.gemm.blockscaled._linear"]
    a16 = modules["b12x.gemm.blockscaled._a16"]
    order = ("b12x.gemm._shared.wo_mxfp8", "b12x._lib.intrinsics", "b12x._lib.utils",
             "b12x._lib.quant.mxfp8_rows", "b12x.gemm._shared.block_fp8",
             "b12x.gemm.blockscaled._a16", "b12x.gemm.blockscaled._linear", "b12x.gemm.blockscaled._ops")
    for fullname in order:
        relative = paths[fullname]
        path = ORIGINAL / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        selection = selections[relative]
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and (isinstance(selection, str) or node.name in selection)]
        module = modules[fullname]
        if fullname.endswith("._a16"):
            module.__dict__["kernels"] = importlib.import_module("b12x.gemm.blockscaled._quantize")
        if fullname == "b12x._lib.quant.mxfp8_rows":
            module.__dict__.update(_THREADS=256, _WARP_SUBGROUP_WIDTH=4,
                                  _get_compiled_mxfp8_rows_quant=_get_native_rows_quant)
        if fullname == "b12x.gemm._shared.block_fp8":
            module.__dict__.update(MXFP8Rows=wo.MXFP8Rows,
                                  empty_mxfp8_rows_bases=wo.empty_mxfp8_rows_bases,
                                  mxfp8_rows_from_bases=wo.mxfp8_rows_from_bases,
                                  quantize_mxfp8_rows_cute=modules["b12x._lib.quant.mxfp8_rows"].quantize_mxfp8_rows_cute)
        if fullname.endswith("._linear"):
            module.__dict__.update(MXFP8Rows=wo.MXFP8Rows, _check_gpu_tensor=wo._check_gpu_tensor,
                                  pack_mxfp8_scales_for_dense_gemm=wo.pack_mxfp8_scales_for_dense_gemm,
                                  NVFP4LinearWeight=a16.NVFP4LinearWeight,
                                  pack_nvfp4_weight=a16.pack_nvfp4_weight, workspace_size=a16.workspace_size,
                                  dense_gemm=_dense_quantized,
                                  cuda_stream_to_int=modules["b12x._lib.utils"].cuda_stream_to_int)
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes]
        selected = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(selected, str(path), "exec"), module.__dict__)  # noqa: S102 - original pinned AST
        _source_records[relative] = {"path": str(path), "sha256": _sha(path),
                                    "definitions": [node.name for node in nodes], "original_AST_modified": False}
    return linear


def _is_supported(device=None):
    import torch
    return (os.name == "nt" and platform.machine().lower() in ("arm64", "aarch64")
            and torch.cuda.get_device_capability(device) == (12, 1) and NATIVE_DLL.is_file())


def get_b12x_mxfp8_linear():
    global _api
    if os.environ.get("MBX_B12X_AOT", "0") != "1":
        from ...runtime.vllm import get_adapter
        return get_adapter().get_b12x_mxfp8_linear()
    if _api is not None:
        return _api
    import torch
    _prepare_archive()
    package = importlib.import_module('b12x')
    if Path(package.__file__).resolve() != (ORIGINAL / '__init__.py').resolve():
        raise RuntimeError('Original b12x policy package with exact image data must precede other package roots')
    original = _load_original_host(torch)
    api = ModuleType("original_b12x_AOT_import_boundary")
    def pack_weight(weight, scale, **kwargs):
        packed = original.pack_weight(weight, scale, **kwargs)
        _preload_shape(packed.out_features, packed.padded_in_features, weight.device.index)
        return packed
    api.pack_weight = pack_weight
    api.mm = original.blockscaled_mm
    api.prewarm = original.prewarm
    api.workspace_size = original.workspace_size
    api.Weight = original.MXFP8LinearWeight
    api.is_supported = _is_supported
    api.original_host_sources = _source_records
    _api = api
    return api
