"""Original install(module) host boundary for the native original-CSF Torch op.

No Graph executor, scratch owner, native kernel implementation
or independent launch policy is supplied by this Python module.
"""

import ast
import functools
import importlib
import inspect
import os
from pathlib import Path
import struct
import sys
import textwrap

from ....runtime import source_root
from ....kernels.csf import resolve_csf_provider
ROOT = source_root() / "kernels" / "csf" / "original"
ORIGINAL_PREP = ROOT / "marlin_utils_fp4.py"
COMPRESS_SOURCE = ROOT / "compress.py"
_installed = None
_native = None
torch = None
_templates = {}
_prepared_metadata = []


def _check_pe(path):
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError("The host library must be a native Windows PE DLL")
        stream.seek(struct.unpack_from("<I", header, 60)[0])
        if stream.read(6) != b"PE\0\0\x64\xaa":
            raise ValueError("The host DLL must be ARM64 PE, not Linux or emulated x64")


def _load_native():
    global _native
    if _native is not None:
        return _native
    torch = importlib.import_module("torch")
    selected = resolve_csf_provider(os.environ.get("MBX_MARLIN_CSF_PROVIDER", "cubin"))
    library = selected["library"]
    _check_pe(library)
    if "cubin" in selected:
        os.environ["MBX_MARLIN_CSF_CUBIN"] = str(selected["cubin"])
    torch.ops.load_library(str(library))
    provider = selected["provider"]
    _native = getattr(getattr(torch.ops, provider.op_namespace), provider.op_name)
    return _native


def _check_compress_source():
    expected = ast.parse(COMPRESS_SOURCE.read_text(encoding="utf-8")).body[0]
    actual = ast.parse(textwrap.dedent(inspect.getsource(compress))).body[0]
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("Preserve the reconstructed original compression AST")


def compress(scales):
    # 0x16ff0..0x17220: contiguous, uint8 view, (-1,16) reshape, int16 cast.
    x = scales.contiguous().view(torch.uint8).reshape(-1, 16).to(torch.int16)
    # 0x17274..0x172d4 and 0x17304..0x17360.
    mn = x.min(1).values
    # Subtract helper 0xb660 -> PyNumber_Subtract; GT opcode4 at 0x18a74;
    # EQ opcode2 at 0x190a8; OR helper 0xb434 -> PyNumber_Or.
    raw = ((x.max(1).values - mn) > 15) | (mn == 255)
    # 0x177b4..0x17948: [:,None], subtract, clamp(0,15), uint8 cast.
    off = (x - mn[:, None]).clamp(0, 15).to(torch.uint8)
    # 0x17990..0x17ae8: even/odd slices, shift4, OR, contiguous.
    packed = (off[:, 0::2] | (off[:, 1::2] << 4)).contiguous()
    # 0x17b2c..0x17ca4: int32 view, (-1,2) reshape, clone.
    nib = packed.view(torch.int32).reshape(-1, 2).clone()
    # 0x17ce8..0x17dc4; only the function invocation uses the tuple/call helper.
    ridx = torch.nonzero(raw).flatten()
    # 0x17e08..0x18068: arange dtype and device explicitly from x.device.
    nib[ridx, 0] = torch.arange(ridx.numel(), device=x.device, dtype=torch.int32)
    # 0x18094..0x18110.
    nib[ridx, 1] = 0
    # 0x18154..0x183d4: where(raw, full_like(mn,255), mn), uint8 cast.
    base = torch.where(raw, torch.full_like(mn, 255), mn).to(torch.uint8)
    # 0x183dc..0x186cc: gather, uint8 cast, contiguous, int32 view, (-1,4).
    rawv = x[ridx].to(torch.uint8).contiguous().view(torch.int32).reshape(-1, 4)
    # 0x186e0..0x18750, 0x19e30..0x19fac: empty raw plane has one zero row.
    if rawv.numel() == 0:
        rawv = torch.zeros((1, 4), dtype=torch.int32, device=x.device)
    # 0x18754..0x18850: three contiguous calls, then a tuple of three tensors.
    return base.contiguous(), nib.contiguous(), rawv.contiguous()


def _plane_metadata(label, value):
    storage = value.untyped_storage()
    return dict(name=label, dtype=str(value.dtype), device=str(value.device),
                shape=list(value.shape), stride=list(value.stride()),
                numel=value.numel(), element_size=value.element_size(),
                bytes=value.numel() * value.element_size(),
                tensor_data_ptr=value.data_ptr(), storage_data_ptr=storage.data_ptr(),
                storage_offset=value.storage_offset(), storage_nbytes=storage.nbytes())


def _pack(value, torch):
    if (value.element_size() != 1
            or tuple(value.shape) not in ((512, 160, 1280), (512, 40, 2560))):
        raise ValueError("Only the original model's prepared E512 FP8 scales are admitted")
    if not value.is_cuda or not value.is_contiguous():
        raise ValueError("Original GPU compression requires contiguous CUDA prepared scales")
    planes = compress(value)
    if not isinstance(planes, tuple) or len(planes) != 3:
        raise RuntimeError("Original compression must return three planes")
    base, nib, raw = planes
    vectors = value.numel() // 16
    if (base.dtype != torch.uint8 or tuple(base.shape) != (vectors,)
            or nib.dtype != torch.int32 or tuple(nib.shape) != (vectors, 2)
            or raw.dtype != torch.int32 or raw.ndim != 2
            or raw.shape[0] < 1 or raw.shape[1] != 4):
        raise RuntimeError("Original base/nib/raw dtype or shape contract changed")
    if any(plane.device != value.device or not plane.is_contiguous() for plane in planes):
        raise RuntimeError("Original compression planes must stay on the input GPU")
    metadata = [_plane_metadata(label, plane)
                for label, plane in zip(("base", "nib", "raw"), planes, strict=True)]
    if len({item["storage_data_ptr"] for item in metadata}) != 3:
        raise RuntimeError("Preserve the original three separate plane storages")
    tail = tuple(value.shape[1:])
    key = (tail, value.dtype, value.device)
    if key not in _templates:
        _templates[key] = torch.zeros((1,) + tail, dtype=value.dtype, device=value.device)
    carrier = _templates[key].view((1,) + tail)
    carrier._mbx_csf = planes  # Original replace_parameter copies these strong owners.
    carrier._mbx_native_scale_metadata = dict(
        prepared_shape=list(value.shape),
        prepared_dtype=str(value.dtype), prepared_device=str(value.device),
        prepared_bytes=value.numel() * value.element_size(), planes=metadata,
        storage_bytes=sum(item["bytes"] for item in metadata),
        original_plane_storage_count=3, raw_empty_sentinel_bytes=16,
        production_cpu_scale_copy=False, production_scale_data_hash=False,
        production_numpy_encoder=False, production_CPU_roundtrip_validation=False,
        weight_requantization=False,
        compression_implementation="recovered_original_device_Torch_body",
    )
    _prepared_metadata.append(dict(carrier._mbx_native_scale_metadata))
    return carrier


def install(module, *, ops=None):
    """Keep original prep/oracle/experts; replace the original three host hooks."""
    global _installed, torch
    if os.environ.get("MBX_MARLIN_CSF", "0") != "1":
        return
    if _installed is not None:
        if _installed[0] is not module:
            raise RuntimeError("Do not install the original binding into another module")
        return
    original_prep = module.prepare_nvfp4_moe_layer_for_marlin
    tree = ast.parse(ORIGINAL_PREP.read_text(encoding="utf-8"))
    expected = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "prepare_nvfp4_moe_layer_for_marlin")
    actual = ast.parse(textwrap.dedent(inspect.getsource(original_prep))).body[0]
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("Reuse the exact original preparation function body")
    _check_compress_source()
    torch = importlib.import_module("torch")
    native = _load_native()
    if ops is None:
        ops = importlib.import_module("vllm._custom_ops")
    original_gemm = ops.moe_wna16_marlin_gemm

    @functools.wraps(original_prep)
    def prepare(*args, **kwargs):
        returned = original_prep(*args, **kwargs)
        if not isinstance(returned, tuple) or len(returned) != 6:
            raise RuntimeError("Preserve the original six preparation returns")
        a, sa, ga, b, sb, gb = returned
        return a, _pack(sa, torch), ga, b, _pack(sb, torch), gb

    @functools.wraps(original_gemm)
    def gemm(a, c_or_none, b_q_weight, b_bias_or_none, b_scales, a_scales,
             global_scale, b_zeros_or_none, workspace, sorted_token_ids,
             expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
             top_k, mul_topk_weights, b_q_type, size_m, size_n, size_k,
             use_atomic_add, use_fp32_reduce, is_zp_float,
             thread_k=-1, thread_n=-1, blocks_per_sm=-1):
        values = (a,c_or_none,b_q_weight,b_bias_or_none,b_scales,a_scales,
                  global_scale,b_zeros_or_none,workspace,sorted_token_ids,
                  expert_ids,num_tokens_past_padded,topk_weights,moe_block_size,
                  top_k,mul_topk_weights,b_q_type,size_m,size_n,size_k,
                  use_atomic_add,use_fp32_reduce,is_zp_float,thread_k,thread_n,blocks_per_sm)
        planes = getattr(b_scales, "_mbx_csf", None)
        if planes is None:
            if b_scales.ndim == 3 and b_scales.shape[0] == 1 and any(tuple(b_scales.shape[1:]) == key[0] for key in _templates):
                raise RuntimeError("A compressed carrier lost its original scale attributes")
            return original_gemm(*values)
        if not isinstance(planes, tuple) or len(planes) != 3:
            raise RuntimeError("Original compressed scales require all three planes")
        return native(*values[:16], b_q_type.id, *values[17:], *planes)

    oracle = sys.modules.get("vllm.model_executor.layers.fused_moe.oracle.nvfp4")
    if oracle is not None and hasattr(oracle, "prepare_nvfp4_moe_layer_for_marlin"):
        if oracle.prepare_nvfp4_moe_layer_for_marlin is not original_prep:
            raise RuntimeError("Do not overwrite a foreign preparation alias")
        oracle.prepare_nvfp4_moe_layer_for_marlin = prepare
    module.prepare_nvfp4_moe_layer_for_marlin = prepare
    ops.moe_wna16_marlin_gemm = gemm
    _installed = (module, original_prep, ops, original_gemm)


def status():
    return dict(installed=_installed is not None, native_op_loaded=_native is not None,
                prepared_scales=list(_prepared_metadata), graph_executor_added=False,
                graph_lifecycle_hooks_added=False, native_scratch_owned_by_original_Torch_op=True,
                compression_implementation="recovered_original_device_Torch_body",
                production_cpu_scale_copy=False, production_scale_data_hash=False,
                production_numpy_encoder=False,
                original_GPU_allocator_peak_equivalence="not established")
