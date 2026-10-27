"""ARM64 C-host boundary for the two unchanged original FlashInfer GDN kernels.

The caller is the original delta_rule_prefill_dsl host function. All recurrence,
TMA construction, occupancy selection and launches remain in original COFF.
Only original generated-header objects are packed here; no tensor data is read.
"""
import ctypes as C
import functools
from pathlib import Path
from . import assets

HERE = Path(__file__).resolve().parent
FAILED_TENSOR_OWNERS = []
_modules = {}


class Tensor3(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int64 * 3),
                ("dynamic_strides", C.c_int64 * 2)]


class Tensor4(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int64 * 4),
                ("dynamic_strides", C.c_int64 * 3)]


class Tensor1(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dynamic_shapes", C.c_int64 * 1)]


def pack_tensor(value, rank, fixed_stride_dimension):
    """Generated header order: ptr, all i64 shapes, non-unit i64 strides."""
    shape, strides = tuple(value.shape), tuple(value.stride())
    if len(shape) != rank or len(strides) != rank or strides[fixed_stride_dimension] != 1:
        raise ValueError("Tensor layout differs from the original compiled layout")
    pointer = int(value.data_ptr())
    if pointer <= 0:
        raise ValueError("Original GDN tensors must have real device storage")
    cls = {1: Tensor1, 3: Tensor3, 4: Tensor4}[rank]
    if rank == 1:
        return cls(pointer, (C.c_int64 * 1)(*shape))
    remaining = [stride for i, stride in enumerate(strides) if i != fixed_stride_dimension]
    return cls(pointer, (C.c_int64 * rank)(*shape), (C.c_int64 * (rank - 1))(*remaining))


class NativeKernel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def __call__(self, q, k, v, o, alpha, beta, state, initial_state, state_indices,
                 checkpoints, checkpoint_starts, tensormaps, cu_seqlens, scale,
                 q_heads, k_heads, v_heads, state_heads, num_seqs, total_checkpoints,
                 checkpoint_every, grid_x, stream):
        import torch
        from original_cute_coff_runtime import load_module

        if state_indices is not None or (q_heads, k_heads, v_heads, state_heads) != (16, 16, 48, 48):
            raise NotImplementedError("AOT archive is for original Qwen GDN 16/16/48 heads, compact states")
        if not 1 <= num_seqs <= 16 or grid_x != num_seqs * state_heads:
            raise ValueError("Original GDN sequence/grid scalar domain changed")
        if (checkpoint_every > 0) != self.checkpoint:
            raise ValueError("Original checkpoint dispatch key differs from call")
        actual_stream = torch.cuda.current_stream(q.device)
        if int(stream) != actual_stream.cuda_stream:
            raise RuntimeError("Use the actual original caller's current CUDA stream")
        owners = [q, k, v, o, alpha, beta, state, initial_state, checkpoints,
                  checkpoint_starts, tensormaps, cu_seqlens]
        key = (self.checkpoint, q.device.index)
        module_entry = _modules.get(key)
        if module_entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up both original GDN AOT variants before capture")
            properties = torch.cuda.get_device_properties(q.device)
            if (properties.major, properties.minor, properties.multi_processor_count) != (12, 1, 48):
                raise NotImplementedError("This original GDN archive targets SM121a with 48 SMs")
            assets.verify()
            configuration = assets.metadata("gdn")
            configuration["library"] = str(assets.resolve(configuration["library"]))
            entry = configuration["variants"]["checkpoint" if self.checkpoint else "plain"]
            module = load_module(configuration["library"],
                                 init_symbol=entry["init_symbol"], load_symbol=entry["load_symbol"],
                                 stream=actual_stream, invoke_symbols={entry["invoke_symbol"]: entry["inputs"]})
            module_entry = module, entry["invoke_symbol"]
            _modules[key] = module_entry
        arguments = [pack_tensor(q, 3, 1), pack_tensor(k, 3, 0),
                     pack_tensor(v, 3, 0), pack_tensor(o, 3, 0),
                     pack_tensor(alpha, 1, 0), pack_tensor(beta, 1, 0),
                     pack_tensor(state, 4, 3), pack_tensor(initial_state, 4, 3)]
        if self.checkpoint:
            arguments.extend([pack_tensor(checkpoints, 1, 0), pack_tensor(checkpoint_starts, 1, 0)])
        arguments.extend([pack_tensor(tensormaps, 1, 0), pack_tensor(cu_seqlens, 1, 0), C.c_float(scale)])
        arguments.extend(C.c_int32(x) for x in (q_heads, k_heads, v_heads, state_heads,
                                              num_seqs, total_checkpoints, checkpoint_every, grid_x))
        arguments.append(C.c_void_p(int(stream)))
        try:
            module_entry[0].invoke(module_entry[1], [C.addressof(argument) for argument in arguments])
        except BaseException:
            # Preserve original live tensor storage if a native launch partially
            # enqueued before failing. Final original teardown drains consumers.
            FAILED_TENSOR_OWNERS.append((owners, module_entry))
            raise


@functools.cache
def get_prefill_kernel(needs_alpha, needs_beta, needs_init_state, needs_checkpointing,
                       kernel_dtype, *, initial_state_dtype, state_dtype, checkpoint_state_dtype,
                       use_state_indices, cu_seqlens_dtype, state_indices_dtype,
                       checkpoint_cu_starts_dtype, state_inner_strides, init_state_inner_strides):
    import torch
    if ((needs_alpha, needs_beta, needs_init_state) != (True, True, True)
            or kernel_dtype != "original_BFloat16" or use_state_indices
            or (initial_state_dtype, state_dtype, checkpoint_state_dtype) != (torch.float32,) * 3
            or cu_seqlens_dtype != torch.int64 or state_indices_dtype is not None
            or checkpoint_cu_starts_dtype != (torch.int64 if needs_checkpointing else None)
            or state_inner_strides is not None or init_state_inner_strides is not None):
        raise NotImplementedError("GDN AOT archive lacks this original specialization; no fallback math")
    return NativeKernel(needs_checkpointing)


def cached_native(kernel, compile_options):
    if compile_options != ("original_ARM64_C_ABI_AOT",):
        raise ValueError("Unexpected original GDN compile-key boundary")
    return kernel
