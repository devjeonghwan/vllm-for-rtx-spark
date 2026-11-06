"""Pure metadata admission. Runtime tensor queries belong inside the opaque op."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    stride: tuple
    dtype: str
    device: str
    device_index: int | None
    data_ptr: int
    conjugate: bool = False
    negative: bool = False
    requires_grad: bool = False

    @classmethod
    def of(cls, tensor):
        return cls(tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
                   tensor.device.type, tensor.device.index, int(tensor.data_ptr()),
                   bool(tensor.is_conj()), bool(tensor.is_neg()), bool(tensor.requires_grad))


def admit(a, b, scale_a, scale_b, out_dtype, output_shape, *, bias_present=False):
    """Only the original provider's verified layout and quantization domain."""
    if bias_present:
        return False, "bias uses the original FP32-add-before-rounding fallback"
    tensors = (a, b, scale_a, scale_b)
    if any(t.device != "cuda" or t.device_index != a.device_index for t in tensors):
        return False, "one CUDA device required"
    if any(t.conjugate or t.negative or t.requires_grad for t in tensors):
        return False, "ordinary inference tensors required"
    if a.dtype != "torch.float8_e4m3fn" or b.dtype != a.dtype:
        return False, "FP8 E4M3FN operands required"
    if scale_a.dtype != "torch.float32" or scale_b.dtype != "torch.float32":
        return False, "FP32 scales required"
    if out_dtype not in ("torch.bfloat16", "torch.float16", "torch.float32"):
        return False, "unsupported output dtype"
    if len(a.shape) != 2 or len(b.shape) != 2 or len(a.stride) != 2 or len(b.stride) != 2:
        return False, "matrix operands required"
    m, k = a.shape
    kb, n = b.shape
    if k != kb or min(m, n, k) <= 0:
        return False, "positive compatible dimensions required"
    if (not output_shape or any(type(x) is not int or x <= 0 for x in output_shape)
            or output_shape[-1] != n or prod(output_shape[:-1]) > m):
        return False, "output shape must select valid leading token rows"
    if a.stride[1] != 1 or a.stride[0] < k:
        return False, "A requires contiguous K rows"
    row_b = b.stride[1] == 1 and b.stride[0] >= n
    column_b = b.stride[0] == 1 and b.stride[1] >= k
    if not (row_b or column_b):
        return False, "B requires row-major or transposed contiguous rows"
    if any(x <= 0 for t in (a, b) for x in t.stride):
        return False, "positive matrix strides required"
    if (scale_a.shape != (m, 1) or scale_a.stride != (1, 1)
            or scale_b.shape not in ((n,), (n, 1))
            or scale_b.stride not in ((1,), (1, 1))):
        return False, "contiguous per-token [M,1] and per-channel [N,1]/[N] scales required"
    # Triton specializes pointer divisibility by16. Equal shape/stride variants
    # must also share that specialization before reusing eager readiness in a
    # capture; offset scale views retain the original fallback.
    if any(t.data_ptr % 16 for t in tensors):
        return False, "aligned operand/scale pointers required"
    if any(sum((size - 1) * stride for size, stride in zip(t.shape, t.stride)) > 2**31 - 1
           for t in tensors):
        return False, "conservative device offset bound exceeded"
    return True, None
