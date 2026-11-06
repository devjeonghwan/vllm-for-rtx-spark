"""Native-Windows boundary for the recipe's full-vocabulary FP4 draft head.

The original image installs mbx_draft_head.make after target/draft head sharing.
That helper is a Linux Cython ELF. This boundary reuses its original sharing
hook, the recipe's published NVFP4 quantizer, and pinned vLLM Marlin functions.
Only a separate draft copy is quantized. The target module/weights are read-only.
"""
from __future__ import annotations

import ast
import functools
import inspect
import math
import os
import textwrap

_ORIGINAL_LOAD = None
_MODE = None
_LAST_INSTALL = None
_LAYOUT_ORIGINAL = None
_LAYOUT_RECORDS = []
_LAYOUT_DROPPED = 0


def marlin_input(x):
    """Marlin accepts padded rows, but the innermost values must be adjacent."""
    return x if x.stride(-1) == 1 else x.contiguous()


def quantizer_with_global_scale(original):
    """One source change: let independent row chunks share the full-head scale."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    function.decorator_list = []
    assignments = [node for node in function.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'g']
    if len(assignments) != 1 or ast.unparse(assignments[0].value) != 'float(amax.max()) / (448.0 * 6.0)':
        raise RuntimeError('Pinned recipe NVFP4 quantizer scale assignment changed')
    function.args.args.append(ast.arg(arg='_full_head_scale'))
    assignments[0].value = ast.Name(id='_full_head_scale', ctx=ast.Load())
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<original NVFP4 quantizer; global-scale row chunk>', 'exec'), namespace)
    return namespace[function.name]


def quantize_weight(weight, original_quantizer, *, chunk_rows=8192):
    """Original FP4 bits/scales with bounded preparation-only float temporaries."""
    import torch
    if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('A dense BF16/FP16 original head is required')
    rows, columns = map(int, weight.shape)
    if rows <= 0 or columns <= 0 or columns % 16 or type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('Positive blockable head geometry and chunk_rows required')
    # On BF16/FP16 input the maximum is exactly representable in FP32. The
    # chunk reduction has the same global maximum as original_quantizer(W.float()).
    maximum = 0.0
    for first in range(0, rows, chunk_rows):
        maximum = max(maximum, float(weight[first:first+chunk_rows].abs().amax()))
    scale = maximum / (448.0 * 6.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError('Original head global scale must be positive and finite')
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, columns // 16), dtype=torch.float8_e4m3fn, device=weight.device)
    quant = quantizer_with_global_scale(original_quantizer)
    for first in range(0, rows, chunk_rows):
        last = min(first + chunk_rows, rows)
        bits, groups, actual_scale = quant(weight[first:last].float(), scale)
        if actual_scale != scale:
            raise RuntimeError('Original quantizer did not retain the global scale')
        packed[first:last].copy_(bits)
        scales[first:last].copy_(groups)
    return packed, scales, scale


class DraftNvfp4Method:
    """Same pinned original Marlin W4A16 boundary used by the recipe FP4 scorer."""
    def __init__(self, rows, columns):
        self.rows, self.columns = rows, columns

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
        return apply_fp4_marlin_linear(marlin_input(x), layer.weight, layer.weight_scale,
            layer.weight_global_scale, None, self.rows, self.columns, bias)


def begin_layout_pilot(capacity=32):
    """Optional CPU-metadata wrapper; caller must stop it before timed work."""
    global _LAYOUT_ORIGINAL, _LAYOUT_RECORDS, _LAYOUT_DROPPED
    if _LAYOUT_ORIGINAL is not None or type(capacity) is not int or not 1 <= capacity <= 128:
        raise RuntimeError('Fresh bounded layout pilot required')
    _LAYOUT_ORIGINAL = DraftNvfp4Method.apply
    _LAYOUT_RECORDS, _LAYOUT_DROPPED = [], 0
    @functools.wraps(_LAYOUT_ORIGINAL)
    def observe(self, layer, x, bias=None):
        global _LAYOUT_DROPPED
        if len(_LAYOUT_RECORDS) < capacity:
            _LAYOUT_RECORDS.append({'shape':list(x.shape), 'stride':list(x.stride()),
                'dtype':str(x.dtype), 'device':str(x.device),
                'normalization_copy_required':x.stride(-1) != 1,
                'normalization_bytes':x.numel()*x.element_size() if x.stride(-1) != 1 else 0})
        else:
            _LAYOUT_DROPPED += 1
        return _LAYOUT_ORIGINAL(self, layer, x, bias)
    observe._rtx_layout_pilot = True
    DraftNvfp4Method.apply = observe


def end_layout_pilot():
    global _LAYOUT_ORIGINAL
    if _LAYOUT_ORIGINAL is not None:
        if not getattr(DraftNvfp4Method.apply, '_rtx_layout_pilot', False):
            raise RuntimeError('Layout pilot was replaced by another owner')
        DraftNvfp4Method.apply = _LAYOUT_ORIGINAL
        _LAYOUT_ORIGINAL = None
    return {'active':False, 'records':list(_LAYOUT_RECORDS), 'dropped':_LAYOUT_DROPPED}


def storage_bytes(module):
    return sum(value.numel() * value.element_size() for value in module.parameters())


def make(target_head, *, chunk_rows=8192):
    import torch
    from vllm._rtx_spark.recipes.qwen3_8_flash_next.helpers.mbx_draft_head2 import _quant_nvfp4
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
    weight = target_head.weight
    if not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError('The tested original CUDA BF16 target head is required')
    if int(getattr(target_head, 'tp_size', 1)) != 1:
        raise ValueError('This candidate supports the existing TP1 configuration')
    before_pointer = int(weight.data_ptr())
    rows, columns = map(int, weight.shape)
    holder = torch.nn.Module()
    holder.input_size_per_partition = columns
    holder.output_size_per_partition = rows
    holder.params_dtype = weight.dtype
    for name in ('tp_size', 'org_vocab_size', 'num_embeddings', 'embedding_dim',
                 'num_embeddings_per_partition', 'vocab_start_index', 'vocab_end_index',
                 'padding_size'):
        if hasattr(target_head, name):
            setattr(holder, name, getattr(target_head, name))
    holder.tp_size = 1
    # DFlash get_top_k_tokens reads this CPU-only shard metadata even for TP1.
    # Keep a separate dataclass object; never change the target's shared owner.
    if hasattr(target_head, 'shard_indices'):
        import dataclasses
        if not dataclasses.is_dataclass(target_head.shard_indices):
            raise TypeError('Original vocabulary shard dataclass required')
        holder.shard_indices = dataclasses.replace(target_head.shard_indices)
    with torch.inference_mode():
        packed, scales, scale = quantize_weight(weight, _quant_nvfp4, chunk_rows=chunk_rows)
        holder.weight = torch.nn.Parameter(packed, requires_grad=False)
        holder.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
        holder.weight_global_scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=torch.float32, device=weight.device), requires_grad=False)
        del packed, scales
        prepare_fp4_layer_for_marlin(holder)
    holder.quant_method = DraftNvfp4Method(rows, columns)
    holder.bias = None
    holder._rtx_nvfp4_head = {'rows':rows, 'columns':columns, 'full_vocabulary':True,
        'chunk_rows':chunk_rows, 'global_scale_before_marlin':scale,
        'target_weight_pointer':before_pointer, 'persistent_tensor_bytes':storage_bytes(holder),
        'original_quantizer':'mbx_draft_head2._quant_nvfp4 (global-scale chunk only)',
        'linear_backend':'pinned vLLM prepare_fp4_layer_for_marlin/apply_fp4_marlin_linear'}
    if target_head.weight is not weight or int(target_head.weight.data_ptr()) != before_pointer:
        raise RuntimeError('Target head alias changed during draft conversion')
    return holder


def replace_draft_head(draft_model, target_head, factory=make):
    """The original image's share hook, restricted to exact shared aliases."""
    from torch import nn
    if getattr(draft_model, 'lm_head', None) is not target_head:
        raise RuntimeError('Original target/draft head sharing was not established')
    if getattr(draft_model, '_mbx_head2', None) is not None:
        raise RuntimeError('HEAD2 must be disabled before loading this isolated candidate')
    layers = getattr(getattr(draft_model, 'model', None), 'layers', None)
    items = list(layers.values() if isinstance(layers, nn.ModuleDict) else (layers or []))
    shared = [layer.shared_head for layer in items if getattr(layer, 'shared_head', None) is not None
              and getattr(layer.shared_head, 'head', None) is target_head]
    replacement = factory(target_head)
    draft_model.lm_head = replacement
    for layer_head in shared:
        del layer_head.head
        layer_head.head = replacement
    if draft_model.lm_head is target_head or any(layer.head is not replacement for layer in shared):
        raise RuntimeError('Draft head aliases were not consistently replaced')
    return replacement, len(shared)
