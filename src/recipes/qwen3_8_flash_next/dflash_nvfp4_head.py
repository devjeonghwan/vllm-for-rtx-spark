"""DFlash loader boundary for the previously tested recipe NVFP4 head.

No new quantization/GEMM algorithm; target head stays read-only. Imports are CPU
only until original draft loading reaches the configured late hook. Startup full-weight SHA
checks copy bounded CPU chunks twice; no weights are serialized.
"""
from __future__ import annotations
import copy
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from . import dflash_nvfp4_core as core

_LOAD_OWNER = None
_LATE_HOOK_CALLED = False
_OWNERS = None
_ACTIVATION = None
_FAILURE = None
CHUNK_ROWS = 8192


def tensor_metadata(value):
    try:
        version = value._version
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        version = "inference_tensor_untracked"
    return {"object_id": id(value), "data_ptr": int(value.data_ptr()),
        "shape": list(value.shape), "stride": list(value.stride()),
        "storage_offset": value.storage_offset(), "dtype": str(value.dtype),
        "device": str(value.device), "requires_grad": value.requires_grad,
        "version": version}


def module_metadata(module):
    return {"object_id": id(module), "class": type(module).__module__ + "." + type(module).__qualname__,
        "quant_method_id": id(module.quant_method),
        "parameters": {name: tensor_metadata(value) for name, value in module.named_parameters()},
        "buffers": {name: tensor_metadata(value) for name, value in module.named_buffers()}}


def target_tree_metadata(module):
    return {"object_id": id(module),
        "modules": {name: {"object_id": id(value), "class": type(value).__module__ + "." + type(value).__qualname__}
                    for name, value in module.named_modules(remove_duplicate=False)},
        "parameters": {name: tensor_metadata(value) for name, value in module.named_parameters(remove_duplicate=False)},
        "buffers": {name: tensor_metadata(value) for name, value in module.named_buffers(remove_duplicate=False)}}


def tree_summary(metadata):
    return {"sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "module_count": len(metadata["modules"]), "parameter_name_count": len(metadata["parameters"]),
        "buffer_name_count": len(metadata["buffers"])}


def hash_weight(weight, *, chunk_rows=CHUNK_ROWS):
    """Logical bytes only. One CPU chunk; memoryview avoids an extra bytes copy."""
    import torch
    assert weight.is_cuda and weight.dtype == torch.bfloat16 and weight.ndim == 2
    assert weight.stride(-1) == 1 and weight.is_contiguous()
    digest = hashlib.sha256()
    copied = 0
    largest = 0
    for first in range(0, weight.shape[0], chunk_rows):
        host = weight[first:first + chunk_rows].detach().view(torch.uint8).cpu()
        view = memoryview(host.numpy()).cast("B")
        digest.update(view)
        copied += view.nbytes
        largest = max(largest, view.nbytes)
        del view, host
    return {"sha256": digest.hexdigest(), "D2H_bytes": copied,
        "max_CPU_chunk_bytes": largest, "CPU_tobytes_copy_bytes": 0,
        "GPU_full_weight_clone_bytes": 0, "weights_written_to_files": False}


def convert(draft, target_head, target_model=None):
    """Shared original head -> separate original recipe copy, before Graphs."""
    global _OWNERS, _ACTIVATION, _FAILURE
    if _OWNERS is not None:
        raise RuntimeError("A fresh process/model is required for NVFP4 head conversion")
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_padded_nk

    assert draft.lm_head is target_head and target_head.weight.ndim == 2
    rows, columns = map(int, target_head.weight.shape)
    assert rows > 0 and columns > 0 and columns % 16 == 0
    expected_bytes = core.expected_storage_bytes(*marlin_padded_nk(rows, columns, 16))
    assert target_head.weight.dtype == torch.bfloat16 and target_head.tp_size == 1
    assert getattr(draft, "_mbx_head2", None) is None
    assert dataclasses.is_dataclass(target_head.shard_indices)
    shard_before = dataclasses.asdict(target_head.shard_indices)
    assert all(type(value) is int for value in shard_before.values())
    assert target_head.shard_indices.org_vocab_start_index == 0
    assert target_head.shard_indices.num_org_vocab_padding == 0
    before = module_metadata(target_head)
    whole_target = target_head if target_model is None else target_model
    tree_before = target_tree_metadata(whole_target)
    weight = target_head.weight
    allocated_before = torch.cuda.memory_allocated(weight.device)
    peak_before = torch.cuda.max_memory_allocated(weight.device)
    reserved_before = torch.cuda.memory_reserved(weight.device)
    reserved_peak_before = torch.cuda.max_memory_reserved(weight.device)
    checksum_before = hash_weight(weight)
    try:
        replacement, aliases = core.replace_draft_head(draft, target_head)
        checksum_after = hash_weight(weight)
        after = module_metadata(target_head)
        tree_after = target_tree_metadata(whole_target)
        assert tree_before == tree_after, "Whole target module/parameter/buffer identity or metadata changed"
        assert before == after, "Target object/parameter/buffer metadata changed"
        assert checksum_before["sha256"] == checksum_after["sha256"], "Target full-weight bytes changed"
        assert target_head.weight is weight and draft.lm_head is replacement and replacement is not target_head
        assert replacement.shard_indices is not target_head.shard_indices
        assert dataclasses.asdict(replacement.shard_indices) == shard_before
        assert dataclasses.asdict(target_head.shard_indices) == shard_before
        assert core.storage_bytes(replacement) == expected_bytes
        assert isinstance(replacement.quant_method, core.DraftNvfp4Method)
        assert replacement._rtx_nvfp4_head["rows"] == rows and replacement._rtx_nvfp4_head["columns"] == columns
        _OWNERS = {"target": target_head, "target_weight": weight, "draft": draft, "holder": replacement,
                   "whole_target": whole_target, "whole_target_tree_before": tree_before}
        _ACTIVATION = {"schema": "dflash-original-NVFP4-head-activation-v1", "pid": os.getpid(),
            "rows": rows, "columns": columns, "full_vocabulary": True,
            "target_module_before": before, "target_module_after": after,
            "target_full_weight_before": checksum_before, "target_full_weight_after": checksum_after,
            "target_full_weight_bytes_unchanged": True, "target_metadata_unchanged": True,
            "whole_target_tree_before": tree_summary(tree_before), "whole_target_tree_after": tree_summary(tree_after),
            "whole_target_tree_metadata_unchanged": True,
            "target_tree_scope": "head-only component" if target_model is None else "entire target model supplied by original load hook",
            "draft_module_id": id(replacement), "draft_is_separate_holder": True,
            "target_and_draft_share_head": False, "holder_metadata": module_metadata(replacement),
            "shard_indices": shard_before, "independent_shard_dataclass_copy": True,
            "per_layer_shared_aliases": aliases,
            "persistent_tensor_bytes": core.storage_bytes(replacement),
            "allocated_before_bytes": allocated_before,
            "allocated_after_bytes": torch.cuda.memory_allocated(weight.device),
            "allocated_delta_bytes": torch.cuda.memory_allocated(weight.device) - allocated_before,
            "allocator_peak_before_bytes": peak_before,
            "allocator_peak_after_bytes": torch.cuda.max_memory_allocated(weight.device),
            "reserved_before_bytes": reserved_before,
            "reserved_after_bytes": torch.cuda.memory_reserved(weight.device),
            "allocator_reserved_peak_before_bytes": reserved_peak_before,
            "allocator_reserved_peak_after_bytes": torch.cuda.max_memory_reserved(weight.device),
            "startup_checksum_D2H_bytes": checksum_before["D2H_bytes"] + checksum_after["D2H_bytes"],
            "startup_max_CPU_chunk_bytes": max(checksum_before["max_CPU_chunk_bytes"], checksum_after["max_CPU_chunk_bytes"]),
            "startup_checksum_CPU_tobytes_copy_bytes": 0, "weights_serialized": False,
            "core_source": str(Path(core.__file__).resolve()),
            "core_source_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "adapter_source": str(Path(__file__).resolve()),
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "quantization": "Original mbx_draft_head2._quant_nvfp4; W4A16 Marlin, BF16 activations; full-head global scale row chunks",
            "lifetime": "Target unchanged; separate draft holder and original core parameters retained through Graph destruction/process lifetime. Original Marlin owns its scratch/cache. No per-request CPU copy or quantization.",
            "memory_scope": "Torch allocated/reserved peaks only when reported; not total NVML/CPU process peak"}
        draft._rtx_dflash_nvfp4_install = _ACTIVATION
        return replacement
    except BaseException as error:
        _FAILURE = {"error": repr(error), "traceback": traceback.format_exc()}
        raise


def maybe_convert(speculator, draft, target_model):
    """Apply the explicit recipe option after original loading, before capture.

    The option travels in VllmConfig to Windows spawned workers. No environment
    flag or parent-process monkey patch owns this conversion.
    """
    global _LOAD_OWNER, _LATE_HOOK_CALLED
    # Other recipes keep their original DFlash loader behavior. Read only
    # this opt-in setting; never validate their keys against Qwen's options.
    additional = getattr(speculator.vllm_config, "additional_config", None) or {}
    options = additional.get("rtx_spark", {}) if isinstance(additional, dict) else {}
    if not isinstance(options, dict):
        return draft
    nested = options.get("optimizations", {})
    key = "dflash2_nvfp4_head"
    flat_present = key in options
    nested_present = isinstance(nested, dict) and key in nested
    if not flat_present and not nested_present:
        return draft
    supplied = ([options[key]] if flat_present else []) + ([nested[key]] if nested_present else [])
    if any(type(value) is not bool for value in supplied):
        raise TypeError("rtx_spark.dflash2_nvfp4_head must be bool")
    if len(supplied) == 2 and supplied[0] != supplied[1]:
        raise ValueError("Conflicting RTX Spark setting: dflash2_nvfp4_head")
    if not supplied[0]:
        return draft
    config = speculator.vllm_config
    speculative = config.speculative_config
    parallel = config.parallel_config
    if (type(speculator).__name__ != "DFlash2Speculator"
            or speculative.method != "dflash"
            or "DFlash2DraftModel" not in (getattr(
                speculative.draft_model_config.hf_config, "architectures", None) or ())):
        raise ValueError("dflash2_nvfp4_head requires the DFlash2 draft model")
    if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1):
        raise ValueError("dflash2_nvfp4_head supports TP1/PP1/CP1")
    if "Qwen4ExpForConditionalGeneration" not in (getattr(
            config.model_config.hf_config, "architectures", None) or ()):
        raise ValueError("dflash2_nvfp4_head requires the tested Qwen4Exp target")
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    language = target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    target_head = get_target_lm_head(target_model, language)
    if target_head is None or draft.lm_head is not target_head:
        raise RuntimeError("Original target/DFlash head sharing was not established")
    _LOAD_OWNER = DFlashSpeculator.load_draft_model
    _LATE_HOOK_CALLED = True
    convert(draft, target_head, target_model)
    return draft


def status():
    """Idle metadata only; checks no input values and adds no GPU operations."""
    value = {"schema": "dflash-original-NVFP4-head-status-v1", "installed": _LATE_HOOK_CALLED,
        "activation": copy.deepcopy(_ACTIVATION), "failure": copy.deepcopy(_FAILURE),
        "strong_owners": _OWNERS is not None, "fresh_worker_required": True}
    value["startup_hook_kind"] = "original-loader late call; no replacement wrapper"
    value["startup_hook_binding_intact"] = _LATE_HOOK_CALLED and sys.modules[
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator"].DFlashSpeculator.load_draft_model is _LOAD_OWNER
    if _LATE_HOOK_CALLED and not value["startup_hook_binding_intact"]:
        raise RuntimeError("The DFlash head late-loader owner changed")
    if _OWNERS is not None:
        value["live_target_metadata_unchanged"] = module_metadata(_OWNERS["target"]) == _ACTIVATION["target_module_before"]
        value["live_draft_holder_binding"] = _OWNERS["draft"].lm_head is _OWNERS["holder"]
        value["live_holder_metadata_unchanged"] = module_metadata(_OWNERS["holder"]) == _ACTIVATION["holder_metadata"]
        value["whole_target_tree_metadata_unchanged"] = target_tree_metadata(_OWNERS["whole_target"]) == _OWNERS["whole_target_tree_before"]
        if not all(value[key] for key in ("live_target_metadata_unchanged", "live_draft_holder_binding", "live_holder_metadata_unchanged", "whole_target_tree_metadata_unchanged")):
            raise RuntimeError("NVFP4 target/holder metadata or binding changed")
    return value
