"""Windows file binding only: original tensors, bounded CSF decode, PLE metadata."""

from __future__ import annotations

import os
import re
import types
from collections import defaultdict
from pathlib import Path

from .raw_reader import CSFCheckpoint

TABLE = re.compile(
    r"\.ple_embedding\.ngram_embedding\.(?:shard_\d+\.(?:weight|weight_scale)|"
    r"nvfp4_shard_\d+\.(?:packed|scales))$"
)
GLOBAL = re.compile(
    r"\.ple_embedding\.ngram_embedding\.(?:weight_scale_2|nvfp4_global)$"
)
DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "U32": "uint32",
    "F32": "float32",
    "I64": "int64",
    "U64": "uint64",
    "F64": "float64",
}


def clone(function, **replacements):
    copied = types.FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def read_original_tensor(checkpoint, name, *, max_bytes=2 * 1024**3):
    """Read one owning original tensor; no shard mmap, conversion or persistent cache."""
    import torch

    metadata = checkpoint.metadata(name)
    size = metadata["byte_length"]
    if max_bytes <= 0 or size > max_bytes:
        raise MemoryError("Original tensor exceeds the explicit two-GiB source budget")
    dtype = getattr(torch, DTYPES[metadata["dtype"]])
    if size == 0:
        return torch.empty(metadata["shape"], dtype=dtype, device="cpu")
    entry = checkpoint._file(name)
    entry["stream"].seek(metadata["byte_offset"])
    payload = bytearray(size)
    view = memoryview(payload)
    consumed = 0
    while consumed < size:
        count = entry["stream"].readinto(
            view[consumed : min(size, consumed + 16 * 1024**2)]
        )
        if not count:
            raise EOFError("Original checkpoint payload was truncated")
        consumed += count
    checkpoint.stats["payload_reads"] += 1
    checkpoint.stats["payload_bytes"] += size
    checkpoint.stats["maximum_payload_bytes"] = max(
        checkpoint.stats["maximum_payload_bytes"], size
    )
    return torch.frombuffer(payload, dtype=dtype).reshape(metadata["shape"])


def iterate_original(checkpoint, files, *, selected=None, require_ple_mmap=True):
    """Yield the original NGram handler's metadata inputs without reading table payload."""
    import torch

    wanted = {Path(value).resolve() for value in files}
    grouped = defaultdict(list)
    for name in checkpoint.index:
        if selected is None or selected(name):
            grouped[checkpoint.path(name)].append(name)
    for path in files:
        actual = Path(path).resolve()
        if actual not in wanted or actual not in checkpoint._paths.values():
            raise ValueError(
                "The original iterator file must belong to its immutable index"
            )
        for name in grouped.get(actual, ()):
            if name.endswith(".weight_scale.nvfp4_csf_exceptions"):
                continue
            if TABLE.search(name):
                if not require_ple_mmap:
                    raise RuntimeError(
                        "PLE metadata placeholders require the original mmap handler"
                    )
                metadata = checkpoint.metadata(name)
                checkpoint.stats["ple_metadata_tensors"] = (
                    checkpoint.stats.get("ple_metadata_tensors", 0) + 1
                )
                yield (
                    name,
                    torch.empty(
                        metadata["shape"],
                        dtype=getattr(torch, DTYPES[metadata["dtype"]]),
                        device="meta",
                    ),
                )
            elif name.endswith(".weight_scale.nvfp4_csf_fixed"):
                projection = name.removesuffix(".weight_scale.nvfp4_csf_fixed")
                yield projection + ".weight_scale", checkpoint.scale(projection)
            else:
                value = read_original_tensor(checkpoint, name)
                if GLOBAL.search(name):
                    if value.dtype != torch.float32 or value.numel() != 1:
                        raise ValueError(
                            "Preserve the original single FP32 PLE global scale"
                        )
                    checkpoint.stats["ple_real_global_scalars"] = (
                        checkpoint.stats.get("ple_real_global_scalars", 0) + 1
                    )
                yield name, value


def make_loader_class(default_loader):
    """Keep original DefaultModelLoader bodies; replace the local file iterator binding."""

    class MbxModelLoader(default_loader):
        def __init__(self, load_config):
            super().__init__(load_config)
            if os.environ.get("MBX_LOADER_PREPARE", "0") != "0":
                raise NotImplementedError(
                    "This exact-weight facade admits original MBX_LOADER_PREPARE=0 only"
                )
            if load_config.model_loader_extra_config.get("enable_multithread_load"):
                raise ValueError(
                    "The native bounded reader requires sequential source iteration"
                )
            if load_config.safetensors_load_strategy not in (None, "lazy"):
                raise ValueError(
                    "The original raw-file binding does not claim mmap strategy semantics"
                )
            self.mbx_windows_source_stats = []
            self._mbx_selected = None
            self._mbx_iterating = False

        def _prepare_weights(
            self,
            model_name_or_path,
            subfolder,
            revision,
            fall_back_to_pt,
            allow_patterns_overrides,
        ):
            model = Path(model_name_or_path)
            if (
                not model.is_dir()
                or revision is not None
                or allow_patterns_overrides is not None
            ):
                raise ValueError(
                    "Native facade requires the explicit original local indexed checkpoint"
                )
            if subfolder is not None:
                model = model / subfolder
            with CSFCheckpoint(model) as checkpoint:
                files = sorted(checkpoint._paths.values())
            if not all(path.is_file() for path in files):
                raise FileNotFoundError("An original indexed source shard is missing")
            return (
                str(model.resolve()),
                list(map(str, files)),
                True,
                "model.safetensors.index.json",
            )

        def _get_weights_iterator(self, source):
            model = Path(source.model_or_path)
            if source.subfolder is not None:
                model = model / source.subfolder

            def original_iterator(files, use_tqdm, strategy=None, **options):
                if options.get("local_expert_ids") is not None:
                    raise NotImplementedError(
                        "Initial original loader binding is single-rank, all experts"
                    )
                with CSFCheckpoint(model) as checkpoint:
                    self.mbx_windows_source_stats.append(checkpoint.stats)
                    selection = self._mbx_selected
                    if source.is_unused_weight is not None:
                        selection = lambda name, prior=selection: (
                            (prior is None or prior(name))
                            and not source.is_unused_weight(name)
                        )
                    yield from iterate_original(
                        checkpoint,
                        files,
                        selected=selection,
                        require_ple_mmap=os.environ.get("MBX_PLE_MMAP") == "1",
                    )

            # Same code object/defaults/closure. Only the OS file-source binding differs.
            method = clone(
                default_loader._get_weights_iterator,
                safetensors_weights_iterator=original_iterator,
            )
            yield from method(self, source)

        def get_all_weights(self, model_config, model):
            if self._mbx_iterating:
                raise RuntimeError(
                    "One original loader iterator may be active at a time"
                )
            self._mbx_iterating = True
            try:
                if (
                    type(model).__module__ == "vllm.models.qwen4_exp.nvidia.mtp"
                    and type(model).__name__ == "Qwen4ExpMTP"
                ):
                    import sys

                    mapper = sys.modules[type(model).__module__]._remap_mtp_weight_name
                    self._mbx_selected = lambda name: mapper(name) is not None
                else:
                    self._mbx_selected = None
                yield from super().get_all_weights(model_config, model)
            finally:
                self._mbx_selected = None
                self._mbx_iterating = False

    return MbxModelLoader
