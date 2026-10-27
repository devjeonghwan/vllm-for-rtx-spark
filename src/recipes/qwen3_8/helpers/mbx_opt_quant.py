"""mbx_opt_quant — opt-in weight-only re-quantization of the remaining bf16 linears, served by vLLM 0.32's Marlin W8A16 kernels.

Ported from Alssndr0's qwen38-flash-next-recipe fork (Apache-2.0, github.com/Alssndr0/qwen38-flash-next-recipe,
image/files/opt_fp8.py + the load hooks in image/patches/{mtp,model}.patch); adapted to vLLM 0.32 (no per-layer Marlin workspace:
the kernels take workspace=None and use the per-stream persistent one; conversion runs AFTER process_weights_after_loading, from
patch 42 for the drafter / patch 43 for the target), to Qwen3.8-Flash-Next-FP4-Local (only linears whose weight is still bf16
qualify: MXFP8 attention/GDN/shared experts and NVFP4 experts are skipped by dtype), and to 0.32's fused hyper-connection
down+inject GEMM (hc_down_silu reads the bf16 weight directly — those modules are skipped unless MBX_OPT_QUANT_HC_DOWN=1, which
converts them and switches their module to the unfused GEMM path).

Switches (all default off — nothing here runs unless set):
  MBX_DRAFT_FP8=1            MTP drafter: bf16 linears → FP8 e4m3, one scale per output channel (Marlin W8A16); a layer whose
                             output relative error on a random activation is >= 5 % stays bf16. Exact for the served text (the
                             verifier's rejection sampling keeps the target's distribution); only acceptance can move.
  MBX_INT8_GPTQ=<codes.pt>   target: bf16 linears → symmetric INT8, one bf16 scale per MBX_INT8_GROUP (128) inputs, Marlin
                             uint8b128, with offline GPTQ codes {module: int8 [N, K]} (tools/gptq_int8.py). NOT output-exact.
                             "rtn" = round-to-nearest without codes (for A/Bs). Modules without codes fall back to RTN (logged).
  MBX_INT8_INCLUDE / MBX_INT8_EXCLUDE   comma-separated module-name substrings (target).
  MBX_INT8_CALIB_DIR=<dir>   target calibration: record sampled input rows of every qualifying bf16 linear on eager prefill
                             calls while <dir>/GO exists (tools/calib_capture.py), plus each weight, for tools/gptq_int8.py.
                             MBX_INT8_CALIB_ROWS (12288), MBX_INT8_CALIB_PER_CALL (300).
"""
import os

import torch
from torch import nn

from vllm.logger import init_logger

log = init_logger("vllm.mbx_opt_quant")

_EXCLUDE = ("router", "lm_head", "embed", "visual")
_EXCLUDE_SUFFIX = (".gate", "shared_expert_gate")
_HC_DOWN = "input_mix_weight_down_block_inject"     # fused hc_down_silu reads its bf16 weight directly (nvidia/hyperconnection.py)


class _Fp8MarlinMethod:
    """Stand-in quant method: weight already repacked for Marlin FP8; process_weights_after_loading is a no-op."""

    def __init__(self, n, k):
        self.n, self.k = n, k

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
        return apply_fp8_marlin_linear(x, layer.weight, layer.weight_scale, None, self.n, self.k, bias)


class _Int8MarlinMethod:
    """Stand-in quant method: INT8 (uint8b128) weight already packed for Marlin GPTQ; no-op after loading."""

    def __init__(self, n, k):
        self.n, self.k = n, k

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
        from vllm.scalar_type import scalar_types
        return apply_gptq_marlin_linear(x, layer.weight, layer.weight_scale, layer.weight_zp, None,
                                        scalar_types.uint8b128, self.n, self.k, bias=bias)


def int8_group(K, group):
    """Largest Marlin-supported group <= `group` that divides K (-1 = per row)."""
    if group <= 0:
        return -1
    for g in (group, 128, 64, 32):
        if g <= group and K % g == 0:
            return g
    return -1


def quant_int8(w, group, q=None):
    """w [N, K] -> (q [N, K] int in [-128, 127], s [N, K/g] bf16, g). Scales are rounded to bf16 BEFORE the codes are chosen
    (the kernel multiplies by the bf16 scale), all in fp32. `q` = codes chosen elsewhere (GPTQ). tools/gptq_int8.py imports
    this function: the offline codes and the serve use the same scales by construction."""
    N, K = w.shape
    g = int8_group(K, group)
    G = K if g == -1 else g
    x = w.float().view(N, K // G, G)
    s = (x.abs().amax(-1) / 127.0).clamp(min=1e-10).to(torch.bfloat16)
    if q is None:
        q = (x / s.float().unsqueeze(-1)).round().clamp(-128, 127).view(N, K)
    return q.to(torch.int32), s, g


def pack_int8_marlin(q, s, g):
    """codes q [N, K] in [-128, 127], scales s [N, K/G] -> Marlin GPTQ uint8b128 weight + permuted scales."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_padded_nk, marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
    N, K = q.shape
    # Marlin needs (n % 64, k % 128) or (n % 128, k % 64): zero-pad codes (as 128 = 0 after the b128 bias) and scales (padded
    # scales are 0, so the padded region contributes nothing); apply_gptq_marlin_linear pads x and slices the output back.
    pn, pk = marlin_padded_nk(N, K, g)
    qk = torch.nn.functional.pad(q + 128, (0, pk - K, 0, pn - N), value=128).t().contiguous()   # [pk, pn] uint
    st = s.t().contiguous()                                                                    # [K/G, N]
    rows = pk // g if g > 0 else 1
    st = torch.nn.functional.pad(st, (0, pn - N, 0, rows - st.shape[0]))
    packed = marlin_weights(qk, pk, pn, 8, get_weight_perm(8))
    sp = marlin_permute_scales(st, pk, pn, g)
    return packed, sp


def _bf16_linear(mod):
    from vllm.model_executor.layers.linear import LinearBase
    w = getattr(mod, "weight", None)
    return (isinstance(mod, LinearBase) and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2
            and not isinstance(getattr(mod, "quant_method", None), (_Fp8MarlinMethod, _Int8MarlinMethod, _Recorder)))


def _selected(model, include, exclude, exclude_suffix, min_bytes):
    """(name, module) of every qualifying bf16 linear."""
    hc_down = os.environ.get("MBX_OPT_QUANT_HC_DOWN", "0") == "1"
    for name, mod in list(model.named_modules()):
        if not _bf16_linear(mod):
            continue
        if include is not None and not any(s in name for s in include):
            continue
        if any(s in name for s in exclude) or name.endswith(tuple(exclude_suffix)):
            continue
        if name.endswith(_HC_DOWN) and not hc_down:
            continue
        if mod.weight.numel() * 2 < min_bytes or mod.weight.shape[1] % 16:
            continue
        yield name, mod


def _unfuse_hc_down(model, name):
    """The converted down+inject weight is Marlin-packed: route its hyper-connection through the module call."""
    if name.endswith(_HC_DOWN):
        parent = model.get_submodule(name.rpartition(".")[0])
        if getattr(parent, "_use_hc_down_silu", False):
            parent._use_hc_down_silu = False


def convert(model: nn.Module, include=None, exclude=_EXCLUDE, exclude_suffix=_EXCLUDE_SUFFIX, min_bytes=1 << 20, tag="",
            fmt="fp8", group=-1, codes=None) -> int:
    """Convert every qualifying bf16 LinearBase of `model`; returns the bytes saved. fmt "fp8" (e4m3 per row) or "int8" (per
    `group` input values, `codes` = {module name: int8 codes [N, K]} from GPTQ instead of round-to-nearest)."""
    if fmt == "int8":
        return _convert_int8(model, include, exclude, exclude_suffix, min_bytes, tag, group, codes or {})
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    saved, done, errs = 0, 0, []
    for name, mod in _selected(model, include, exclude, exclude_suffix, min_bytes):
        w = mod.weight.data
        N, K = w.shape
        amax = w.float().abs().amax(dim=1)
        scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))   # all-zero (padding) rows: any scale
        q = (w.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        # one-time accuracy check on a random activation (relative error of the output)
        x = torch.randn(4, K, device=w.device, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w).float()
        holder = nn.Module()
        holder.weight = nn.Parameter(q, requires_grad=False)
        holder.weight_scale = nn.Parameter(scale.float(), requires_grad=False)
        holder.output_size_per_partition, holder.input_size_per_partition = N, K
        holder.orig_dtype = torch.bfloat16
        meth = _Fp8MarlinMethod(N, K)
        try:
            prepare_fp8_layer_for_marlin(holder, size_k_first=False)
            out = meth.apply(holder, x).float()
            err = ((out - ref).norm() / ref.norm().clamp(min=1e-12)).item()
        except Exception as e:  # noqa: BLE001  (a shape Marlin cannot take: keep bf16)
            log.warning("mbx fp8%s: %s %s not converted (%r)", tag, name, tuple(w.shape), e)
            continue
        errs.append(err)
        if not err < 0.05:
            log.warning("mbx fp8%s: %s %s rel.err %.4f — kept bf16", tag, name, tuple(w.shape), err)
            continue
        _unfuse_hc_down(model, name)
        mod.weight = holder.weight
        mod.weight_scale = holder.weight_scale
        mod.quant_method = meth
        saved += N * K
        done += 1
        log.info("mbx fp8%s: %s %s rel.err %.4f", tag, name, (N, K), err)
    torch.cuda.empty_cache()
    log.info("mbx fp8%s: %d linears converted, %.1f MB of weights saved, output rel.err max %.4f mean %.4f", tag, done,
             saved / 1e6, max(errs) if errs else 0.0, sum(errs) / len(errs) if errs else 0.0)
    return saved


def _convert_int8(model, include, exclude, exclude_suffix, min_bytes, tag, group, codes):
    saved, done, errs, gptq, rtn_fallback = 0, 0, [], 0, []
    used = set()
    for name, mod in _selected(model, include, exclude, exclude_suffix, min_bytes):
        w = mod.weight.data
        N, K = w.shape
        if not torch.isfinite(w).all():
            log.warning("mbx int8%s: %s %s has non-finite weights — kept bf16", tag, name, tuple(w.shape))
            continue
        cq = codes.get(name)
        if cq is not None and tuple(cq.shape) != (N, K):
            log.warning("mbx int8%s: %s codes %s != weight %s — round-to-nearest", tag, name, tuple(cq.shape), (N, K))
            cq = None
        if codes and cq is None:
            rtn_fallback.append(name)
        q, s, g = quant_int8(w, group, q=cq.to(w.device) if cq is not None else None)
        x = torch.randn(4, K, device=w.device, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w).float()
        holder = nn.Module()
        meth = _Int8MarlinMethod(N, K)
        try:
            packed, sp = pack_int8_marlin(q, s, g)
            holder.weight = nn.Parameter(packed, requires_grad=False)
            holder.weight_scale = nn.Parameter(sp, requires_grad=False)
            holder.weight_zp = torch.empty(0, dtype=torch.int, device=w.device)
            out = meth.apply(holder, x).float()
            err = ((out - ref).norm() / ref.norm().clamp(min=1e-12)).item()
        except Exception as e:  # noqa: BLE001
            log.warning("mbx int8%s: %s %s not converted (%r)", tag, name, tuple(w.shape), e)
            continue
        errs.append(err)
        if not err < 0.05:
            log.warning("mbx int8%s: %s %s rel.err %.4f — kept bf16", tag, name, tuple(w.shape), err)
            continue
        _unfuse_hc_down(model, name)
        mod.weight = holder.weight
        mod.weight_scale = holder.weight_scale
        mod.weight_zp = holder.weight_zp
        mod.quant_method = meth
        saved += N * K - s.numel() * 2
        done += 1
        gptq += cq is not None
        if cq is not None:
            used.add(name)
        log.debug("mbx int8%s: %s %s g%d rel.err %.5f", tag, name, (N, K), g, err)
    torch.cuda.empty_cache()
    if rtn_fallback:
        log.warning("mbx int8%s: %d modules had no GPTQ codes (round-to-nearest): %s", tag, len(rtn_fallback),
                    rtn_fallback[:8])
    unused = sorted(set(codes) - used)
    if unused:
        log.warning("mbx int8%s: %d code entries matched no converted module (other checkpoint/layout?): %s", tag,
                    len(unused), unused[:8])
    log.info("mbx int8%s: %d linears converted (group %d, %d with offline codes), %.1f MB saved, output rel.err "
             "max %.5f mean %.5f", tag, done, group, gptq, saved / 1e6, max(errs) if errs else 0.0,
             sum(errs) / len(errs) if errs else 0.0)
    return saved


class _Recorder:
    """Calibration (MBX_INT8_CALIB_DIR): wraps a module's quant_method and saves randomly sampled input rows of eager (prefill)
    calls to OUT/<module>.<n>.pt for offline GPTQ. Armed only while OUT/GO exists (so the startup profile run's dummy tokens are
    not recorded); each call's rows go straight to disk (host memory is tight)."""
    _armed = [False, 0.0]

    def __init__(self, inner, name, out, rows, per_call):
        self.inner, self.name, self.out, self.rows, self.per_call = inner, name, out, rows, per_call
        self.n, self.k = 0, 0

    @classmethod
    def armed(cls, out):
        import time
        now = time.monotonic()
        if now - cls._armed[1] > 1.0:
            cls._armed[0], cls._armed[1] = os.path.exists(os.path.join(out, "GO")), now
        return cls._armed[0]

    def process_weights_after_loading(self, layer):
        return self.inner.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        if self.n < self.rows and x.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing() \
                and self.armed(self.out):
            x2 = x.reshape(-1, x.shape[-1])
            take = min(self.per_call, self.rows - self.n, x2.shape[0])
            idx = torch.randperm(x2.shape[0], device=x2.device)[:take]
            torch.save(x2.index_select(0, idx).to("cpu"), f"{self.out}/{self.name}.{self.k}.pt")
            self.n += take
            self.k += 1
        return self.inner.apply(layer, x, bias)


def install_recorders(model, out, include=None, exclude=_EXCLUDE, exclude_suffix=_EXCLUDE_SUFFIX, min_bytes=1 << 20,
                      rows=4096, per_call=160):
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, mod in _selected(model, include, exclude, exclude_suffix, min_bytes):
        mod.quant_method = _Recorder(mod.quant_method, name, out, rows, per_call)
        torch.save(mod.weight.data.to("cpu"), f"{out}/{name}.W.pt")      # the weight in vLLM's merged layout
        n += 1
    log.info("mbx calib: recording inputs of %d linears into %s (armed while %s/GO exists)", n, out, out)
    return n


def _split(v):
    return [x for x in (v or "").split(",") if x]


def target_loaded(model, vllm_config, model_config) -> None:
    """Patch 43 (base_loader, after process_weights_after_loading): the target only — drafters go through draft_loaded."""
    if "MTP" in type(model).__name__ or model_config is not vllm_config.model_config:
        return
    inc = _split(os.environ.get("MBX_INT8_INCLUDE")) or None
    exc = _EXCLUDE + tuple(_split(os.environ.get("MBX_INT8_EXCLUDE")))
    calib = os.environ.get("MBX_INT8_CALIB_DIR")
    if calib:
        install_recorders(model, calib, include=inc, exclude=exc,
                          rows=int(os.environ.get("MBX_INT8_CALIB_ROWS", "12288")),
                          per_call=int(os.environ.get("MBX_INT8_CALIB_PER_CALL", "300")))
        return                                             # calibration records the bf16 layers: no conversion on that boot
    src = os.environ.get("MBX_INT8_GPTQ")
    if not src:
        return
    codes = None
    if src.lower() != "rtn":
        codes = torch.load(src, map_location="cpu", weights_only=True)
        log.info("mbx int8: %d GPTQ code entries from %s", len(codes), src)
    convert(model, include=inc, exclude=exc, tag=" (target)", fmt="int8",
            group=int(os.environ.get("MBX_INT8_GROUP", "128")), codes=codes)


def draft_loaded(draft_model) -> None:
    """Patch 42 (MTPSpeculator.load_draft_model, after the target's embed/lm_head are shared in): FP8 drafter projections."""
    if os.environ.get("MBX_DRAFT_FP8", "0") == "1":
        convert(draft_model, tag=" (mtp drafter)")
