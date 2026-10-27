"""mbx_draft_head2 — opt-in two-stage vocabulary head for the MTP drafter (MBX_DRAFT_HEAD2=<basis.pt>).

Ported from Alssndr0's qwen38-flash-next-recipe fork (Apache-2.0, github.com/Alssndr0/qwen38-flash-next-recipe,
image/files/opt_draft_head.py + the compute_logits hook in image/patches/mtp.patch). A draft pass normally reads the whole
vocabulary head (248,320 x 2,560; bf16 on FP4-Local = 1.27 GB per pass). Here it instead
  1. projects the drafter hidden state onto a basis P [H, r] fitted on drafter hidden states (tools/fit_draft_basis.py),
  2. scores the whole vocabulary with U = W Q [V, r] (Q = P unless the basis file carries a fitted "Q"), stored FP8 per row
     (~64 MB at r = 256, Marlin W8A16) or NVFP4 (MBX_DRAFT_HEAD2_SCORER=fp4, ~32 MB, Marlin W4A16),
  3. keeps the top MBX_DRAFT_HEAD2_K (2048) candidates and computes their exact logits from the head's own rows,
  4. returns full-vocabulary logits with -inf outside the candidates.
Exact for the served text: only the draft proposals change (q is renormalized over the candidates; a target token outside them
cannot be drafted), verification against the target is unchanged.

Adapted for 0.32 / FP4-Local: the head is read from the drafter's lm_head MODULE after vLLM shares the target's head into it
(spec_decode/eagle/utils.load_eagle_model), not from the checkpoint — FP4-Local's head is bf16 (modelopt `ignore`), and the
module weight is used in place (no copy). A head that is not a plain bf16/fp16 matrix (e.g. an NVFP4 head under patch 11 is
Marlin-repacked after loading) is read raw from the checkpoint's lm_head.weight / weight_scale / weight_scale_2 instead
(Alssndr0's NVFP4 path). His INT4-AutoRound head path and his draft truncation (OPT_DRAFT_TOPK/TOPP) are not ported.
TP = 1 only (a vocab-parallel shard would need the gather).

Calibration for the basis (MBX_DRAFT_HEAD2_DUMP=/cache/draft_h.pt, MBX_DRAFT_HEAD2_DUMP_ROWS=8192): saves drafter hidden states
(the head's input) of EAGER compute_logits calls, after skipping the first 2,000 (start-up, warm-up). Under FULL CUDA graphs the
drafter's compute_logits is inside the graph and never runs eagerly — boot the dump serve with --enforce-eager.
"""
import json
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

log = init_logger("vllm.mbx_draft_head2")
_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _quant_nvfp4(U):
    """U [N, K] float -> (packed u8 [N, K/2], fp8 group scales [N, K/16], global scale); dequant = e2m1 * s * g."""
    N, K = U.shape
    G = U.view(N, K // 16, 16)
    amax = G.abs().amax(-1)
    g = float(amax.max()) / (448.0 * 6.0)
    s = (amax / 6.0 / g).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn)
    x = G / (s.float() * g).unsqueeze(-1)
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=U.device)
    idx = torch.bucketize(x.abs().clamp(max=6.0), mid)
    code = (idx | ((x < 0) & (idx > 0)).to(idx.dtype) * 8).to(torch.uint8).view(N, K)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), s.contiguous(), g


@triton.jit
def _cand_logits_dense_kernel(h_ptr, cand_ptr, w_ptr, out_ptr, K, stride_h, stride_c, stride_w, stride_o,
                              H: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[m, j] = h[m] . W[cand[m, j]] for one row m and BLOCK_C candidates per program (bf16/fp16 head rows, fp32 acc)."""
    m = tl.program_id(0)
    jb = tl.program_id(1)
    offs_c = jb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < K
    rows = tl.load(cand_ptr + m * stride_c + offs_c, mask=mask_c, other=0).to(tl.int64)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        w = tl.load(w_ptr + rows[:, None] * stride_w + offs_h[None, :], mask=mask_c[:, None] & mask_h[None, :],
                    other=0.0).to(tl.float32)
        x = tl.load(h_ptr + m * stride_h + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + m * stride_o + offs_c, acc, mask=mask_c)


@triton.jit
def _cand_logits_nvfp4_kernel(h_ptr, cand_ptr, pk_ptr, sc_ptr, lut_ptr, out_ptr, gscale, K, stride_h, stride_c, stride_o,
                              H: tl.constexpr, BLOCK_C: tl.constexpr):
    """out[m, j] = h[m] . dequant(W[cand[m, j]]) for one row m and BLOCK_C candidates per program (raw NVFP4 rows)."""
    m = tl.program_id(0)
    jb = tl.program_id(1)
    offs_c = jb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < K
    rows = tl.load(cand_ptr + m * stride_c + offs_c, mask=mask_c, other=0).to(tl.int64)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for g in range(0, H // 16):                       # one NVFP4 group of 16 values (8 packed bytes, 1 fp8 scale)
        offs_b = tl.arange(0, 8)
        pk = tl.load(pk_ptr + rows[:, None] * (H // 2) + g * 8 + offs_b[None, :], mask=mask_c[:, None], other=0)
        lo = tl.load(lut_ptr + (pk & 0x0F).to(tl.int32))
        hi = tl.load(lut_ptr + (pk >> 4).to(tl.int32))
        sc = tl.load(sc_ptr + rows * (H // 16) + g, mask=mask_c, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        h_lo = tl.load(h_ptr + m * stride_h + g * 16 + 2 * offs_b).to(tl.float32)
        h_hi = tl.load(h_ptr + m * stride_h + g * 16 + 2 * offs_b + 1).to(tl.float32)
        acc += tl.sum(lo * h_lo[None, :] + hi * h_hi[None, :], axis=1) * sc
    tl.store(out_ptr + m * stride_o + offs_c, acc * gscale, mask=mask_c)


class DraftHead2:
    def __init__(self, lm_head, vocab: int, basis_path: str, k: int, model_dir: str | None):
        w = getattr(lm_head, "weight", None)
        self.V = int(vocab)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.dtype in (torch.bfloat16, torch.float16) \
                and w.shape[0] >= self.V:
            self.fmt = "dense"
            self.W = w                                     # the (shared) head weight itself, [V_padded, H]; rows < V used
            self.H = int(w.shape[1])
            self.device = w.device
        else:
            self.fmt = "nvfp4"
            from safetensors import safe_open
            assert model_dir, "draft head2: head is not a dense matrix and the checkpoint path is unknown"
            wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
            assert "lm_head.weight_scale" in wm, "draft head2: unsupported lm_head format (not dense, not NVFP4)"
            dev = w.device if isinstance(w, torch.Tensor) else torch.device("cuda")
            with safe_open(os.path.join(model_dir, wm["lm_head.weight"]), framework="pt", device="cpu") as f:
                self.pk = f.get_tensor("lm_head.weight").to(dev)                                # [V, H/2] u8
                self.sc = f.get_tensor("lm_head.weight_scale").view(torch.uint8).to(dev)         # [V, H/16] e4m3 bits
                self.g = float(f.get_tensor("lm_head.weight_scale_2").float().max())
            self.H = int(self.pk.shape[1]) * 2
            self.device = dev
            self.lut = torch.tensor(_LUT, dtype=torch.float32, device=dev)
        self.fp4 = os.environ.get("MBX_DRAFT_HEAD2_SCORER", "fp8") == "fp4"
        self.k = min(int(k), self.V)
        self.set_basis(basis_path)

    def dequant(self, a: int, b: int) -> torch.Tensor:
        """Rows a..b of the vocabulary head as float32 [b - a, H]."""
        if self.fmt == "dense":
            return self.W[a:b].float()
        p = self.pk[a:b]
        codes = torch.stack(((p & 0x0F), (p >> 4)), dim=-1).view(p.shape[0], -1)
        sc = self.sc[a:b].view(torch.float8_e4m3fn).float()
        w = self.lut[codes.long()].view(b - a, -1, 16) * sc.unsqueeze(-1) * self.g
        return w.reshape(b - a, -1)

    def set_basis(self, basis_path: str) -> None:
        """(Re)build the scorer from {"P": [H, r]} (U = W P) or {"P", "Q"} (z = h P, U = W Q)."""
        device = self.device
        obj = torch.load(basis_path, map_location="cpu", weights_only=True)
        P = obj["P"].float().to(device)                                                             # [H, r]
        Q = obj["Q"].float().to(device) if "Q" in obj else P
        assert P.shape[0] == self.H and Q.shape == P.shape, f"basis {tuple(P.shape)} does not match hidden size {self.H}"
        self.P = P.to(torch.bfloat16).contiguous()
        r = P.shape[1]
        assert r % 16 == 0, "basis rank must be a multiple of 16 (Marlin K)"
        torch.cuda.empty_cache()
        U = torch.empty(self.V, r, dtype=torch.float32, device=device)
        for a in range(0, self.V, 16384):
            b = min(a + 16384, self.V)
            U[a:b] = self.dequant(a, b) @ Q
        holder = torch.nn.Module()
        holder.output_size_per_partition, holder.input_size_per_partition = self.V, r
        if self.fp4:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
            pk4, s4, g4 = _quant_nvfp4(U)
            holder.weight = torch.nn.Parameter(pk4, requires_grad=False)
            holder.weight_scale = torch.nn.Parameter(s4, requires_grad=False)
            holder.weight_global_scale = torch.nn.Parameter(torch.tensor(g4, dtype=torch.float32, device=device),
                                                            requires_grad=False)
            holder.params_dtype = torch.bfloat16
            prepare_fp4_layer_for_marlin(holder)
        else:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
            sa = U.abs().amax(1).clamp(min=1e-12) / 448.0
            holder.weight = torch.nn.Parameter((U / sa[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn),
                                               requires_grad=False)
            holder.weight_scale = torch.nn.Parameter(sa, requires_grad=False)
            holder.orig_dtype = torch.bfloat16
            prepare_fp8_layer_for_marlin(holder, size_k_first=False)
        del U
        torch.cuda.empty_cache()
        self.scorer = holder
        self.r = r
        self.basis_path = basis_path
        log.info("mbx draft head2: %s head (%d x %d), basis %s, rank %d, %d candidates, %s scorer %.0f MB",
                 self.fmt, self.V, self.H, basis_path, r, self.k, "nvfp4" if self.fp4 else "fp8",
                 self.V * r / (2e6 if self.fp4 else 1e6))

    def logits(self, h: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        n = h.shape[0]
        hc = h.to(torch.bfloat16).contiguous()
        z = hc @ self.P                                                                             # [n, r]
        sc = self.scorer
        if self.fp4:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
            scores = apply_fp4_marlin_linear(z, sc.weight, sc.weight_scale, sc.weight_global_scale, None, self.V, self.r)
        else:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
            scores = apply_fp8_marlin_linear(z, sc.weight, sc.weight_scale, None, self.V, self.r, None)
        cand = scores.topk(self.k, dim=-1, sorted=False).indices.to(torch.int32).contiguous()      # [n, k]
        cl = torch.empty(n, self.k, dtype=torch.float32, device=h.device)
        if n > 0:
            if self.fmt == "dense":
                BLOCK_C = 32
                _cand_logits_dense_kernel[(n, triton.cdiv(self.k, BLOCK_C))](
                    hc, cand, self.W, cl, self.k, hc.stride(0), cand.stride(0), self.W.stride(0), cl.stride(0),
                    H=self.H, BLOCK_C=BLOCK_C, BLOCK_H=128)
            else:
                BLOCK_C = 64
                _cand_logits_nvfp4_kernel[(n, triton.cdiv(self.k, BLOCK_C))](
                    hc, cand, self.pk, self.sc, self.lut, cl, self.g, self.k, hc.stride(0), cand.stride(0), cl.stride(0),
                    H=self.H, BLOCK_C=BLOCK_C)
        if scale != 1.0:
            cl = cl * scale
        out = torch.full((n, self.V), float("-inf"), dtype=h.dtype, device=h.device)
        out.scatter_(1, cand.long(), cl.to(out.dtype))
        return out


def install(draft_model, vllm_config) -> None:
    """Patch 42 (MTPSpeculator.load_draft_model, after vLLM shares the target's lm_head into the drafter)."""
    path = os.environ.get("MBX_DRAFT_HEAD2")
    if not path:
        return
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        if get_tensor_model_parallel_world_size() != 1:
            log.warning("mbx draft head2: tensor parallel > 1 is not supported — off")
            return
    except Exception:  # noqa: BLE001
        pass
    lp = draft_model.logits_processor
    vocab = getattr(lp, "org_vocab_size", None) or lp.vocab_size
    model_dir = str(vllm_config.model_config.model)
    draft_model._mbx_head2 = DraftHead2(draft_model.lm_head, vocab, path, int(os.environ.get("MBX_DRAFT_HEAD2_K", "2048")),
                                        model_dir)


def _maybe_dump(model, h) -> None:
    path = os.environ.get("MBX_DRAFT_HEAD2_DUMP")
    if not path or torch.cuda.is_current_stream_capturing():
        return
    model._mbx_dump_calls = getattr(model, "_mbx_dump_calls", 0) + 1
    if model._mbx_dump_calls <= 2000:
        return
    buf = getattr(model, "_mbx_dump", None)
    if buf is None:
        buf = model._mbx_dump = []
    cap = int(os.environ.get("MBX_DRAFT_HEAD2_DUMP_ROWS", "8192"))
    n = sum(x.shape[0] for x in buf)
    if n >= cap:
        return
    buf.append(h[:4].detach().to("cpu", torch.bfloat16))
    if len(buf) % 50 == 0 or n + buf[-1].shape[0] >= cap:
        torch.save(torch.cat(buf), path)


def logits(model, hidden_states):
    """Patch 42 (Qwen4ExpMTP.compute_logits): the two-stage head's logits, or None for the stock path."""
    _maybe_dump(model, hidden_states)
    dh = getattr(model, "_mbx_head2", None)
    if dh is None:
        return None
    lp = model.logits_processor
    if getattr(lp, "soft_cap", None) is not None:
        return None
    return dh.logits(hidden_states, lp.scale)
