"""Optional selector projection overlap for the Qwen DFlash2 draft path."""


def validate_config(config, enabled):
    if not enabled:
        return
    speculative = config.speculative_config
    draft = getattr(speculative, "draft_model_config", None)
    draft_hf = getattr(draft, "hf_config", None)
    target_hf = getattr(config.model_config, "hf_config", None)
    if (getattr(speculative, "method", None) != "dflash"
            or "DFlash2DraftModel" not in (getattr(draft_hf, "architectures", None) or ())
            or "Qwen4ExpForConditionalGeneration" not in (getattr(target_hf, "architectures", None) or ())):
        raise ValueError("dflash2_head_overlap requires the Qwen4Exp target and DFlash2 draft")
    parallel = config.parallel_config
    if any(getattr(parallel, name) != 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "prefill_context_parallel_size", "decode_context_parallel_size")):
        raise ValueError("dflash2_head_overlap supports TP1/PP1/DP1/CP1")


def create_overlap(config, device):
    # Read only this opt-in key, as other recipes can also load DFlash models.
    additional = getattr(config, "additional_config", None) or {}
    options = additional.get("rtx_spark", {}) if isinstance(additional, dict) else {}
    if not isinstance(options, dict):
        return None
    nested = options.get("optimizations", {})
    key = "dflash2_head_overlap"
    values = ([options[key]] if key in options else []) + (
        [nested[key]] if isinstance(nested, dict) and key in nested else [])
    if not values:
        return None
    if any(type(value) is not bool for value in values):
        raise TypeError("rtx_spark.dflash2_head_overlap must be bool")
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError("Conflicting RTX Spark setting: dflash2_head_overlap")
    if not values[0]:
        return None
    validate_config(config, True)
    return SelectorProjectionOverlap(device)


class SelectorProjectionOverlap:
    """Own one serial query's side stream, events and outstanding tensors.

    Construction is startup-only, before query Graph capture. Every captured
    query forks after its hidden states are ready and joins before edge scoring.
    A failure before the join retains the owners and prevents another start.
    """

    def __init__(self, device):
        import torch
        from vllm.model_executor.models.qwen3_dflash2 import CandidateSelector

        if device.type != "cuda":
            raise ValueError("dflash2_head_overlap requires a CUDA device")
        self.stream = torch.cuda.Stream(device=device)
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.pending = False
        self.hidden = None
        self.projected = None
        self.selector = None
        self.projector = None
        self._selector_type = CandidateSelector
        self._selector_forward = CandidateSelector.forward
        self._selector_call = CandidateSelector.__call__

    def _supported(self, selector):
        import torch

        hooks = torch.nn.modules.module
        return (type(selector) is self._selector_type
                and getattr(selector.forward, "__func__", None) is self._selector_forward
                and type(selector).__call__ is self._selector_call
                and getattr(selector, "_compiled_call_impl", None) is None
                and not selector._forward_pre_hooks and not selector._forward_hooks
                and not hooks._global_forward_pre_hooks and not hooks._global_forward_hooks)

    def start(self, hidden, selector):
        import torch

        if self.pending:
            raise RuntimeError("Unjoined DFlash2 selector projection; restart the worker")
        # Parent module hooks/overrides can change the projection's inputs or
        # scores. Keep their original __call__ path instead of bypassing them.
        if not self._supported(selector):
            return False
        self.pending = True
        self.hidden = hidden
        self.selector = selector
        self.projector = selector.hidden_projection
        main = torch.cuda.current_stream(hidden.device)
        self.ready.record(main)
        self.stream.wait_event(self.ready)
        # Protect the input even if a projection raises after submitting work.
        hidden.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            self.projected = self.projector(hidden)
            self.done.record(self.stream)
        return True

    def score(self, speculator, candidate_ids, unary_logits):
        import torch
        from vllm.model_executor.models.qwen3_dflash2 import _score_edges

        if not self.pending or self.projected is None:
            raise RuntimeError("DFlash2 selector projection has not completed submission")
        main = torch.cuda.current_stream(self.projected.device)
        main.wait_event(self.done)
        self.projected.record_stream(main)
        selector = speculator.model.model.candidate_selector
        num_reqs = candidate_ids.shape[0]
        anchors = speculator.input_buffers.input_ids[
            :num_reqs * speculator.num_query_per_req:speculator.num_query_per_req]
        try:
            if (selector is not self.selector or selector.hidden_projection is not self.projector
                    or not self._supported(selector)):
                return selector(candidate_ids, unary_logits, self.hidden, anchors)
            return _score_edges(selector.predecessor_codebook, selector.successor_codebook,
                                candidate_ids, unary_logits, self.projected, anchors, selector.top_k)
        finally:
            # Both consumers now have stream lifetime tracking. A scoring error
            # can release Python references without reusing side-stream storage.
            self.projected = None
            self.hidden = None
            self.selector = None
            self.projector = None
            self.pending = False
