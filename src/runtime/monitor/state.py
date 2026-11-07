"""Recent request snapshots; no tensors, device reads, token-ID or state copies."""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from contextlib import aclosing
from io import StringIO
import math
import os
from pathlib import Path
import time

HISTORY = 100


def number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def interval(end, start):
    if number(end) is None or number(start) is None or end <= 0 or start <= 0 or end < start:
        return None
    return end - start


def rate(tokens, seconds):
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return number(tokens / seconds)


def config_path(args):
    path = getattr(args, "_rtx_spark_config_file", None) or getattr(args, "config", None)
    if not path:
        return ""
    try:
        return Path(os.path.relpath(path)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across different drives.
        return Path(path).as_posix()


def config_rows(config, args):
    model = config.model_config
    cache = config.cache_config
    spec = config.speculative_config
    options = (getattr(config, "additional_config", None) or {}).get("rtx_spark", {})
    kernel = getattr(config, "kernel_config", None)
    values = [
        ("Model", model.model), ("Address", f"{args.host or '0.0.0.0'}:{args.port}"),
        ("Context length", model.max_model_len), ("Dtype", str(model.dtype)),
        ("Target KV", cache.cache_dtype), ("KV budget (bytes)", cache.kv_cache_memory_bytes),
        ("KV precision skip layers", getattr(cache, "kv_cache_dtype_skip_layers", None)),
        ("Prefix cache", cache.enable_prefix_caching),
        ("Draft Model", getattr(getattr(spec, "draft_model_config", None), "model", None)),
        ("Max draft tokens", getattr(spec, "num_speculative_tokens", None)),
        ("Min draft depth", options.get("draft_min_depth")),
        ("FP8 draft", options.get("draft_fp8")), ("HEAD2", options.get("draft_head2")),
        ("Draft KV", options.get("draft_kv_cache_dtype")),
        ("Decode graph", str(config.compilation_config.cudagraph_mode)),
        ("MoE backend", getattr(kernel, "moe_backend", None)),
    ]
    return [[key, value] for key, value in values if value is not None]


class Monitor:
    """Owned by one API process and its event loop; retains scalars and full request text."""

    def __init__(self, engine, args, version=None):
        self.engine = engine
        self.version = version
        self.rows = OrderedDict()
        self.events = deque(maxlen=HISTORY)
        self.sequence = 0
        self.closed = False
        self.shutdown = None
        self.scheduler = {}
        self.config = config_rows(engine.vllm_config, args)
        self.profile = config_path(args)
        self.original_generate = engine.generate
        self.had_generate_override = "generate" in vars(engine)
        self.stats_manager = engine.logger_manager
        if self.stats_manager is None:
            raise ValueError("--monitor requires engine statistics")
        self.stats_manager.stat_loggers.append(self)
        engine.generate = self.generate
        self.event("INFO", "Monitor started")

    def event(self, level, message, key=None):
        self.events.appendleft({"created": time.time(), "level": level,
                                "message": message[:512], "key": key})

    # StatLoggerManager's narrow, pinned 0.32 subscriber interface.
    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if self.closed or scheduler_stats is None:
            return
        self.scheduler[engine_idx] = {
            "running": scheduler_stats.num_running_reqs,
            "waiting": scheduler_stats.num_waiting_reqs + scheduler_stats.num_skipped_waiting_reqs,
            "kv_usage": number(scheduler_stats.kv_cache_usage),
        }

    def log(self):
        pass

    def log_engine_initialized(self):
        pass

    def record_sleep_state(self, *args, **kwargs):
        pass

    def record_admission_rejection(self, reason):
        self.event("WARN", "Request admission rejected")

    def begin(self, request_id, params, prompt, kwargs):
        self.sequence += 1
        text = kwargs.get("prompt_text") or (prompt if isinstance(prompt, str) else
            prompt.get("prompt") if isinstance(prompt, dict) else None)
        row = {"key": self.sequence, "id": str(request_id)[:256], "created": time.time(),
               "status": "pending", "input": None, "output": None, "cached": None,
               "queue": None, "prefill": None, "decode": None, "ttft": None, "total": None,
               "prefill_rate": None, "decode_rate": None, "gen_rate": None,
               "finish_reason": None, "error": None, "prompt": text if isinstance(text, str) else None,
               "response": "",
               "single_sequence": getattr(params, "n", 1) == 1,
               "sampling": {key: number(getattr(params, key, None)) for key in ("n", "temperature", "top_p", "max_tokens")},
               "_delta": getattr(getattr(params, "output_kind", None), "name", "") == "DELTA",
               "_tokens": 0, "_start": time.monotonic()}
        if row["_delta"]:
            row["_response"] = StringIO()
        self.rows[row["key"]] = row
        while len(self.rows) > HISTORY:
            self.rows.popitem(last=False)
        return row

    def observe(self, row, output):
        prompt_ids = getattr(output, "prompt_token_ids", None)
        if prompt_ids is not None:
            row["input"] = len(prompt_ids)  # Borrow length; never retain token arrays.
        cached = getattr(output, "num_cached_tokens", None)
        row["cached"] = max(cached, 0) if cached is not None else None
        prompt = getattr(output, "prompt", None)
        if row["prompt"] is None and isinstance(prompt, str):
            row["prompt"] = prompt
        metrics = getattr(output, "metrics", None)
        if row["single_sequence"]:
            for completion in output.outputs:
                token_count = len(completion.token_ids)
                row["_tokens"] = row["_tokens"] + token_count if row["_delta"] else token_count
                text = completion.text or ""
                if row["_delta"]:
                    row["_response"].write(text)
                else:
                    row["response"] = text
                if completion.finish_reason is not None:
                    row["finish_reason"] = str(completion.finish_reason)[:128]
            # Engine counters include coalesced output/MTP bursts; do not count SSE chunks.
            row["output"] = (getattr(metrics, "num_generation_tokens", None)
                             if metrics is not None else row["_tokens"])
            if metrics is not None:
                row["queue"] = interval(metrics.scheduled_ts, metrics.queued_ts)
                row["prefill"] = interval(metrics.first_token_ts, metrics.scheduled_ts)
                row["decode"] = interval(metrics.last_token_ts, metrics.first_token_ts)
                row["ttft"] = number(metrics.first_token_latency) if metrics.first_token_ts > 0 else None
                if row["input"] is not None and row["cached"] is not None:
                    row["prefill_rate"] = rate(max(row["input"] - row["cached"], 0), row["prefill"])
                row["decode_rate"] = rate(row["output"] - 1, row["decode"]) if row["output"] and row["output"] > 1 else None
            row["total"] = max(time.monotonic() - row["_start"], 0)
            row["gen_rate"] = rate(row["output"], row["total"])
        row["status"] = "completed" if output.finished else "decode" if row["output"] else "pending"
        error = getattr(output, "error", None)
        if error is not None:
            row["status"] = "failed"
            row["error"] = str(error)[:512]

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        row = self.begin(request_id, sampling_params, prompt, kwargs)
        try:
            # Closing the wrapper must close the original generator and execute its abort/drain.
            async with aclosing(self.original_generate(prompt, sampling_params, request_id, **kwargs)) as stream:
                async for output in stream:
                    if not row.get("_observe_failed"):
                        try:
                            self.observe(row, output)
                        except Exception as exc:
                            # UI collection must never break a successful generation.
                            row["_observe_failed"] = True
                            row["monitor_error"] = type(exc).__name__
                            self.event("WARN", "Monitor collection stopped", row["key"])
                    if row.get("_observe_failed") and output.finished:
                        row["status"] = "completed"
                    yield output  # Preserve identity, ordering and output semantics.
            if row["status"] not in ("completed", "failed"):
                row["status"] = "cancelled"
        except (asyncio.CancelledError, GeneratorExit):
            if row["status"] != "completed":
                row["status"] = "cancelled"
            raise
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"[:512]
            raise
        finally:
            if row["total"] is None:
                row["total"] = max(time.monotonic() - row["_start"], 0)
            self.event("ERROR" if row["status"] == "failed" else "INFO",
                       f"{row['id']} · {row['status']}", row["key"])

    def public_row(self, row, detail=False):
        result = {key: value for key, value in row.items()
                  if not key.startswith("_") and (detail or key not in ("prompt", "response", "sampling", "error"))}
        if detail and "_response" in row:
            # Append streaming chunks without repeatedly copying the complete response.
            # Only a selected detail needs a materialized string; list polling stays small.
            result["response"] = row["_response"].getvalue()
        return result

    def snapshot(self):
        sched = list(self.scheduler.values())
        usages = [s["kv_usage"] for s in sched if s["kv_usage"] is not None]
        config = self.engine.vllm_config
        return {"status": "stopped" if self.closed else "error" if getattr(self.engine, "errored", False) else "ready",
                "version": self.version,
                "model": config.model_config.model, "profile": self.profile,
                "kv_budget_bytes": config.cache_config.kv_cache_memory_bytes,
                "kv_usage": sum(usages) / len(usages) if usages else None,
                "running": sum(s["running"] for s in sched) if sched else None,
                "waiting": sum(s["waiting"] for s in sched) if sched else None,
                "config": self.config, "requests": [self.public_row(row) for row in reversed(self.rows.values())],
                "events": list(self.events)}

    def close(self):
        self.closed = True
        if self in self.stats_manager.stat_loggers:
            self.stats_manager.stat_loggers.remove(self)
        if self.engine.generate == self.generate:
            if self.had_generate_override:
                self.engine.generate = self.original_generate
            else:
                del self.engine.generate
        self.shutdown = None
