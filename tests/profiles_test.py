"""Memory and draft-KV profiles preserve the verified fast execution settings."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import re
import unittest

from pydantic import TypeAdapter, ValidationError
import yaml


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SIZES = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 20, 21,
    24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
]
DFLASH_CAPTURE_SIZES = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 20, 21,
    24, 28, 32, 36, 40, 44, 45, 48, 50, 52, 55, 56, 60, 64,
    65, 70, 72, 75, 80, 88, 96, 104, 112, 120, 128,
]
DFLASH_PROFILE = "gpu-96gb-dflash2.yaml"


def uniform_decode_routes(capture_sizes, minimum_depth, maximum_depth, max_requests):
    """Model the pinned runner's token-group lookup without importing Torch.

    The runner first selects the nearest token-count group, then checks its
    query width. A matching width in a later group does not make a FULL hit.
    Unmatched uniform batches retain the existing PIECEWISE route.
    """
    by_tokens = defaultdict(set)
    max_tokens = max_requests * (maximum_depth + 1)
    for size in capture_sizes:
        for width in range(minimum_depth + 1, maximum_depth + 2):
            requests = (size + width - 1) // width
            tokens = requests * width
            if requests <= max_requests and tokens <= min(max_tokens, max(capture_sizes)):
                by_tokens[tokens].add((width, requests))

    routes = {}
    for depth in range(minimum_depth, maximum_depth + 1):
        width = depth + 1
        for requests in range(1, max_requests + 1):
            nearest = min(tokens for tokens in by_tokens if tokens >= width * requests)
            full = any(w == width and n >= requests for w, n in by_tokens[nearest])
            routes[depth, requests] = "FULL" if full else "PIECEWISE"
    return routes


class ProfileTests(unittest.TestCase):
    def test_memory_tiers_preserve_the_fast_pipeline(self):
        directory = ROOT / "configs/qwen3.8-flash-next"
        budgets = {"gpu-90gb": 12_000_000_000, "gpu-96gb": 18_000_000_000, "gpu-108gb": 27_000_000_000}
        draft_kv = {"bf16": ["48"], "fp8": []}
        expected_files = {f"{name}-draft-kv-{dtype}.yaml" for name in budgets for dtype in draft_kv}
        self.assertEqual({path.name for path in directory.iterdir()}, expected_files | {DFLASH_PROFILE})
        common = None
        for name, budget in budgets.items():
            for dtype, skip_layers in draft_kv.items():
                filename = f"{name}-draft-kv-{dtype}.yaml"
                with self.subTest(profile=filename):
                    profile = yaml.safe_load((directory / filename).read_text(encoding="utf-8"))
                    self.assertIs(type(profile["kv-cache-memory-bytes"]), int)
                    self.assertEqual(profile["kv-cache-memory-bytes"], budget)
                    self.assertIs(type(profile["kv-cache-dtype-skip-layers"]), list)
                    self.assertEqual(profile["kv-cache-dtype-skip-layers"], skip_layers)
                    self.assertTrue(all(type(layer) is str for layer in profile["kv-cache-dtype-skip-layers"]))
                    normalized = deepcopy(profile)
                    del normalized["kv-cache-memory-bytes"]
                    del normalized["kv-cache-dtype-skip-layers"]
                    if common is None:
                        common = normalized
                    self.assertEqual(normalized, common)

        expected = {
            "load-format": "mbx",
            "distributed-executor-backend": "mp",
            "gpu-memory-utilization": 0.88,
            "kv-cache-dtype": "fp8",
            "max-model-len": 409600,
            "block-size": 3456,
            "prefix-match-unit": 64,
            "max-num-seqs": 16,
            "max-num-batched-tokens": 8192,
            "enable-prefix-caching": True,
            "async-scheduling": True,
            "use-replayssm": True,
            "dtype": "bfloat16",
            "tensor-parallel-size": 1,
            "pipeline-parallel-size": 1,
            "mamba-cache-mode": "align",
        }
        self.assertEqual({key: common[key] for key in expected}, expected)
        self.assertEqual(common["kernel-config"], {
            "linear_backend_per_quant": {"mxfp8": "b12x"}, "moe_backend": "marlin",
        })
        self.assertEqual(common["engram-config"], {"cpu_offload": False})
        self.assertEqual(common["additional-config"], {
            "gdn_prefill_backend": "flashinfer",
            "rtx_spark": {
                "model": "qwen3.8-flash-next", "draft_fp8": True,
                "draft_head2": True, "draft_min_depth": 3,
            },
        })
        self.assertEqual(common["speculative-config"], {
            "method": "mtp", "num_speculative_tokens": 7,
            "rejection_sample_method": "block", "draft_sample_method": "probabilistic",
            "max_model_len": 409600,
        })
        self.assertEqual(common["hf-overrides"], {"text_config": {
            "max_position_embeddings": 409600,
            "rope_parameters": {
                "rope_type": "yarn", "factor": 1.5625,
                "original_max_position_embeddings": 262144, "rope_theta": 10000000,
                "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        }})
        compilation = common["compilation-config"]
        self.assertEqual(compilation, {
            "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": CAPTURE_SIZES,
            "compile_ranges_endpoints": [32],
        })
        # These seven routes are existing sparse-list behavior, not new coverage.
        routes = uniform_decode_routes(CAPTURE_SIZES, 3, 7, 16)
        self.assertEqual(len(routes), 80)
        self.assertEqual({shape for shape, mode in routes.items() if mode == "PIECEWISE"}, {
            (3, 15), (4, 9), (4, 11), (4, 14), (5, 9), (5, 13), (6, 9),
        })
        self.assertEqual(sum(mode == "FULL" for mode in routes.values()), 73)

        # NVMe PLE is supplied by runtime.bootstrap (covered by runtime_test),
        # rather than invented vLLM YAML flags or a CPU offload substitution.
        self.assertTrue(all(not key.startswith("MBX_") for key in common))
        self.assertTrue({"env", "environment", "cpu-offload-gb", "swap-space"}.isdisjoint(common))

    def test_dflash2_shared_profile_preserves_the_tested_scope(self):
        directory = ROOT / "configs/qwen3.8-flash-next"
        text = (directory / DFLASH_PROFILE).read_text(encoding="utf-8")
        profile = yaml.safe_load(text)
        mtp = yaml.safe_load((directory / "gpu-96gb-draft-kv-bf16.yaml").read_text(encoding="utf-8"))
        # The extra draft-only head is explicit; target/KV residency, NVMe PLE,
        # host memory allocation and the original execution flags are retained.
        different = {key for key in set(profile) | set(mtp) if profile.get(key) != mtp.get(key)}
        self.assertEqual(different, {"speculative-config", "compilation-config", "additional-config"})
        self.assertEqual(profile["kv-cache-memory-bytes"], 18_000_000_000)
        self.assertEqual(profile["kv-cache-dtype"], "fp8")
        self.assertEqual(profile["kv-cache-dtype-skip-layers"], ["48"])
        self.assertEqual(profile["additional-config"], {
            "gdn_prefill_backend": "flashinfer",
            "rtx_spark": {
                "model": "qwen3.8-flash-next", "draft_fp8": False,
                "draft_head2": False, "draft_min_depth": 3,
                "dflash2_nvfp4_head": True,
            },
        })
        spec = profile["speculative-config"]
        self.assertEqual(spec, {
            "method": "dflash",
            "model": "incoai/Qwen3.8-Flash-Next-DFlash2",
            "revision": "a22a1744e85c17364dcc54a6a420af7135d4c7f6",
            "num_speculative_tokens": 7,
            "rejection_sample_method": "block",
            "draft_sample_method": "probabilistic",
            "max_model_len": 262144,
            "draft_load_config": {"load_format": "safetensors"},
            "kv_cache_dtype": "bfloat16",
            "attention_backend": "TRITON_ATTN",
            "num_speculative_tokens_per_batch_size": [[1, 7, 7], [8, 16, 4]],
        })
        # No draft RoPE override: the trained checkpoint config is preserved.
        self.assertNotIn("hf_overrides", spec)
        self.assertEqual(profile["compilation-config"], {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": DFLASH_CAPTURE_SIZES,
            "compile_ranges_endpoints": [32],
        })
        routes = uniform_decode_routes(DFLASH_CAPTURE_SIZES, 3, 7, 16)
        self.assertTrue(all(routes[4, requests] == "FULL" for requests in range(8, 17)))
        self.assertTrue(all(routes[7, requests] == "FULL" for requests in range(1, 17)))
        self.assertEqual(profile["port"], 8000)
        self.assertTrue({"worker-extension-cls", "per-request-spec-decode-metrics", "disable-log-stats",
                         "env", "environment", "cpu-offload-gb", "swap-space"}.isdisjoint(profile))
        self.assertIsNone(re.search(r"(?i)([a-z]:[\\/]|/home/|/tmp/|research[\\/]|api_control|MBX_)", text))

    def test_dflash2_schedule_survives_pydantic_tuple_conversion(self):
        path = ROOT / "configs/qwen3.8-flash-next" / DFLASH_PROFILE
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))["speculative-config"]
        # This is the pinned SpeculativeConfig field type. Pydantic converts
        # YAML list rows to tuples; the Tail admission must handle that shape.
        adapter = TypeAdapter(list[tuple[int, int, int]] | None)
        parsed = adapter.validate_python(spec["num_speculative_tokens_per_batch_size"])
        self.assertEqual(parsed, [(1, 7, 7), (8, 16, 4)])
        self.assertTrue(all(type(row) is tuple for row in parsed))
        self.assertEqual(tuple(tuple(row) for row in parsed), ((1, 7, 7), (8, 16, 4)))
        lookup = {batch: depth for start, end, depth in parsed for batch in range(start, end + 1)}
        self.assertEqual(lookup, {batch: 7 if batch < 8 else 4 for batch in range(1, 17)})
        for invalid in ([[1, 7]], [[1, 7, "invalid"]]):
            with self.subTest(schedule=invalid), self.assertRaises(ValidationError):
                adapter.validate_python(invalid)


if __name__ == "__main__":
    unittest.main()
