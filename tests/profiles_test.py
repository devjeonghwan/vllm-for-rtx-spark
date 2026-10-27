"""Memory profiles must preserve the verified fast execution settings."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SIZES = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 20, 21,
    24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
]


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
        budgets = {"gpu-90gb": 12_000_000_000, "gpu-94gb": 16_000_000_000, "gpu-96gb": 18_000_000_000}
        self.assertEqual({path.name for path in directory.iterdir()}, {name + ".yaml" for name in budgets})
        common = None
        for name, budget in budgets.items():
            with self.subTest(profile=name):
                profile = yaml.safe_load((directory / (name + ".yaml")).read_text(encoding="utf-8"))
                self.assertIs(type(profile["kv-cache-memory-bytes"]), int)
                self.assertEqual(profile["kv-cache-memory-bytes"], budget)
                normalized = deepcopy(profile)
                del normalized["kv-cache-memory-bytes"]
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


if __name__ == "__main__":
    unittest.main()
