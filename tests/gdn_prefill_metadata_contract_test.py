"""CPU contracts for backend-owned GDN prefill metadata."""
import ast
from types import SimpleNamespace as NS
import unittest

from support import source_path


class PreparedSourceContracts(unittest.TestCase):
    def test_actual_backend_controls_metadata_including_triton_fallback(self):
        tree = ast.parse(source_path("vllm/v1/attention/backends/gdn_attn.py").read_text(encoding="utf-8"))
        builder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GDNAttentionMetadataBuilder")
        build = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "build")
        branch = next(n for n in ast.walk(build) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "self.gdn_prefill_backend == 'aiter_flydsl'")
        # The field is the resolved backend, including FlashInfer -> Triton
        # fallback. FLA/CuTe still receive their required arrays.
        for backend in ("flashinfer", "triton", "cutedsl"):
            calls = []

            def make_metadata(*args):
                calls.append(args)
                return "indices", "offsets"

            env = {"self": NS(gdn_prefill_backend=backend, _build_chunk_metadata=make_metadata),
                   "prefill_query_start_loc": "gpu starts", "prefill_query_start_loc_cpu": "cpu starts",
                   "query_start_loc": NS(device="cuda"), "chunk_indices": None, "chunk_offsets": None}
            exec(compile(ast.Module(body=[branch], type_ignores=[]), "actual-gdn-branch", "exec"), env)
            with self.subTest(backend=backend):
                expected = backend != "flashinfer"
                self.assertEqual(len(calls), int(expected))
                self.assertEqual((env["chunk_indices"], env["chunk_offsets"]),
                                 ("indices", "offsets") if expected else (None, None))

    def test_actual_flashinfer_consumer_does_not_read_the_removed_fields(self):
        tree = ast.parse(source_path("vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChunkGatedDeltaRule")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_cuda")
        reads = [n.id for n in ast.walk(forward) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        self.assertNotIn("chunk_indices", reads)
        self.assertNotIn("chunk_offsets", reads)


if __name__ == "__main__":
    unittest.main()
