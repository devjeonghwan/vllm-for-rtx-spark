"""Finite source/metadata contracts; no Torch/vLLM/Triton/native imports.

Uses VLLM_RTX_SPARK_SOURCE_DIR for current composed-source behavior. These checks are
not a substitute for native kernels, prefix restore or model validation.
"""
import ast
from collections.abc import Sequence
import copy
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from support import ROOT as PROJECT, SOURCE as SITE
KERNELS = PROJECT / "src/kernels/dflash2"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def owner(t, name):
    return next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == name)


def fn(t, name, cls=None):
    body = t.body if cls is None else owner(t, cls).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)




def extract_functions(path, names, namespace):
    t = tree(path)
    nodes = [copy.deepcopy(fn(t, name)) for name in names]
    for node in nodes:
        node.decorator_list = []
    mod = ast.Module(body=nodes, type_ignores=[])
    exec(compile(mod, "source-only-metadata-fixture", "exec"), namespace)
    return namespace


class TensorMetadata:
    def __init__(self, shape, dtype="bf16", device=None, stride=1):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or NS(type="cuda")
        self._stride = stride

    def stride(self, axis):
        return self._stride


class Contracts(unittest.TestCase):
    def test_vendored_kernels_keep_their_license_notices(self):
        for name in ("__init__.py", "candidate_edges.py", "kv_fusion.py"):
            text = (KERNELS / name).read_text(encoding="utf-8")
            if name != "__init__.py":
                self.assertIn("Apache", text[:1200])
        self.assertIn("Apache License", (KERNELS / "LICENSE.Apache-2.0").read_text(encoding="utf-8"))

    def test_kv_metadata_admission_and_fallback(self):
        scope = extract_functions(KERNELS / "kv_fusion.py", ("is_supported", "can_materialize"),
            {"torch": NS(bfloat16="bf16", int64="int64"), "HAS_TRITON": True})
        rotary = NS(rotary_dim=128)
        model = NS(config=NS(architectures=["DFlash2DraftModel"], hidden_size=2560),
            _fused_kv_weight=TensorMetadata((10240, 2560)), _fused_kv_bias=None,
            _num_attn_layers=5, _num_kv_heads=8, _head_dim=128, _attn_layers=[None]*5,
            _rope_is_neox=True, layers=[NS(self_attn=NS(rotary_emb=rotary))],
            _k_norm_weights=TensorMetadata((5, 128)))
        self.assertTrue(scope["is_supported"](model, 8192))
        for path, value in (("config.architectures", ["DFlashDraftModel"]),
                ("config.hidden_size", 1280), ("_fused_kv_bias", object()),
                ("_num_attn_layers", 4), ("_num_kv_heads", 4), ("_head_dim", 64),
                ("_rope_is_neox", False), ("_fused_kv_weight.dtype", "fp32"),
                ("_fused_kv_weight.device.type", "cpu"), ("_fused_kv_weight.shape", (10240, 1280))):
            fake = copy.deepcopy(model)
            parent, _, leaf = path.rpartition(".")
            dst = fake
            for field in parent.split(".") if parent else ():
                dst = getattr(dst, field)
            setattr(dst, leaf, value)
            self.assertFalse(scope["is_supported"](fake, 8192), path)
        self.assertFalse(scope["is_supported"](model, 0))
        states = TensorMetadata((16, 2560))
        pos = TensorMetadata((16,), "int64", device=states.device)
        model._dflash2_sg_kv = {"capacity": 8192, "dtype": states.dtype, "device": states.device}
        self.assertTrue(scope["can_materialize"](model, states, pos))
        pos._stride = 2
        self.assertFalse(scope["can_materialize"](model, states, pos))
        pos._stride = 1
        states.shape = (8193, 2560)
        self.assertFalse(scope["can_materialize"](model, states, pos))

    def test_kv_import_and_fallback_without_triton(self):
        """Execute the complete helper import using vLLM's no-Triton boundary."""
        fake_torch = ModuleType("torch")
        fake_torch.Tensor = TensorMetadata
        fake_vllm = ModuleType("vllm")
        fake_vllm.__path__ = []
        utils = ModuleType("vllm.triton_utils")
        utils.HAS_TRITON = False
        utils.triton = NS(jit=lambda function: function)
        utils.tl = NS(constexpr=object())
        scope = {"__name__": "isolated_no_triton_dflash2_helper"}
        def import_guard(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError("No direct Triton package available")
            return original_import(name, *args, **kwargs)
        import builtins
        original_import = builtins.__import__
        with patch.dict(sys.modules, {"torch": fake_torch, "vllm": fake_vllm, "vllm.triton_utils": utils}), patch("builtins.__import__", import_guard):
            exec(compile((KERNELS / "kv_fusion.py").read_text(encoding="utf-8"), "production_helper_no_triton", "exec"), scope)
            self.assertFalse(scope["is_supported"](NS(), 8192))
            self.assertIsNone(scope["prepare_if_supported"](NS(), 8192))
    def test_recipe_external_rope_guard_and_mtp_override(self):
        cls = owner(tree(PROJECT / "src/recipes/qwen3_8_flash_next/__init__.py"), "TargetAndDraftOverrides")
        scope = {"copy": copy}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "source-only-overrides", "exec"), scope)
        class Config(NS):
            def update(self, fields):
                self.__dict__.update(fields)
        override = scope["TargetAndDraftOverrides"]({"text_config": {"max_position_embeddings": 409600}})
        draft = Config(architectures=["DFlash2DraftModel"], num_attention_heads=32, max_position_embeddings=262144)
        before = copy.deepcopy(draft.__dict__)
        self.assertIs(override(draft), draft)
        self.assertEqual(draft.__dict__, before)
        mtp = Config(architectures=["Qwen4ExpMTP"], num_attention_heads=32)
        self.assertEqual(override(mtp).max_position_embeddings, 409600)

    @unittest.skipUnless(SITE, "Set VLLM_RTX_SPARK_SOURCE_DIR for composed-source contracts")
    def test_drop_policy_keeps_mtp_and_changes_only_context_drafts(self):
        path = SITE / "vllm/config/speculative.py"
        t = tree(path)
        methods = [copy.deepcopy(fn(t, n, "SpeculativeConfig")) for n in
            ("use_eagle", "use_dflash", "use_dspark", "needs_eagle_block_drop", "use_eagle_block_drop")]
        cls = ast.ClassDef(name="Policy", bases=[], keywords=[], body=methods, decorator_list=[])
        mod = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        scope = {}
        exec(compile(mod, "source-only-policy", "exec"), scope)
        for method in ("eagle", "eagle3", "mtp", "dflash", "dspark", "ngram"):
            for disabled in (False, True):
                value = scope["Policy"]()
                value.method, value.disable_eagle_block_drop = method, disabled
                self.assertEqual(value.use_eagle_block_drop(), method in ("eagle", "eagle3", "mtp") and not disabled)

    @unittest.skipUnless(SITE, "Set VLLM_RTX_SPARK_SOURCE_DIR for composed-source contracts")
    def test_external_draft_request_context_boundary(self):
        node = fn(tree(SITE / "vllm/v1/engine/input_processor.py"), "_validate_dflash2_context_budget", "InputProcessor")
        class ValidationError(ValueError):
            def __init__(self, message, **kwargs):
                super().__init__(message)
        scope = {"VLLMValidationError": ValidationError}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "source-only-request-boundary", "exec"), scope)
        check = scope[node.name]
        processor = NS(speculative_config=NS(method="dflash", num_speculative_tokens=7,
            draft_model_config=NS(architectures=["DFlash2DraftModel"], max_model_len=262144,
                hf_config=NS(max_position_embeddings=262144))),
            model_config=NS(architectures=["Qwen4ExpForConditionalGeneration"], max_model_len=409600))
        for tokens in (3,7):
            processor.speculative_config.num_speculative_tokens=tokens
            budget=262144-(tokens+1)
            check(processor,NS(max_tokens=128),budget-128)
            with self.assertRaisesRegex(ValidationError,"external draft context limit 262144"):
                check(processor,NS(max_tokens=129),budget-128)
        self.assertEqual(processor.model_config.max_model_len,409600)
        # Existing None resolution happens before the guard; inspect its actual
        # source order and then execute the same resolved finite request budget.
        process=ast.unparse(fn(tree(SITE / "vllm/v1/engine/input_processor.py"),"process_inputs","InputProcessor"))
        self.assertLess(process.index("if sampling_params.max_tokens is None:"),
                        process.index("self._validate_dflash2_context_budget(sampling_params, prompt_len)"))
        resolved=processor.model_config.max_model_len-100
        with self.assertRaises(ValidationError):
            check(processor,NS(max_tokens=resolved),100)
        for kind in ("none","mtp","other_target","other_draft"):
            other=copy.deepcopy(processor)
            if kind=="none":other.speculative_config=None
            elif kind=="mtp":other.speculative_config.method="mtp"
            elif kind=="other_target":other.model_config.architectures=["Qwen3ForCausalLM"]
            else:other.speculative_config.draft_model_config.architectures=["DFlashDraftModel"]
            check(other,NS(max_tokens=1000000),1000000)

    @unittest.skipUnless(SITE, "Set VLLM_RTX_SPARK_SOURCE_DIR for composed-source contracts")
    def test_precise_target_draft_admission(self):
        t = tree(SITE / "vllm/model_executor/models/config.py")
        node = fn(t, "_flash_next_dflash2_admitted")
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "source-only-admission", "exec"), scope)
        check = scope[node.name]
        draft = NS(architectures=["DFlash2DraftModel"], hf_config=NS(num_hidden_layers=5, hidden_size=2560,
            num_target_layers=48, dflash_config={"target_layer_ids": [5,15,25,35,45]}))
        value = NS(speculative_config=NS(method="dflash", draft_model_config=draft),
            model_config=NS(hf_text_config=NS(hc_count=4, num_hidden_layers=48, hidden_size=2560)),
            parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                prefill_context_parallel_size=1, decode_context_parallel_size=1))
        self.assertTrue(check(value))
        for method in ("eagle", "eagle3", "mtp", "dspark", "draft_model"):
            other = copy.deepcopy(value)
            other.speculative_config.method = method
            self.assertFalse(check(other), method)
        for field in vars(value.parallel_config):
            other = copy.deepcopy(value)
            setattr(other.parallel_config, field, 2)
            self.assertFalse(check(other), field)
        other = copy.deepcopy(value)
        other.speculative_config.draft_model_config.hf_config.dflash_config["target_layer_ids"] = [6,16,26,36,46]
        self.assertFalse(check(other))

    @unittest.skipUnless(SITE, "Set VLLM_RTX_SPARK_SOURCE_DIR for composed-source contracts")
    def test_swa_live_window_alias_indices_and_cow_metadata(self):
        t = tree(SITE / "vllm/v1/core/single_type_kv_cache_manager.py")
        names = ("eagle_drop_tokens", "_contiguous_blocks_for_hit", "find_longest_cache_hit",
            "_find_fine_grained_cache_hit", "reachable_block_mask", "_publish_partial_tail",
            "_partial_tail_boundaries", "_reachable_hit_boundaries")
        base = ast.ClassDef(name="Base", bases=[], keywords=[], decorator_list=[], body=[
            copy.deepcopy(fn(t, name, "SingleTypeKVCacheManager")) for name in ("eagle_drop_tokens", "_apply_cow")])
        swa = ast.ClassDef(name="SWA", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], decorator_list=[],
            body=[copy.deepcopy(fn(t, name, "SlidingWindowManager")) for name in names])
        mod = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), base, swa], type_ignores=[]))
        class Block:
            def __init__(self, i, null=False):
                self.block_id, self.is_null, self.ref_cnt = i, null, 1
        class Spec:
            def __init__(self, size, window):
                self.block_size, self.sliding_window = size, window
        class Pool:
            def __init__(self):
                self.hash_block_size, self.null_block = 2, Block(-1, True)
                self.entries, self.published = {}, []
            def get_cached_block(self, key, groups):
                return self.entries.get(key)
            def cache_partial_block(self, **kwargs):
                self.published.append(kwargs)
        scope = {"Sequence": Sequence, "SlidingWindowSpec": Spec, "cdiv": lambda a,b: (a+b-1)//b}
        exec(compile(mod, "source-only-SWA-metadata", "exec"), scope)
        Base, SWA = scope["Base"], scope["SWA"]
        Base.supports_fine_grained_hash_lookup = False
        SWA.supports_fine_grained_hash_lookup = True
        pool = Pool()
        b0, b1, b2 = Block(10), Block(11), Block(12)
        pool.entries = {1:[b0],3:[b1],5:[b2]}
        def hit(eagle=False):
            return SWA._find_fine_grained_cache_hit(list(range(20)),12,[6],pool,Spec(4,8),eagle,2)
        blocks, length = hit()
        self.assertEqual(length, 12)
        self.assertEqual(blocks[0], [pool.null_block,b1,b2])
        pool.entries = {1:[b0],2:[b1],5:[b2]}
        blocks, length = hit()
        self.assertEqual((blocks,length), (([b0,b1],),6))
        pool.entries = {5:[b2]}
        self.assertEqual(hit(), (([],),0))
        pool.entries = {1:[b0],3:[b1],5:[b2]}
        self.assertEqual(hit(True), (([b0,b1],),8))
        self.assertEqual(SWA.eagle_drop_tokens(4,2),4)
        value = SWA()
        value.block_pool, value.block_size, value.scheduler_block_size = Pool(),4,8
        value.use_eagle, value.kv_cache_group_id = False,6
        req = NS(num_prompt_tokens=22,shared_prefix_boundary=10,request_id="r")
        self.assertEqual(value._reachable_hit_boundaries(req), [20,10,16])
        value.use_eagle=True
        self.assertEqual(value._reachable_hit_boundaries(req), [16,10])
        value.req_to_blocks = {"r":[Block(x) for x in range(8)]}
        value._publish_partial_tail(req,22)
        self.assertEqual([x["num_tokens"] for x in value.block_pool.published], [22])
        value.block_pool.published=[]
        value.req_to_blocks["r"][5].is_null=True
        value._publish_partial_tail(req,22)
        self.assertEqual(value.block_pool.published, [])
        value = Base()
        src, dst = Block(20),Block(21)
        value.req_to_blocks, value._pending_cow_copies = {"r":[b0,src]},[]
        value._apply_cow("r",1,src,dst)
        self.assertIs(value.req_to_blocks["r"][1],dst)
        self.assertEqual(value._pending_cow_copies,[(src,dst)])
        self.assertEqual((src.ref_cnt,dst.ref_cnt),(1,2))







if __name__ == "__main__":
    unittest.main()
