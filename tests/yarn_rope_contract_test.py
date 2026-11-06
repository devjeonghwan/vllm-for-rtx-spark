"""CPU-only RoPE contract regressions against a prepared pinned vLLM source.

VLLM_RTX_SPARK_SOURCE_DIR selects the actual prepared source; installed vLLM is never imported.
Uses actual constructors/frequency/cache formula; full cache allocation is
replaced with one placeholder row except for a tiny selected-position check.
No checkpoint, native DLL, CUDA operation or model generation is needed.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import copy
import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from support import SOURCE, SOURCE_REQUIRED, source_path


def reference(context, factor, dim=64, theta=10_000_000.):
    lower=max(0,math.floor(dim*math.log(context/(32*2*math.pi))/(2*math.log(theta))))
    upper=min(dim-1,math.ceil(dim*math.log(context/(2*math.pi))/(2*math.log(theta))))
    return [theta**(-2*j/dim)*(1-min(1.,max(0.,(j-lower)/(upper-lower)))*(1-1/factor))
            for j in range(dim//2)]


@unittest.skipUnless(SOURCE, SOURCE_REQUIRED)
class YaRNRoPEContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        # Execute actual source constructors and frequency/cache formulas. Only
        # the unused custom-op dispatch is inert; no installed vLLM can leak in.
        class CpuCustomOp(torch.nn.Module):
            def enabled(self):
                return False
        scope = {'torch': torch, 'math': math, '_ROPE_DICT': {}, 'CustomOp': CpuCustomOp,
                 'ApplyRotaryEmb': lambda **kwargs: SimpleNamespace(**kwargs)}
        nodes = []
        relative = 'vllm/model_executor/layers/rotary_embedding/'
        for filename, names in (
                ('common.py', {'yarn_find_correction_dim', 'yarn_find_correction_range',
                               'yarn_linear_ramp_mask', 'yarn_get_mscale'}),
                ('base.py', {'RotaryEmbeddingBase', 'RotaryEmbedding'}),
                ('yarn_scaling_rope.py', {'YaRNScalingRotaryEmbedding'}),
                ('mrope.py', {'MRotaryEmbedding'}), ('__init__.py', {'get_rope'})):
            tree = ast.parse(source_path(relative + filename).read_text(encoding='utf-8'))
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                    if isinstance(node, ast.ClassDef):
                        node.decorator_list = []
                        node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                                     and method.name in ('__init__', '_compute_inv_freq', '_compute_cos_sin_cache')]
                    nodes.append(node)
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                     'prepared-source-CPU-rope', 'exec'), scope)
        cls.torch = torch
        cls.R = SimpleNamespace(_ROPE_DICT=scope['_ROPE_DICT'], get_rope=scope['get_rope'])
        cls.M = scope['MRotaryEmbedding']
        cls.Y = SimpleNamespace(torch=torch, YaRNScalingRotaryEmbedding=scope['YaRNScalingRotaryEmbedding'])
        @contextmanager
        def config_context(config):
            module = ModuleType('vllm.config')
            module.get_current_vllm_config = lambda: config
            with patch.dict(sys.modules, {'vllm': ModuleType('vllm'), 'vllm.config': module}):
                yield
        cls.VllmConfig = staticmethod(lambda **kwargs: SimpleNamespace(speculative_config=None, **kwargs))
        cls.set_config = staticmethod(config_context)
        cls.CompilationConfig, cls.NONE = SimpleNamespace, 0
        if torch.cuda.is_initialized(): raise AssertionError('CPU contract process must not initialize CUDA')

    def setUp(self):
        self.keys=set(self.R._ROPE_DICT)
        self.parameters={'rope_type':'yarn','factor':1.5625,
            'original_max_position_embeddings':262144,'rope_theta':10_000_000.,
            'partial_rotary_factor':.25,'mrope_section':[11,11,10],'mrope_interleaved':True}

    def tearDown(self):
        for key in set(self.R._ROPE_DICT)-self.keys: self.R._ROPE_DICT.pop(key)
        self.assertFalse(self.torch.cuda.is_initialized())

    def make(self,parameters=None,*,served=409600,positions_factor=1):
        params=self.parameters if parameters is None else parameters
        before=copy.deepcopy(params)
        cfg=self.VllmConfig(compilation_config=self.CompilationConfig(mode=self.NONE,custom_ops=['none'],backend='eager'))
        cfg.model_config=SimpleNamespace(max_model_len=served)
        def tiny_cache(obj):
            return self.torch.empty((1,obj.rotary_dim),dtype=self.torch.float32,device='cpu')
        with self.set_config(cfg),patch.object(self.M,'_compute_cos_sin_cache',tiny_cache),self.torch.device('cpu'):
            result=self.R.get_rope(256,served,rope_parameters=params,dtype=self.torch.float32,
                                   mrope_positions_factor=positions_factor)
        self.assertEqual(params,before)
        return result

    def check_frequency(self,obj,expected):
        scaling=obj.scaling_factor if obj.scaling_factor is not None else obj.base
        actual=obj._compute_inv_freq(scaling).tolist()
        self.assertEqual(len(actual),len(expected))
        for j,(observed,truth) in enumerate(zip(actual,expected)):
            self.assertAlmostEqual(observed,truth,delta=max(1e-12,abs(truth)*2e-6),msg=f'inverse frequency {j}')

    def test_served_window_does_not_change_original_context_frequencies(self):
        obj=self.make()
        self.check_frequency(obj,reference(262144,1.5625))
        self.assertEqual(obj.original_max_position_embeddings,262144)
        self.assertEqual(obj.max_position_embeddings,409600)
        self.assertEqual(obj.cache_max_position_num,409600)
        # Preserve the previous cache extent; fix only the frequency context.
        self.assertEqual(obj.max_position_embeddings*obj.scaling_factor,640000)

    def test_pruned_position_extent_does_not_change_frequencies(self):
        obj=self.make(positions_factor=4)
        self.check_frequency(obj,reference(262144,1.5625))
        self.assertEqual(obj.original_max_position_embeddings,262144)
        self.assertEqual(obj.cache_max_position_num,1048576)
        self.assertEqual(obj.max_position_embeddings*obj.scaling_factor,1638400)

    def test_restored_original_rope_uses_default_frequencies(self):
        params={k:v for k,v in self.parameters.items() if k not in ('factor','original_max_position_embeddings')}
        params['rope_type']='default'
        obj=self.make(params,served=262144)
        self.assertIsNone(obj.scaling_factor)
        self.check_frequency(obj,[10_000_000.**(-2*j/64) for j in range(32)])
        self.assertEqual(obj.mscale,1.)

    def test_default_rope_extended_cache_does_not_gain_yarn_scaling(self):
        params={k:v for k,v in self.parameters.items() if k not in ('factor','original_max_position_embeddings')}
        params['rope_type']='default'
        obj=self.make(params)
        self.check_frequency(obj,[10_000_000.**(-2*j/64) for j in range(32)])
        self.assertEqual(obj.mscale,1.)

    def test_plain_yarn_without_multimodal_sections_keeps_existing_formula(self):
        params={k:v for k,v in self.parameters.items() if k not in ('mrope_section','mrope_interleaved')}
        cfg=self.VllmConfig(compilation_config=self.CompilationConfig(mode=self.NONE,custom_ops=['none'],backend='eager'))
        cfg.model_config=SimpleNamespace(max_model_len=409600)
        with self.set_config(cfg),patch.object(self.Y.YaRNScalingRotaryEmbedding,'_compute_cos_sin_cache',
            lambda obj:self.torch.empty((1,obj.rotary_dim),dtype=self.torch.float32)),self.torch.device('cpu'):
            obj=self.R.get_rope(256,409600,rope_parameters=params,dtype=self.torch.float32)
        self.check_frequency(obj,reference(262144,1.5625))

    def test_actual_yarn_cache_formula_at_small_and_context_edge_positions(self):
        torch=self.torch
        obj=self.make()
        original_arange=torch.arange
        positions=torch.tensor([0,1,32,4095,262143,262144,409599,409600],dtype=torch.float32)
        def selected_arange(*args,**kwargs):
            if len(args)==1 and args[0]==640000: return positions.clone()
            return original_arange(*args,**kwargs)
        with patch.object(self.Y.torch,'arange',side_effect=selected_arange),torch.device('cpu'):
            cache=self.Y.YaRNScalingRotaryEmbedding._compute_cos_sin_cache(obj)
        self.assertEqual(tuple(cache.shape),(len(positions),64))
        mscale=1+.1*math.log(1.5625)
        self.assertAlmostEqual(obj.mscale,mscale,places=12)
        self.assertTrue(torch.equal(cache[0,:32],torch.full((32,),mscale)))
        self.assertTrue(torch.equal(cache[0,32:],torch.zeros(32)))
        phase=positions.double()[:,None]*torch.tensor(reference(262144,1.5625),dtype=torch.float64)
        self.check_frequency(obj,reference(262144,1.5625))
        rounded_phase=positions[:,None]*obj._compute_inv_freq(obj.scaling_factor)
        # Sin/cos Lipschitz bound for independently checked frequency/phase rounding.
        bound=mscale*((rounded_phase.double()-phase).abs()+3e-6)
        cos,sin=cache[:,:32].double(),cache[:,32:].double()
        self.assertTrue(bool(((cos-mscale*phase.cos()).abs()<=bound).all()))
        self.assertTrue(bool(((sin-mscale*phase.sin()).abs()<=bound).all()))


if __name__=='__main__':
    unittest.main()
