"""Structural source-extracted schedule test. No tensor/numerical/GPU claim."""
from __future__ import annotations
import argparse
import ast
import sys
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--site',type=Path)
parser.add_argument('--baseline',type=Path)
ARGS, REST = parser.parse_known_args()
REL = 'vllm/models/qwen4_exp/nvidia/model.py'
ORIGINAL = ARGS.baseline.resolve()/REL if ARGS.baseline else None
CANDIDATE = ARGS.site.resolve()/REL if ARGS.site else None



class Tensor:
    def __init__(self, value, shape=(16, 2560)):
        self.value = value
        self.shape = shape

    def op(self, name, *others, shape=None):
        digest = hashlib.sha256(repr((name, self.value, [x.value for x in others])).encode()).hexdigest()
        return Tensor(digest, shape or self.shape)

    def repeat(self, a, b):
        return self.op('repeat', shape=(self.shape[0], self.shape[1] * b))

    def unflatten(self, axis, shape):
        return self.op('unflatten', shape=(self.shape[0], *shape))

    def mean(self, dim):
        return self.op('mean', shape=(self.shape[0], self.shape[-1]))


class HC:
    def __init__(self, name, counts):
        self.name = name
        self.counts = counts

    def _combined(self, hidden, block, injection):
        return hidden.op('materialized_combine', block, injection)

    def combine(self, hidden, block, injection):
        self.counts['standalone_combine'] += 1
        return self._combined(hidden, block, injection)

    def mix(self, hidden):
        return hidden, hidden.op(self.name + ':mix', shape=(16, 2560)), hidden.op(self.name + ':inject', shape=(16, 4))

    def combine_and_mix(self, hidden, block, injection):
        self.counts['combine_norm'] += 1
        return self.mix(self._combined(hidden, block, injection))


def methods(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in ('Qwen4ExpDecoderLayer', 'Qwen4ExpModel'):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in ('forward', '_dflash2_append_completed_aux'):
                    node.name = cls.name + '_' + node.name
                    node.decorator_list = []
                    selected.append(node)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    env = {'islice': islice, 'get_pp_group': lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
           '_QSA_LAYER_TYPES': {'full_attention'}}
    exec(compile(module, str(path), 'exec'), env)
    return env


def run(path, *, aux=(6, 16, 26, 36, 46), deepstack=None, blocked=False, supported=True):
    f = methods(path)
    counts = {'standalone_combine': 0, 'combine_norm': 0}
    config = SimpleNamespace(hc_count=4, hidden_size=2560)
    class Layer:
        forward = f['Qwen4ExpDecoderLayer_forward']
        def __init__(self, index):
            self.config = config
            self.use_hc_sequence_parallel = self.use_sequence_parallel = False
            self.attn_hyper_connection = HC(str(index) + ':attn', counts)
            self.mlp_hyper_connection = HC(str(index) + ':mlp', counts)
            self.layer_type = 'linear_attention'
            self.linear_attn = lambda hidden_states: hidden_states.op(str(index) + ':attention')
            self.mlp = lambda x: x.op(str(index) + ':mlp')
            self.ple = (lambda hidden, *args: hidden.op('PLE')) if index == 1 or (blocked and index == 6) else None
        __call__ = forward
    class Model:
        forward = f['Qwen4ExpModel_forward']
        _dflash2_append_completed_aux = f['Qwen4ExpModel__dflash2_append_completed_aux']
        def _start_layer_ple_prefetch(self, *args):
            pass
    model = Model()
    model.config = config
    model._dflash2_aux_supported = supported
    model.aux_hidden_state_layers = aux
    model.start_layer = 0; model.end_layer = 48
    model.layers = [Layer(i) for i in range(48)]
    model.use_hc_sequence_parallel = model.use_sequence_parallel = False
    model._mtp_hidden_buffer = None
    model.hyper_connection_mixer = HC('final', counts)
    result = model.forward(Tensor('ids'), Tensor('positions', (16,)), inputs_embeds=Tensor('embed'),
                           query_start_loc=Tensor('qloc'), ngram_context=Tensor('ngram'), deepstack_input_embeds=deepstack)
    if isinstance(result, tuple):
        result = (result[0].value, [item.value for item in result[1]])
    else:
        result = result.value
    return result, counts


@unittest.skipUnless(ORIGINAL is not None and CANDIDATE is not None, "Pass --site and --baseline for source-extracted aux contracts")
class Contracts(unittest.TestCase):
    def test_exact_capture_order_and_target_route_structurally_equal(self):
        old, old_counts = run(ORIGINAL)
        new, new_counts = run(CANDIDATE)
        self.assertEqual(old, new)
        self.assertEqual(len(new[1]), 5)
        self.assertEqual(old_counts['standalone_combine'] - new_counts['standalone_combine'], 5)
        self.assertEqual(old_counts['combine_norm'], new_counts['combine_norm'])

    def test_no_aux_MTP_path_unchanged(self):
        self.assertEqual(run(ORIGINAL, aux=()), run(CANDIDATE, aux=()))

    def test_deepstack_nonNone_keeps_original_capture(self):
        self.assertEqual(run(ORIGINAL, deepstack={}), run(CANDIDATE, deepstack={}))

    def test_PLE_capture_boundary_keeps_original_capture(self):
        self.assertEqual(run(ORIGINAL, blocked=True), run(CANDIDATE, blocked=True))

    def test_unsupported_geometry_keeps_original_capture(self):
        self.assertEqual(run(ORIGINAL, supported=False), run(CANDIDATE, supported=False))

    def test_other_capture_tuple_keeps_original_capture(self):
        self.assertEqual(run(ORIGINAL, aux=(7, 17)), run(CANDIDATE, aux=(7, 17)))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]]+REST,verbosity=2)
