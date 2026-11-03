"""Source-extracted resolved Mamba checkpoint-unit regressions, stdlib only.

Run against a freshly prepared source with --site. No Torch/model/GPU import.
"""
import argparse
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--site", type=Path)
ARGS, REST = parser.parse_known_args()
SITE = ARGS.site.resolve() if ARGS.site else None
REL = "vllm/v1/core/sched/scheduler.py"


def node(path, name, owner=None):
    value = ast.parse(path.read_text(encoding="utf-8"))
    if owner:
        value = next(n for n in value.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return copy.deepcopy(next(n for n in value.body if isinstance(n, ast.FunctionDef) and n.name == name))


class MambaSpec:
    pass


def compile_nodes(nodes):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    scope = {"cdiv": lambda a, b: -(a // -b), "MambaSpec": MambaSpec}
    exec(compile(ast.fix_missing_locations(module), "source-extracted-resolved-Mamba-unit", "exec"), scope)
    return scope


def scheduler(unit=6912, configured=3456, hash_unit=64, drop=False):
    return NS(mamba_state_block_size=unit,
        cache_config=NS(mamba_block_size=configured, block_size=1728),
        block_size=6912, hash_block_size=hash_unit, mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=True, mamba_prefill_checkpoint_alignment=64,
        mamba_shared_prefix_checkpoint=True, use_eagle_block_drop=drop,
        max_num_scheduled_tokens=8192, scheduler_config=NS(long_prefill_token_threshold=0))


def chunks(fn, value, prompt, start=0):
    ends = []
    for _ in range(20):
        if start == prompt: return ends
        request = NS(num_computed_tokens=start, num_prompt_tokens=prompt,
                     num_tokens=prompt, shared_prefix_boundary=0)
        count = fn(value, request, min(8192, prompt - start))
        if count <= 0: raise AssertionError("Source chunk splitter deferred fixture")
        start += count; ends.append(start)
    raise AssertionError("Source splitter did not terminate")


@unittest.skipUnless(SITE, "Pass --site for actual source-extracted regression")
class ResolvedStateUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = [node(SITE / "vllm/v1/kv_cache_interface.py", n) for n in
            ("get_mamba_prefill_checkpoint_position", "is_mamba_prefill_checkpoint_valid")]
        split = node(SITE / REL, "_mamba_block_aligned_split", "Scheduler")
        scope = compile_nodes(helpers + [split])
        cls.split = staticmethod(scope[split.name])
        # Counterfactual reintroduces only the stale owner, preserving the exact
        # scheduling body and all boundary decisions for before/after witness.
        before = copy.deepcopy(split)
        for n in ast.walk(before):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "block_size" for t in n.targets):
                n.value = ast.parse("self.cache_config.mamba_block_size", mode="eval").body
        cls.before = staticmethod(compile_nodes(helpers + [before])[before.name])
        init = node(SITE / REL, "__init__", "Scheduler")
        selected = []
        for n in init.body:
            text = ast.unparse(n)
            if ((isinstance(n, ast.Assign) and any(
                    (isinstance(t, ast.Name) and t.id == "mamba_state_block_sizes") or
                    (isinstance(t, ast.Attribute) and t.attr == "mamba_state_block_size") for t in n.targets))
                or (isinstance(n, ast.If) and "len(mamba_state_block_sizes)" in text)):
                selected.append(n)
        if len(selected) != 3: raise AssertionError("Expected resolved-unit derivation and fail-closed check")
        derive = ast.FunctionDef(name="derive", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=selected, decorator_list=[])
        cls.derive = staticmethod(compile_nodes([derive])["derive"])

    def test_actual_6690_prompt_does_not_skip_6656_state_stop(self):
        self.assertEqual(chunks(self.before, scheduler(), 6690), [6690])
        self.assertEqual(chunks(self.split, scheduler(), 6690), [6656, 6690])

    def test_12800_prefix_preserves_12736_checkpoint(self):
        self.assertEqual(chunks(self.before, scheduler(), 12800), [6912, 12800])
        self.assertEqual(chunks(self.split, scheduler(), 12800), [6912, 12736, 12800])

    def test_mid_physical_block_must_not_claim_internal_export(self):
        self.assertEqual(chunks(self.before, scheduler(), 7063, 3456), [7063])
        self.assertEqual(chunks(self.split, scheduler(), 7063, 3456), [6912, 7040, 7063])

    def test_matching_state_units_keep_existing_hash_and_Eagle_behavior(self):
        for unit in (3456, 6912):
            for h in (32, 64):
                for drop in (False, True):
                    for prompt in (6690, 8721, 12800, 17442):
                        with self.subTest(unit=unit, hash=h, drop=drop, prompt=prompt):
                            state = scheduler(unit, unit, h, drop)
                            self.assertEqual(chunks(self.split, state, prompt), chunks(self.before, state, prompt))

    def state(self, sizes, align=True):
        managers = [NS(kv_cache_spec=MambaSpec(), block_size=n) for n in sizes]
        managers.append(NS(kv_cache_spec=object(), block_size=1728))
        return NS(has_mamba_layers=bool(sizes), cache_config=NS(mamba_cache_mode="align" if align else "none", mamba_block_size=3456),
                  kv_cache_manager=NS(coordinator=NS(single_type_managers=managers)))

    def test_resolved_spec_wins_without_mutating_config(self):
        state = self.state([6912] * 4); config = state.cache_config
        self.derive(state)
        self.assertEqual(state.mamba_state_block_size, 6912)
        self.assertIs(state.cache_config, config)
        self.assertEqual(config.mamba_block_size, 3456)

    def test_no_mamba_keeps_inactive_unit_none(self):
        state = self.state([]); self.derive(state)
        self.assertIsNone(state.mamba_state_block_size)
        self.assertEqual(state.cache_config.mamba_block_size, 3456)

    def test_heterogeneous_align_state_units_fail_closed(self):
        with self.assertRaisesRegex(AssertionError, "one resolved Mamba state unit"):
            self.derive(self.state([3456, 6912]))

    def test_nonalign_mode_is_not_given_a_false_common_unit(self):
        state = self.state([3456, 6912], align=False); self.derive(state)
        self.assertIsNone(state.mamba_state_block_size)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + REST, verbosity=2)
