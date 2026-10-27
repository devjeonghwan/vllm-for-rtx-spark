"""Maintainer hook extraction must accept fresh and already-hooked sources."""

import ast
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORT = runpy.run_path(ROOT / "scripts/export_vllm_changes.py")
HOOKS = EXPORT["rtx_spark_hooks"]
INIT = "vllm/__init__.py"
ARGS = "vllm/engine/arg_utils.py"
FRESH_INIT = b"from .version import __version__, __version_tuple__  # isort:skip\n\nimport typing\n"
ADDITION = (
    b"\n# Configure the native platform before importing vLLM consumers.\n"
    b"from vllm._rtx_spark.runtime.bootstrap import initialize as _rtx_spark_initialize\n"
    b"\n_rtx_spark_initialize()\n"
)
FRESH_ARGS = b'''class EngineArgs:
    def create_engine_config(self):
        """Build configuration."""
        config = make_config(self)
        return config
'''


def direct_calls(raw, name):
    return [node for node in ast.walk(ast.parse(raw)) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == name]


class ExtractHookTests(unittest.TestCase):
    def test_private_modules_and_vendor_copies_are_not_exported_as_dependency_patches(self):
        for name in ("vllm/_rtx_spark/runtime/vllm/adapter.py",
                     "vllm/_rtx_spark/recipes/qwen3_8/helpers/mbx_loader.py",
                     "vllm/third_party/deep_gemm/__init__.py",
                     "vllm/third_party/flashmla/flash_mla_interface.py"):
            with self.subTest(name=name):
                self.assertFalse(EXPORT["is_patch_source"](name))
        for name in ("vllm/engine/arg_utils.py", "csrc/quantization/example.cu",
                     "vllm/third_party/flashmla/__init__.py"):
            with self.subTest(name=name):
                self.assertTrue(EXPORT["is_patch_source"](name))

    def test_fresh_bootstrap_matches_the_pinned_insertion(self):
        needle = FRESH_INIT.splitlines(keepends=True)[0]
        expected = FRESH_INIT.replace(needle, needle + ADDITION)
        result = HOOKS(INIT, FRESH_INIT)
        self.assertEqual(result, expected)
        self.assertEqual(len(direct_calls(result, "_rtx_spark_initialize")), 1)

    def test_new_hooks_are_byte_idempotent_for_both_files(self):
        for name, original in ((INIT, FRESH_INIT), (ARGS, FRESH_ARGS)):
            with self.subTest(name=name):
                hooked = HOOKS(name, original)
                self.assertEqual(HOOKS(name, hooked), hooked)
                self.assertEqual(HOOKS(name, HOOKS(name, hooked)), hooked)

    def test_previous_final_config_hook_is_upgraded_to_preserve_engine_selection(self):
        current = HOOKS(ARGS, FRESH_ARGS)
        previous = current.replace(b"configure_engine(config, engine_args=self)", b"configure_engine(config)")
        self.assertNotEqual(current, previous)
        self.assertEqual(HOOKS(ARGS, previous), current)
        self.assertEqual(HOOKS(ARGS, current), current)

    def test_original_helper_imports_become_qualified_without_changing_literals(self):
        original = (
            b'# import mbx_depth remains a comment.\n'
            b'literal = "from loader_core import anything"\n'
            b'import mbx_depth as depth\n'
            b'from vllm_fp8_hybrid import apply as hybrid\n'
        )
        expected = (
            b'# import mbx_depth remains a comment.\n'
            b'literal = "from loader_core import anything"\n'
            b'from vllm._rtx_spark.recipes.qwen3_8.helpers import mbx_depth as depth\n'
            b'from vllm._rtx_spark.recipes.qwen3_8.helpers.vllm_fp8_hybrid import apply as hybrid\n'
        )
        result = HOOKS("vllm/example.py", original)
        self.assertEqual(result, expected)
        self.assertEqual(HOOKS("vllm/example.py", result), result)

    def test_old_namespace_and_alias_normalize_once(self):
        for name, original in ((INIT, FRESH_INIT), (ARGS, FRESH_ARGS)):
            with self.subTest(name=name):
                new = HOOKS(name, original)
                old = new.replace(b"vllm._rtx_spark", b"vllm._spark").replace(
                    b"_rtx_spark_initialize", b"_spark_initialize")
                self.assertEqual(HOOKS(name, old), new)

    def test_identifier_normalization_preserves_strings_comments_and_longer_names(self):
        original = (
            b'# _spark_initialize should remain literal here.\n'
            b'literal = "vllm._spark and _spark_initialize"\n'
            b'_spark_initialize_extra = 1\n'
            b'from vllm._spark.runtime.bootstrap import initialize as _spark_initialize\n'
            b'_spark_initialize()\n'
        )
        expected = original.replace(b'from vllm._spark.', b'from vllm._rtx_spark.').replace(
            b'as _spark_initialize\n', b'as _rtx_spark_initialize\n').replace(
            b'\n_spark_initialize()\n', b'\n_rtx_spark_initialize()\n')
        result = HOOKS(INIT, original)
        self.assertEqual(result, expected)
        self.assertEqual(HOOKS(INIT, result), result)

    def test_ast_recognizes_multiline_and_aliased_existing_hooks(self):
        original = (
            b'from vllm._rtx_spark.runtime.bootstrap import (\n'
            b'    initialize as early_initialize,\n'
            b')\n'
            b'early_initialize()\n'
        )
        self.assertEqual(HOOKS(INIT, original), original)

    def test_hook_mentions_in_strings_do_not_suppress_insertion(self):
        original = FRESH_ARGS.replace(b'        config =', (
            b'        note = "from vllm._rtx_spark.runtime import prepare_engine_args"\n'
            b'        other = "from vllm._rtx_spark.runtime import configure_engine"\n'
            b'        config ='
        ))
        hooked = HOOKS(ARGS, original)
        for name in ("prepare_engine_args", "configure_engine"):
            self.assertEqual(len(direct_calls(hooked, name)), 1)
        self.assertEqual(HOOKS(ARGS, hooked), hooked)

    def test_one_existing_engine_hook_does_not_get_duplicated(self):
        original = FRESH_ARGS.replace(b'        config =', (
            b'        from vllm._rtx_spark.runtime import prepare_engine_args\n'
            b'        prepare_engine_args(self)\n'
            b'        config ='
        ))
        hooked = HOOKS(ARGS, original)
        for name in ("prepare_engine_args", "configure_engine"):
            self.assertEqual(len(direct_calls(hooked, name)), 1)
        self.assertEqual(HOOKS(ARGS, hooked), hooked)

    def test_partial_duplicate_and_wrong_argument_hooks_are_rejected(self):
        cases = (
            (INIT, FRESH_INIT + b'from vllm._rtx_spark.runtime.bootstrap import initialize as _rtx_spark_initialize\n'),
            (INIT, HOOKS(INIT, FRESH_INIT) + b'_rtx_spark_initialize()\n'),
            (ARGS, HOOKS(ARGS, FRESH_ARGS).replace(b'prepare_engine_args(self)', b'prepare_engine_args(config)')),
            (ARGS, HOOKS(ARGS, FRESH_ARGS).replace(b'engine_args=self', b'engine_args=config')),
        )
        for name, original in cases:
            with self.subTest(name=name, length=len(original)), self.assertRaises(RuntimeError):
                HOOKS(name, original)


if __name__ == "__main__":
    unittest.main()
