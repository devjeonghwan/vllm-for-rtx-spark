"""Recipe-declared optimizations resolve before their callbacks mutate state."""

from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()

from _rtx_spark.optimizations import (  # noqa: E402
    Option,
    apply_options,
    resolve_options,
    validate_options,
)


class OptimizationTests(unittest.TestCase):
    def test_a_new_declaration_supplies_default_type_validation_and_application(self):
        calls = []

        def validate(arguments, value):
            calls.append(("validate", value))
            if value > arguments.limit:
                raise ValueError("fixture limit")

        def apply(arguments, value):
            calls.append(("apply", value))
            arguments.selected = value

        declarations = (Option("fixture_window", int, default=4, validate=validate, apply=apply),)
        arguments = SimpleNamespace(limit=8)
        resolved = resolve_options({}, declarations)
        validate_options(arguments, resolved, declarations)
        apply_options(arguments, resolved, declarations)
        self.assertEqual(arguments.selected, 4)
        self.assertEqual(calls, [("validate", 4), ("apply", 4)])

        calls.clear()
        supplied = resolve_options({"optimizations": {"fixture_window": 6}}, declarations)
        validate_options(arguments, supplied, declarations)
        apply_options(arguments, supplied, declarations)
        self.assertEqual(arguments.selected, 6)
        self.assertEqual(calls, [("validate", 6), ("apply", 6)])

        for invalid in (True, "6", 6.0):
            with self.subTest(value=invalid), self.assertRaisesRegex(TypeError, "fixture_window must be int"):
                resolve_options({"fixture_window": invalid}, declarations)
        with self.assertRaisesRegex(ValueError, "fixture limit"):
            validate_options(arguments, {"fixture_window": 9}, declarations)
        self.assertEqual(arguments.selected, 6)

    def test_flat_and_grouped_options_share_one_definition_and_detect_conflicts(self):
        declarations = (Option("fixture_fast", bool, default=True),)
        flat = resolve_options({"fixture_fast": False}, declarations)
        grouped = resolve_options({"optimizations": {"fixture_fast": False}}, declarations)
        repeated = resolve_options({"fixture_fast": False, "optimizations": {"fixture_fast": False}}, declarations)
        self.assertEqual(flat, grouped)
        self.assertEqual(grouped, repeated)
        for conflicting in (False, 1):
            with self.subTest(value=conflicting), self.assertRaisesRegex(ValueError, "Conflicting RTX Spark setting"):
                resolve_options({"fixture_fast": True, "optimizations": {"fixture_fast": conflicting}}, declarations)

    def test_undeclared_options_are_rejected_and_omitted_values_stay_absent(self):
        declarations = (Option("fixture_external_policy", str),)
        self.assertEqual(resolve_options({}, declarations), {})
        with self.assertRaisesRegex(ValueError, "Unknown RTX Spark setting"):
            resolve_options({"fixture_external_polciy": "path.json"}, declarations)
        with self.assertRaisesRegex(ValueError, "belongs outside optimizations"):
            resolve_options({"optimizations": {"model": "fixture-model"}}, declarations)
        with self.assertRaisesRegex(TypeError, "optimizations must be an object"):
            resolve_options({"optimizations": []}, declarations)


if __name__ == "__main__":
    unittest.main()
