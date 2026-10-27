"""One engine must prepare and validate the same unambiguous recipe."""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import pickle
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(ROOT / "scripts/load_source_package.py")["load"]()

from _rtx_spark import plugin, recipes, runtime  # noqa: E402
from _rtx_spark.optimizations import Option  # noqa: E402


@dataclass
class Arguments:
    model: str
    additional_config: dict
    hf_config: object = None


class RecipeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        for context in (patch.dict(recipes._MODELS, {}, clear=True),
                        patch.object(recipes, "register_builtin_models"),
                        patch.object(plugin, "register")):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def register(self, name, *, matches=None, architectures=()):
        def prepare(arguments, options):
            self.calls.append((name, "prepare", options["model"]))

        def validate(config, options):
            self.calls.append((name, "validate", options["model"]))

        def apply(arguments, value):
            self.calls.append((name, "apply", value))

        recipes.register_model(name, prepare, validate, matches=matches,
                               architectures=architectures,
                               options=(Option("setting", int, 7, apply=apply),))

    @staticmethod
    def args(model="checkpoint", *, options=None, architectures=()):
        return SimpleNamespace(model=model,
                               additional_config={} if options is None else {"rtx_spark": options},
                               hf_config=SimpleNamespace(architectures=list(architectures)))

    def test_explicit_recipe_wins_without_running_other_matchers(self):
        def forbidden(arguments):
            raise AssertionError("Explicit selection must not use automatic matchers")

        self.register("first", matches=forbidden, architectures=("Shared",))
        self.register("second", matches=forbidden, architectures=("Shared",))
        args = self.args(options={"model": "second"}, architectures=("Shared",))
        self.assertIs(recipes.resolve_model(args), recipes.resolve_model(args, name="second"))
        runtime.prepare_engine_args(args)
        runtime.configure_engine(args, engine_args=args)
        self.assertEqual(self.calls, [("second", "prepare", "second"), ("second", "apply", 7),
                                     ("second", "validate", "second")])

    def test_automatic_multiple_matches_fail_before_any_recipe_mutation(self):
        self.register("first", matches=lambda args: True)
        self.register("second", architectures=("Shared",))
        args = self.args(architectures=("Shared",))
        with self.assertRaisesRegex(ValueError, "Ambiguous.*first, second"):
            runtime.prepare_engine_args(args)
        self.assertEqual(self.calls, [])
        self.assertEqual(args.additional_config, {})

    def test_same_recipe_matching_name_and_architecture_is_one_candidate(self):
        self.register("one", matches=lambda args: True, architectures=("Shared",))
        self.assertEqual(recipes.resolve_model_name(self.args(architectures=("Shared",))), "one")

    def test_prepared_recipe_survives_different_final_architecture_and_pickle(self):
        self.register("first", matches=lambda args: args.model == "first-checkpoint")
        self.register("second", architectures=("Other",))
        args = self.args("first-checkpoint")
        runtime.prepare_engine_args(args)
        # The pinned hook passes EngineArgs even if config construction changes
        # which architecture would be chosen by automatic matching.
        config = self.args("final-checkpoint", architectures=("Other",))
        runtime.configure_engine(config, engine_args=pickle.loads(pickle.dumps(args)))
        self.assertEqual(config.additional_config["rtx_spark"]["model"], "first")
        copied = pickle.loads(pickle.dumps(config))
        runtime.configure_engine(copied)
        self.assertEqual(self.calls, [("first", "prepare", "first"), ("first", "apply", 7),
                                     ("first", "validate", "first"), ("first", "validate", "first")])

    def test_final_config_uses_only_declared_fields_and_supports_strict_clone(self):
        self.register("first", matches=lambda args: args.model == "first-checkpoint")
        args = self.args("first-checkpoint")
        runtime.prepare_engine_args(args)
        config = Arguments("final-checkpoint", {})
        runtime.configure_engine(config, engine_args=args)
        self.assertLessEqual(vars(config).keys(), {field.name for field in fields(config)})
        # vLLM's MTP draft config replacement rejects undeclared __dict__ keys.
        # A constructor round-trip has the same strict declared-field contract.
        draft = type(config)(**dict(vars(config), model="draft-checkpoint"))
        self.assertEqual(draft.model, "draft-checkpoint")
        self.assertEqual(draft.additional_config["rtx_spark"]["model"], "first")
        self.assertEqual(self.calls[-1], ("first", "validate", "first"))

    def test_no_match_is_preserved_when_architecture_only_appears_later(self):
        self.register("late", architectures=("LateArchitecture",))
        args = self.args()
        runtime.prepare_engine_args(args)
        config = self.args(architectures=("LateArchitecture",))
        runtime.configure_engine(config, engine_args=pickle.loads(pickle.dumps(args)))
        self.assertEqual(self.calls, [])
        self.assertEqual(config.additional_config, {})
        self.assertNotIn("_rtx_spark_prepared_recipe", vars(config))

    def test_auto_recipe_name_survives_dataclass_field_serialization(self):
        self.register("first", matches=lambda args: args.model == "first-checkpoint")
        self.register("second", architectures=("Other",))
        args = Arguments("first-checkpoint", {})
        runtime.prepare_engine_args(args)
        serialized = asdict(args)
        self.assertEqual(serialized["additional_config"]["rtx_spark"]["model"], "first")
        config = SimpleNamespace(**serialized)
        config.model = "different-checkpoint"
        config.hf_config = SimpleNamespace(architectures=["Other"])
        runtime.configure_engine(config)
        self.assertEqual(self.calls[-1], ("first", "validate", "first"))

    def test_changing_explicit_recipe_between_stages_is_rejected(self):
        self.register("first")
        self.register("second")
        args = self.args(options={"model": "first"})
        runtime.prepare_engine_args(args)
        with self.assertRaisesRegex(ValueError, "changed after preparation"):
            runtime.configure_engine(self.args(options={"model": "second"}), engine_args=args)
        self.assertFalse(any(call[1] == "validate" for call in self.calls))

    def test_no_match_cannot_become_an_explicit_recipe_without_preparation(self):
        self.register("late")
        args = self.args()
        runtime.prepare_engine_args(args)
        with self.assertRaisesRegex(ValueError, "changed after preparation"):
            runtime.configure_engine(self.args(options={"model": "late"}), engine_args=args)
        self.assertEqual(self.calls, [])

    def test_two_engines_keep_independent_selection(self):
        self.register("first", matches=lambda args: args.model == "first")
        self.register("second", matches=lambda args: args.model == "second")
        first, second = self.args("first"), self.args("second")
        runtime.prepare_engine_args(first)
        runtime.prepare_engine_args(second)
        runtime.configure_engine(self.args(), engine_args=first)
        runtime.configure_engine(self.args(), engine_args=second)
        self.assertEqual([call[0] for call in self.calls if call[1] == "validate"], ["first", "second"])

    def test_binding_does_not_mutate_caller_owned_config_maps(self):
        self.register("selected", matches=lambda args: True)
        original = {"another_plugin": {"value": [1]}, "rtx_spark": {"optimizations": {"setting": 9}}}
        expected = deepcopy(original)
        args = self.args()
        args.additional_config = original
        runtime.prepare_engine_args(args)
        self.assertEqual(original, expected)
        self.assertIsNot(args.additional_config, original)
        self.assertIsNot(args.additional_config["rtx_spark"], original["rtx_spark"])
        self.assertEqual(args.additional_config["rtx_spark"]["model"], "selected")
        self.assertEqual(args.additional_config["another_plugin"], expected["another_plugin"])

    def test_hook_requires_preparation_when_engine_args_are_supplied(self):
        with self.assertRaisesRegex(RuntimeError, "were not prepared"):
            runtime.configure_engine(self.args(), engine_args=self.args())


if __name__ == "__main__":
    unittest.main()
