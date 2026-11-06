"""Actual FlexibleArgumentParser nested-command YAML bool regressions; CPU only."""
import argparse
from contextlib import redirect_stderr
import io
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import yaml

from support import ROOT, SOURCE as SITE


def load_parser(site):
    # Execute the full pinned utility module with only its logging import
    # isolated. This neither imports vLLM/Torch nor constructs a model/config.
    path=site/"vllm/utils/argparse_utils.py"
    logger=ModuleType("vllm.logger")
    logger.init_logger=logging.getLogger
    scope={"__name__":"yaml_configuration_source_fixture"}
    with patch.dict(sys.modules,{"vllm":ModuleType("vllm"),"vllm.logger":logger}):
        exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),scope)
    return scope["FlexibleArgumentParser"]


@unittest.skipUnless(SITE,"Set VLLM_RTX_SPARK_SOURCE_DIR with a freshly composed source tree")
class YAMLFalseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Parser=load_parser(SITE)

    def setUp(self):
        parent=ROOT/".build/yaml-false-tests"
        parent.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.directory=Path(temporary.name)

    def file(self,values):
        path=self.directory/(str(len(list(self.directory.iterdir())))+".yaml")
        path.write_text(yaml.safe_dump(values),encoding="utf-8")
        return str(path)

    def nested(self,parser_class=None):
        parser=(parser_class or self.Parser)(prog="vllm")
        commands=parser.add_subparsers(dest="command",required=True)
        serve=commands.add_parser("serve",aliases=["s"])
        serve.add_argument("model")
        serve.add_argument("--enable-prefix-caching",action=argparse.BooleanOptionalAction,default=True)
        serve.add_argument("--enable-simple",action="store_true")
        serve.add_argument("--compilation-config")
        other=commands.add_parser("other")
        other.add_argument("model")
        other.add_argument("--sibling-only",action=argparse.BooleanOptionalAction,default=True)
        return parser

    def test_selected_serve_false_overrides_true_default(self):
        parser=self.nested()
        file=self.file({"enable-prefix-caching":False})
        self.assertFalse(parser.parse_args(["serve","checkpoint","--config",file]).enable_prefix_caching)

    def test_selected_serve_true_is_preserved(self):
        parser=self.nested()
        file=self.file({"enable-prefix-caching":True})
        self.assertTrue(parser.parse_args(["serve","checkpoint","--config",file]).enable_prefix_caching)

    def test_explicit_cli_true_wins_before_and_after_config_argument(self):
        file=self.file({"enable-prefix-caching":False})
        for tail in (["--enable-prefix-caching","--config",file],
                     ["--config",file,"--enable-prefix-caching"]):
            self.assertTrue(self.nested().parse_args(["serve","checkpoint",*tail]).enable_prefix_caching)

    def test_explicit_cli_false_wins_before_and_after_config_argument(self):
        file=self.file({"enable-prefix-caching":True})
        for tail in (["--no-enable-prefix-caching","--config",file],
                     ["--config",file,"--no-enable-prefix-caching"]):
            self.assertFalse(self.nested().parse_args(["serve","checkpoint",*tail]).enable_prefix_caching)

    def test_direct_parser_load_and_parse_keep_their_own_boolean_actions(self):
        parser=self.Parser()
        parser.add_argument("--enable-prefix-caching",action=argparse.BooleanOptionalAction,default=True)
        file=self.file({"enable-prefix-caching":False})
        self.assertEqual(parser.load_config_file(file),["--no-enable-prefix-caching"])
        self.assertFalse(parser.parse_args(["--config",file]).enable_prefix_caching)

    def test_snapshot_create_resolves_the_selected_grandchild(self):
        parser=self.Parser()
        first=parser.add_subparsers(dest="command",required=True)
        snapshot=first.add_parser("snapshot")
        second=snapshot.add_subparsers(dest="operation",required=True)
        create=second.add_parser("create")
        create.add_argument("model")
        create.add_argument("--enable-prefix-caching",action=argparse.BooleanOptionalAction,default=True)
        file=self.file({"enable-prefix-caching":False})
        self.assertFalse(parser.parse_args(["snapshot","create","checkpoint","--config",file]).enable_prefix_caching)

    def test_unknown_false_setting_reaches_argparse_error(self):
        file=self.file({"enable-prefix-cachng":False})
        errors=io.StringIO()
        with redirect_stderr(errors),self.assertRaises(SystemExit) as raised:
            self.nested().parse_args(["serve","checkpoint","--config",file])
        self.assertEqual(raised.exception.code,2)
        self.assertIn("--no-enable-prefix-cachng",errors.getvalue())

    def test_unselected_sibling_boolean_cannot_make_an_unknown_setting_disappear(self):
        file=self.file({"sibling-only":False})
        errors=io.StringIO()
        with redirect_stderr(errors),self.assertRaises(SystemExit) as raised:
            self.nested().parse_args(["serve","checkpoint","--config",file])
        self.assertEqual(raised.exception.code,2)
        self.assertIn("--no-sibling-only",errors.getvalue())

    def test_selected_sibling_uses_its_own_negative_action(self):
        file=self.file({"sibling-only":False})
        self.assertFalse(self.nested().parse_args(["other","checkpoint","--config",file]).sibling_only)

    def test_known_store_true_false_keeps_existing_omission_semantics(self):
        file=self.file({"enable-simple":False})
        value=self.nested().parse_args(["serve","checkpoint","--config",file])
        self.assertFalse(value.enable_simple)

    def test_underscore_yaml_names_match_normalized_cli_option_names(self):
        file=self.file({"enable_prefix_caching":False})
        self.assertFalse(self.nested().parse_args(["serve","checkpoint","--config",file]).enable_prefix_caching)

    def test_nested_json_false_payload_is_not_treated_as_a_cli_boolean(self):
        file=self.file({"compilation-config":{"pass_config":{"flag":False}}})
        value=self.nested().parse_args(["serve","checkpoint","--config",file])
        self.assertEqual(__import__("json").loads(value.compilation_config),{"pass_config":{"flag":False}})

    def test_command_alias_resolves_the_same_selected_parser(self):
        file=self.file({"enable-prefix-caching":False})
        self.assertFalse(self.nested().parse_args(["s","checkpoint","--config",file]).enable_prefix_caching)



if __name__=="__main__":
    unittest.main()
