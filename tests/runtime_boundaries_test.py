"""Model isolation, optional backend import and Windows logging CPU contracts."""
import ast
from contextlib import ExitStack
import importlib
import io
import logging
import logging.config
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from support import SOURCE as SITE
ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(ROOT / "scripts/load_source_package.py")["load"]()
from _rtx_spark.runtime.depth import get_depth_helper  # noqa: E402
from _rtx_spark.runtime import logging as runtime_logging  # noqa: E402

DEPTH_PACKAGE = "_rtx_spark.recipes.qwen3_8_flash_next.helpers"
DEPTH_MODULE = DEPTH_PACKAGE + ".mbx_depth"


def config(recipe="qwen3.8-flash-next", architecture="Qwen4ExpForConditionalGeneration", method="mtp"):
    return NS(additional_config={"rtx_spark": {"model": recipe}},
              model_config=NS(hf_config=NS(architectures=[architecture])),
              speculative_config=None if method is None else NS(method=method))


def fake_depth():
    module = ModuleType(DEPTH_MODULE)
    module.D = NS(on=True)
    return module


def _spawn_acquire(queue, gemma, qwen):
    try:
        # Deliberately valid-looking inherited paths are not model selection.
        os.environ.update(MBX_DEPTH_LIB="existing-qwen.dll", MBX_DEPTH_CFG="qwen.json",
                          MBX_DRAFT_FP8="1", MBX_DRAFT_HEAD2="qwen-basis.pt", MBX_PLE_MMAP="1")
        package = importlib.import_module(DEPTH_PACKAGE)
        helper = fake_depth()
        with patch.dict(sys.modules, {DEPTH_MODULE: helper}), patch.object(package, "mbx_depth", helper, create=True):
            first = get_depth_helper(gemma)
            selected = get_depth_helper(qwen)
            second = get_depth_helper(gemma)
            dflash = get_depth_helper(config(method="dflash"))
        queue.put({"gemma_first": first is None, "qwen_same_helper": selected is helper,
                   "gemma_after_qwen": second is None, "dflash": dflash is None,
                   "torch_loaded": "torch" in sys.modules, "cutlass_loaded": "cutlass" in sys.modules,
                   "caller_env_retained": os.environ["MBX_DEPTH_LIB"] == "existing-qwen.dll"})
    except BaseException as error:
        queue.put({"error": repr(error)})


def _spawn_logging(queue, filename):
    try:
        runtime_logging.configure_standard_streams()
        config_map = {"version": 1, "disable_existing_loggers": False,
            "handlers": {"file": {"class": "logging.FileHandler", "filename": filename}},
            "loggers": {"spawn_boundary": {"handlers": ["file"], "level": "INFO", "propagate": False}}}
        logging.config.dictConfig(runtime_logging.with_file_handler_encoding(config_map))
        logger = logging.getLogger("spawn_boundary")
        logger.info("깊이 · →")
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers.clear()
        queue.put({"encoding": sys.stdout.encoding.lower().replace("-", ""),
                   "bytes": Path(filename).read_bytes().hex(),
                   "torch_loaded": "torch" in sys.modules})
    except BaseException as error:
        queue.put({"error": repr(error)})


def backend_support_function():
    if SITE:
        source = (SITE / "vllm/utils/b12x.py").read_text(encoding="utf-8")
    else:
        # The complete added function is tested even without an upstream tree.
        source = (ROOT / "patches/vllm/0013-runtime-model-boundaries.patch").read_text(encoding="utf-8")
        lines = source.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("+def check_b12x_backend_support("))
        chosen = []
        for line in lines[start:]:
            if not line.startswith("+") or line.startswith("+++"):
                break
            chosen.append(line[1:])
        source = "\n".join(chosen)
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                and node.name == "check_b12x_backend_support")
    scope = {"ModuleType": ModuleType}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "b12x-capability-source", "exec"), scope)
    return scope[node.name]


class DepthBoundaryTests(unittest.TestCase):
    def test_non_qwen_no_spec_dflash_and_mismatched_architecture_never_import_helper(self):
        original = __import__("builtins").__import__
        def guarded(name, *args, **kwargs):
            if "mbx_depth" in name or name.endswith("qwen3_8_flash_next.helpers"):
                raise AssertionError("Qwen helper entered an unrelated model path")
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {"MBX_DEPTH_LIB": __file__, "MBX_DEPTH_CFG": __file__,
                                    "MBX_DRAFT_FP8": "1", "MBX_PLE_MMAP": "1"}), \
                patch("builtins.__import__", guarded):
            for item in (config("gemma4-12b-fp8", "Gemma4UnifiedForConditionalGeneration", None),
                         config("gemma4-12b-fp8", "Gemma4UnifiedForConditionalGeneration", "mtp"),
                         config(method=None), config(method="dflash"),
                         config(architecture="Gemma4UnifiedForConditionalGeneration")):
                self.assertIsNone(get_depth_helper(item))
            self.assertEqual(os.environ["MBX_DEPTH_LIB"], __file__)

    def test_qwen_returns_original_helper_and_other_instances_do_not_borrow_its_state(self):
        package = importlib.import_module(DEPTH_PACKAGE)
        helper = fake_depth()
        with patch.dict(sys.modules, {DEPTH_MODULE: helper}), patch.object(package, "mbx_depth", helper, create=True):
            qwen_owner = NS(depth=get_depth_helper(config()))
            gemma_owner = NS(depth=get_depth_helper(config("gemma4-12b-fp8", "Gemma4UnifiedForConditionalGeneration", None)))
            self.assertIs(qwen_owner.depth, helper)
            self.assertTrue(qwen_owner.depth.D.on)
            self.assertIsNone(gemma_owner.depth)
            self.assertIsNone(get_depth_helper(config(method="dflash")))

    def test_prepared_recipe_is_required_even_if_qwen_architecture_or_env_is_present(self):
        item = config()
        item.additional_config = {}
        with patch.dict(os.environ, {"MBX_DEPTH_LIB": __file__}):
            self.assertIsNone(get_depth_helper(item))

    def test_missing_qwen_helper_import_is_not_silenced(self):
        original = __import__("builtins").__import__
        def guarded(name, *args, **kwargs):
            if name.endswith("qwen3_8_flash_next.helpers"):
                raise ImportError("selected Qwen helper is broken")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guarded), self.assertRaisesRegex(ImportError, "Qwen helper is broken"):
            get_depth_helper(config())

    def test_real_spawn_uses_serialized_model_config_despite_qwen_environment(self):
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(target=_spawn_acquire, args=(queue,
            config("gemma4-12b-fp8", "Gemma4UnifiedForConditionalGeneration", None), config()))
        process.start()
        try:
            value = queue.get(timeout=30)
            process.join(timeout=10)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(value, {"gemma_first": True, "qwen_same_helper": True,
                "gemma_after_qwen": True, "dflash": True, "torch_loaded": False,
                "cutlass_loaded": False, "caller_env_retained": True})
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            queue.close()
            queue.join_thread()


class OptionalBackendTests(unittest.TestCase):
    def test_lazy_cutlass_import_failure_is_capability_rejection(self):
        class LazyBackend:
            def __getattr__(self, name):
                raise ModuleNotFoundError("No module named 'cutlass'", name="cutlass")
        supported, reason = backend_support_function()(LazyBackend(), name="tensor-FP8")
        self.assertFalse(supported)
        self.assertIn("cutlass", reason)

    def test_dependency_error_inside_support_call_is_also_guarded(self):
        def missing():
            raise ModuleNotFoundError("No module named 'cutlass.cute'", name="cutlass.cute")
        supported, reason = backend_support_function()(NS(is_supported=missing), name="block-FP8")
        self.assertFalse(supported)
        self.assertIn("cutlass.cute", reason)

    def test_available_false_missing_and_invalid_backend_states(self):
        check = backend_support_function()
        self.assertEqual(check(NS(is_supported=lambda: True), name="FP8"), (True, None))
        for backend in (None, NS(), NS(is_supported=lambda: False)):
            self.assertFalse(check(backend, name="FP8")[0])

    def test_runtime_errors_are_not_converted_to_silent_fallbacks(self):
        def broken():
            raise RuntimeError("broken capability implementation")
        with self.assertRaisesRegex(RuntimeError, "broken capability"):
            backend_support_function()(NS(is_supported=broken), name="FP8")

    @unittest.skipUnless(SITE, "Set VLLM_RTX_SPARK_SOURCE_DIR for composed-source integration contracts")
    def test_native_qwen_mxfp8_aot_is_unchanged_by_optional_regular_fp8_guard(self):
        text = (SITE / "vllm/model_executor/kernels/linear/mxfp8/b12x.py").read_text(encoding="utf-8")
        self.assertIn("from original_b12x_aot import (", text)
        self.assertNotIn("check_b12x_backend_support", text)
        regular = (SITE / "vllm/model_executor/kernels/linear/scaled_mm/b12x.py").read_text(encoding="utf-8")
        self.assertEqual(regular.count("return check_b12x_backend_support("), 2)


class LoggingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(sys, "platform", "win32"))
        self.stack.enter_context(patch.dict(os.environ))
        for name in ("PYTHONIOENCODING", "PYTHONUTF8"):
            os.environ.pop(name, None)
        self.stack.enter_context(patch.dict(sys._xoptions, {}, clear=True))

    def test_redirected_ascii_streams_write_utf8_without_replacing_the_streams(self):
        for name, errors in (("stdout", "strict"), ("stderr", "backslashreplace")):
            raw = io.BytesIO()
            stream = io.TextIOWrapper(raw, encoding="ascii", errors=errors)
            with patch.object(sys, name, stream):
                runtime_logging.configure_standard_streams()
                self.assertIs(getattr(sys, name), stream)
                self.assertEqual(stream.errors, errors)
                stream.write("깊이 · →\n")
                stream.flush()
                self.assertEqual(raw.getvalue(), ("깊이 · →" + os.linesep).encode("utf-8"))
            stream.close()

    def test_explicit_python_encoding_policies_are_authoritative(self):
        for option, value in (("PYTHONIOENCODING", "cp1252"), ("PYTHONUTF8", "0"), ("PYTHONUTF8", "1")):
            with self.subTest(option=option, value=value), patch.dict(os.environ, {option: value}):
                stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
                with patch.object(sys, "stdout", stream):
                    runtime_logging.configure_standard_streams()
                    self.assertEqual(stream.encoding, "ascii")
                stream.close()
        with patch.dict(sys._xoptions, {"utf8": "0"}):
            stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
            with patch.object(sys, "stdout", stream):
                runtime_logging.configure_standard_streams()
                self.assertEqual(stream.encoding, "ascii")
            stream.close()

    def test_console_and_application_owned_wrappers_remain_untouched(self):
        class Console(io.TextIOWrapper):
            def isatty(self):
                return True
        stream = Console(io.BytesIO(), encoding="ascii")
        with patch.object(sys, "stdout", stream), patch.object(sys, "stderr", io.StringIO()):
            runtime_logging.configure_standard_streams()
            self.assertEqual(stream.encoding, "ascii")
        stream.close()

    def test_stdlib_file_handlers_get_real_unicode_file_output(self):
        parent = ROOT / ".build/runtime-boundary-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            for index, handler_class in enumerate(sorted(runtime_logging._FILE_HANDLERS)):
                path = Path(temporary) / (str(index) + ".log")
                original = {"version": 1, "disable_existing_loggers": False,
                    "handlers": {"test": {"class": handler_class, "filename": str(path)}},
                    "loggers": {"rtx_boundary_test": {"handlers": ["test"], "level": "INFO", "propagate": False}}}
                prepared = runtime_logging.with_file_handler_encoding(original)
                self.assertNotIn("encoding", original["handlers"]["test"])
                logging.config.dictConfig(prepared)
                logger = logging.getLogger("rtx_boundary_test")
                try:
                    logger.info("깊이 · →")
                    for handler in logger.handlers:
                        handler.flush()
                    self.assertEqual(path.read_bytes(), ("깊이 · →" + os.linesep).encode("utf-8"))
                finally:
                    for handler in logger.handlers:
                        handler.close()
                    logger.handlers.clear()

    def test_explicit_latin_file_encoding_is_used_by_the_real_handler(self):
        parent = ROOT / ".build/runtime-boundary-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            path = Path(temporary) / "latin.log"
            config_map = {"version": 1, "disable_existing_loggers": False,
                "handlers": {"test": {"class": "logging.FileHandler", "filename": str(path), "encoding": "latin-1"}},
                "loggers": {"rtx_boundary_test": {"handlers": ["test"], "level": "INFO", "propagate": False}}}
            logging.config.dictConfig(runtime_logging.with_file_handler_encoding(config_map))
            logger = logging.getLogger("rtx_boundary_test")
            try:
                logger.info("caf\u00e9")
                for handler in logger.handlers:
                    handler.flush()
                self.assertEqual(path.read_bytes(), ("caf\u00e9" + os.linesep).encode("latin-1"))
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()

    def test_explicit_file_encoding_factories_custom_handlers_and_linux_are_preserved(self):
        handlers = {"latin": {"class": "logging.FileHandler", "encoding": "latin-1"},
                    "locale": {"class": "logging.FileHandler", "encoding": None},
                    "custom": {"class": "application.CustomFileHandler"},
                    "factory": {"()": "application.factory", "class": "logging.FileHandler"},
                    "stream": {"class": "logging.StreamHandler"}}
        config_map = {"version": 1, "handlers": handlers}
        self.assertIs(runtime_logging.with_file_handler_encoding(config_map), config_map)
        with patch.object(sys, "platform", "linux"):
            self.assertIs(runtime_logging.with_file_handler_encoding({"handlers": {
                "plain": {"class": "logging.FileHandler"}}})["handlers"]["plain"].get("encoding"), None)

    def test_fresh_spawn_redirected_stdio_and_file_logging_use_utf8(self):
        parent = ROOT / ".build/runtime-boundary-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()
            process = ctx.Process(target=_spawn_logging, args=(queue, str(Path(temporary)/"spawn.log")))
            process.start()
            try:
                value = queue.get(timeout=30)
                process.join(timeout=10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(value, {"encoding": "utf8", "bytes": ("깊이 · →" + os.linesep).encode("utf-8").hex(),
                                         "torch_loaded": False})
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                queue.close()
                queue.join_thread()


class ComposedBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(SITE, "Set VLLM_RTX_SPARK_SOURCE_DIR for composed-source integration contracts")
    def test_common_workers_have_only_instance_depth_references_and_no_recipe_imports(self):
        relative = ("model_runner.py", "cudagraph_utils.py", "structured_outputs.py", "spec_decode/speculator.py",
                    "spec_decode/target_dependent_ar/speculator.py", "spec_decode/target_dependent_ar/cudagraph_utils.py")
        for name in relative:
            text = (SITE / "vllm/v1/worker/gpu" / name).read_text(encoding="utf-8")
            tree = ast.parse(text)
            self.assertNotIn("recipes.qwen3_8_flash_next.helpers", text)
            self.assertFalse(any(isinstance(node, ast.Name) and node.id == "_mbx_depth" for node in ast.walk(tree)))
        runner = (SITE / "vllm/v1/worker/gpu/model_runner.py").read_text(encoding="utf-8")
        self.assertIn("depth_helper=self._mbx_depth", runner)
        cg = (SITE / "vllm/v1/worker/gpu/cudagraph_utils.py").read_text(encoding="utf-8")
        self.assertIn('speculative_config.method == "dflash"', cg)
        logger = (SITE / "vllm/logger.py").read_text(encoding="utf-8")
        self.assertIn("dictConfig(with_file_handler_encoding(logging_config))", logger)


if __name__ == "__main__":
    unittest.main()
