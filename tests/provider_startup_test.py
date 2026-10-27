"""Provider metadata must be replayed in fresh Windows spawn workers."""

from contextlib import ExitStack
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()

from _rtx_spark.kernels import csf, cute, providers  # noqa: E402
from _rtx_spark.kernels.cute import builder  # noqa: E402
from _rtx_spark.runtime import bootstrap  # noqa: E402


FIXTURE = '''
REGISTER_CALLS = 0
CONFIGURE_CALLS = 0

def configure():
    global CONFIGURE_CALLS
    CONFIGURE_CALLS += 1

def compiler(*args):
    raise RuntimeError("CPU fixture compiler selected")

def register():
    global REGISTER_CALLS
    REGISTER_CALLS += 1
    from _rtx_spark.kernels import csf, cute
    from _rtx_spark.kernels.cute import builder
    cute.register_runtime("spawn-fixture", configure)
    csf.register_csf_provider(csf.CsfProvider("spawn-fixture", "csf_host"))
    builder.register_provider("spawn-fixture", compiler)
'''


def _spawn_worker(queue, module_name):
    """Imported afresh by multiprocessing; parent registries are unavailable."""
    before = {name for name in ("torch", "cutlass", "triton") if name in sys.modules}
    try:
        bootstrap.initialize()
        bootstrap.initialize()
        module = importlib.import_module(module_name)
        resolved = csf.resolve_csf_provider("spawn-fixture")
        queue.put({"register_calls": module.REGISTER_CALLS,
                   "configure_calls": module.CONFIGURE_CALLS,
                   "cute": "spawn-fixture" in cute._runtime_providers,
                   "csf": resolved["provider"].name,
                   "compiler": "spawn-fixture" in builder._PROVIDERS,
                   "GPU_imports_added": sorted({name for name in ("torch", "cutlass", "triton")
                                                if name in sys.modules} - before)})
    except BaseException:
        queue.put({"error": traceback.format_exc()})


class ProviderStartupTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".build/provider-startup-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=parent)))
        self.stack.enter_context(patch.object(sys, "path", [str(self.directory), *sys.path]))
        self.stack.enter_context(patch.dict(os.environ, {
            "VLLM_RTX_SPARK_PROVIDER_MODULES": "[]",
            "VLLM_RTX_SPARK_CUTE_MODE": "spawn-fixture",
            "VLLM_RTX_SPARK_CACHE": str(self.directory / "cache"),
        }))
        self.stack.enter_context(patch.object(providers, "PROVIDER_MODULES", ()))
        self.stack.enter_context(patch.object(providers, "_initialized_modules", set()))
        self.stack.enter_context(patch.object(providers, "_initializing", False))
        self.stack.enter_context(patch.dict(cute._runtime_providers, {}, clear=True))
        self.stack.enter_context(patch.object(cute, "_preferred_runtime", "aot"))
        self.stack.enter_context(patch.object(cute, "_installed_mode", None))
        self.stack.enter_context(patch.dict(csf._CSF_PROVIDERS, dict(csf._CSF_PROVIDERS), clear=True))
        self.stack.enter_context(patch.dict(builder._PROVIDERS, dict(builder._PROVIDERS), clear=True))
        self.stack.enter_context(patch.object(bootstrap, "_initialized", False))
        self.module_names = []
        self.addCleanup(self.remove_fixture_modules)

    def remove_fixture_modules(self):
        for name in self.module_names:
            sys.modules.pop(name, None)

    def fixture(self, name="rtx_provider_spawn_fixture", content=FIXTURE):
        (self.directory / (name + ".py")).write_text(content, encoding="utf-8")
        importlib.invalidate_caches()
        self.module_names.append(name)
        return name

    def test_registration_replays_once_in_parent_and_real_spawn_worker(self):
        module_name = self.fixture()
        before = {name for name in ("torch", "cutlass", "triton") if name in sys.modules}
        providers.register_provider_module(module_name)
        providers.register_provider_module(module_name)
        bootstrap.initialize()
        bootstrap.initialize()
        module = importlib.import_module(module_name)
        self.assertEqual(module.REGISTER_CALLS, 1)
        self.assertEqual(module.CONFIGURE_CALLS, 1)
        self.assertEqual(json.loads(os.environ[providers._ENVIRONMENT]), [module_name])
        self.assertEqual(before, {name for name in ("torch", "cutlass", "triton") if name in sys.modules})
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        child = context.Process(target=_spawn_worker, args=(queue, module_name))
        child.start()
        try:
            result = queue.get(timeout=30)
            child.join(timeout=15)
            self.assertFalse(child.is_alive(), "CPU provider worker did not exit")
            self.assertEqual(child.exitcode, 0)
            self.assertEqual(result, {"register_calls": 1, "configure_calls": 1,
                                     "cute": True, "csf": "spawn-fixture", "compiler": True,
                                     "GPU_imports_added": []})
        finally:
            if child.is_alive():
                child.terminate()
                child.join(timeout=10)
            queue.close()
            queue.join_thread()

    def test_failed_callback_rolls_back_metadata_and_can_retry(self):
        module_name = self.fixture("rtx_provider_retry_fixture", '''
ATTEMPTS = 0
def register():
    global ATTEMPTS
    ATTEMPTS += 1
    from _rtx_spark.kernels import csf
    csf.register_csf_provider(csf.CsfProvider("retry-fixture", "csf_host"))
    if ATTEMPTS == 1:
        raise RuntimeError("fixture failure after registration")
''')
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            providers.register_provider_module(module_name)
        self.assertNotIn("retry-fixture", csf._CSF_PROVIDERS)
        self.assertNotIn(module_name, providers._initialized_modules)
        self.assertEqual(os.environ[providers._ENVIRONMENT], "[]")
        providers.register_provider_module(module_name)
        self.assertEqual(importlib.import_module(module_name).ATTEMPTS, 2)
        self.assertIn("retry-fixture", csf._CSF_PROVIDERS)

    def test_conflicting_module_fails_without_replacing_first_provider(self):
        first = self.fixture("rtx_provider_first_fixture")
        second = self.fixture("rtx_provider_conflict_fixture")
        providers.register_provider_module(first)
        original = cute._runtime_providers["spawn-fixture"]
        with self.assertRaisesRegex(ValueError, "duplicate runtime provider"):
            providers.register_provider_module(second)
        self.assertIs(cute._runtime_providers["spawn-fixture"], original)
        self.assertEqual(json.loads(os.environ[providers._ENVIRONMENT]), [first])
        self.assertNotIn(second, providers._initialized_modules)

    def test_lookup_and_build_entrypoints_initialize_declared_modules(self):
        module_name = self.fixture()
        os.environ[providers._ENVIRONMENT] = json.dumps([module_name])
        self.assertEqual(csf.resolve_csf_provider("spawn-fixture")["provider"].name, "spawn-fixture")
        cute.configure("spawn-fixture")
        with self.assertRaisesRegex(RuntimeError, "CPU fixture compiler selected"):
            builder.build(self.directory / "compiler-output", provider="spawn-fixture")
        self.assertEqual(importlib.import_module(module_name).REGISTER_CALLS, 1)

    def test_bad_declarations_fail_before_build_output_is_created(self):
        output = self.directory / "must-not-exist"
        os.environ[providers._ENVIRONMENT] = "{}"
        with self.assertRaisesRegex(ValueError, "JSON list"):
            builder.build(output)
        self.assertFalse(output.exists())
        for name in ("", "file.py/path", "../provider", 7):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "importable module"):
                providers.register_provider_module(name)

    def test_registration_module_must_expose_callback(self):
        module_name = self.fixture("rtx_provider_missing_callback", "VALUE = 7\n")
        with self.assertRaisesRegex(TypeError, "expose register"):
            providers.register_provider_module(module_name)
        self.assertEqual(os.environ[providers._ENVIRONMENT], "[]")
        self.assertNotIn(module_name, providers._initialized_modules)


if __name__ == "__main__":
    unittest.main()
