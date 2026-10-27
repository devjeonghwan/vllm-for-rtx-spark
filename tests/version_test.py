"""A project-version edit must propagate without changing pinned source patches."""

import ast
import importlib
import importlib.metadata
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src/version.py"
RUN_PATH = runpy.run_path


def future_release():
    """Change exactly one assignment in memory; leave the checkout untouched."""
    tree = ast.parse(VERSION_FILE.read_text(encoding="utf-8"))
    assignments = [
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PROJECT_VERSION"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise AssertionError("Expected one authoritative PROJECT_VERSION assignment")
    current = assignments[0].value
    if not isinstance(current, ast.Constant) or not isinstance(current.value, str):
        raise AssertionError("PROJECT_VERSION must be editable as one string value")
    # Choose a later release without duplicating the current release literal.
    next_major = Version(current.value).major + 1
    assignments[0].value = ast.copy_location(ast.Constant(f"{next_major}.2.3"), current)
    result = {"__file__": str(VERSION_FILE), "__name__": "future_rtx_spark_version"}
    exec(compile(ast.fix_missing_locations(tree), str(VERSION_FILE), "exec"), result)
    return result


def rendered_module(version, commit, **kwargs):
    source = version["render_vllm_version"](commit, **kwargs)
    if not isinstance(source, bytes):
        raise AssertionError("The generated version payload must be bytes")
    namespace = {}
    exec(compile(source, "vllm/_version.py", "exec"), namespace)
    return source, namespace


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.current = RUN_PATH(VERSION_FILE)
        self.future = future_release()

    def test_one_project_version_edit_updates_runtime_and_preserves_upstream(self):
        self.assertNotEqual(self.current["PROJECT_VERSION"], self.future["PROJECT_VERSION"])
        self.assertEqual(self.current["UPSTREAM_VERSION"], self.future["UPSTREAM_VERSION"])
        self.assertNotEqual(self.current["WHEEL_VERSION"], self.future["WHEEL_VERSION"])
        self.assertNotEqual(self.current["WHEEL_VERSION_TUPLE"], self.future["WHEEL_VERSION_TUPLE"])

        for version in (self.current, self.future):
            with self.subTest(project_version=version["PROJECT_VERSION"]):
                _, module = rendered_module(version, 'test-commit-"quoted"\\path\n')
                self.assertEqual(module["__version__"], version["WHEEL_VERSION"])
                self.assertEqual(module["__version_tuple__"], version["WHEEL_VERSION_TUPLE"])
                self.assertEqual(module["__commit_id__"], 'test-commit-"quoted"\\path\n')
                self.assertEqual(module["version"], module["__version__"])
                self.assertEqual(module["version_tuple"], module["__version_tuple__"])
                self.assertEqual(module["commit_id"], module["__commit_id__"])

        current_source, current_base = rendered_module(self.current, "pinned", upstream_only=True)
        future_source, future_base = rendered_module(self.future, "pinned", upstream_only=True)
        self.assertEqual(current_source, future_source)
        self.assertEqual(current_base["__version__"], self.current["UPSTREAM_VERSION"])
        self.assertEqual(current_base["__version_tuple__"],
                         tuple(int(part) for part in self.current["UPSTREAM_VERSION"].split(".")))
        self.assertEqual(future_base["__version__"], current_base["__version__"])

    def test_wheel_filename_accepts_the_underscore_local_version(self):
        for version in (self.current, self.future):
            with self.subTest(project_version=version["PROJECT_VERSION"]):
                wheel_version = version["WHEEL_VERSION"]
                self.assertIn("+rtx_spark.", wheel_version)
                distribution, parsed, build, tags = parse_wheel_filename(
                    f"vllm-{wheel_version}-cp313-cp313-win_arm64.whl")
                self.assertEqual(distribution, "vllm")
                self.assertEqual(parsed, Version(wheel_version))
                self.assertEqual(parsed.release,
                                 tuple(int(part) for part in version["UPSTREAM_VERSION"].split(".")))
                self.assertEqual(parsed.local, "rtx.spark." + version["PROJECT_VERSION"])
                self.assertEqual(build, ())
                self.assertEqual({str(tag) for tag in tags}, {"cp313-cp313-win_arm64"})

    def test_builder_and_packager_follow_the_same_single_edit(self):
        def redirected_version(path, *args, **kwargs):
            if Path(path).resolve() == VERSION_FILE.resolve():
                return dict(self.future)
            return RUN_PATH(path, *args, **kwargs)

        # Loading these modules must not run their build/archive entry points.
        with patch("runpy.run_path", side_effect=redirected_version), \
                patch("subprocess.run", side_effect=AssertionError("No build process in version tests")), \
                patch("subprocess.check_output", side_effect=AssertionError("No Git/process query in version tests")), \
                patch("zipfile.ZipFile", side_effect=AssertionError("No archives in version tests")):
            builder = RUN_PATH(ROOT / "scripts/build_vllm.py")
            packager = RUN_PATH(ROOT / "scripts/package_release.py")

        self.assertEqual(builder["PROJECT_VERSION"], self.future["PROJECT_VERSION"])
        self.assertEqual(builder["UPSTREAM_VERSION"], self.future["UPSTREAM_VERSION"])
        self.assertEqual(builder["VERSION"], self.future["WHEEL_VERSION"])
        self.assertEqual(builder["PATCH_ROOT"], ROOT / "patches/vllm")
        self.assertEqual(builder["render_vllm_version"]("pinned"),
                         self.future["render_vllm_version"]("pinned"))
        self.assertEqual(packager["VERSION"], self.future["PROJECT_VERSION"])
        self.assertEqual(packager["WHEEL_VERSION"], self.future["WHEEL_VERSION"])
        self.assertEqual(packager["CORE_WHEEL_PREFIX"], "vllm-" + self.future["WHEEL_VERSION"] + "-")

    def test_one_manifest_pins_the_upstream_commit_independently_of_project_version(self):
        lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8-sig"))
        manifest = self.current["load_manifest"]()
        self.assertEqual(self.current["manifest_path"](), ROOT / "patches/vllm/manifest.json")
        self.assertEqual(manifest["vllm"]["version"], "0.32.0")
        self.assertEqual(manifest["vllm"]["commit"], "5df63a7203cdb9328d824a27b682e77c7b9928be")
        self.assertNotIn("targets", lock)
        self.assertNotIn("vllm_manifest", lock)
        self.assertNotIn("files", manifest)
        self.assertNotIn("overlay_manifest", manifest)
        for version in (self.current, self.future):
            self.assertEqual(version["load_manifest"](), manifest)
            _, generated = rendered_module(version, None, upstream_only=True)
            self.assertEqual(generated["__version__"], manifest["vllm"]["version"])
            self.assertEqual(generated["__commit_id__"], manifest["vllm"]["commit"])

    def test_runtime_inputs_live_in_the_regular_private_source_package(self):
        manifest = self.current["load_manifest"]()
        for relative in ("runtime/vllm/adapter.py", "recipes/qwen3_8/helpers/mbx_loader.py"):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / "src" / relative).is_file())
        self.assertNotIn("resolve_version_input", self.current)
        self.assertTrue(all(isinstance(name, str) for name in manifest["patches"]))
        self.assertTrue(all(set(rule) == {"source", "destination"} for rule in manifest["copy"]))

    def test_constraints_do_not_override_a_newly_built_vllm_version(self):
        constraints = ROOT / self.current["load_manifest"]()["requirements"]["constraints"]
        for number, raw in enumerate(constraints.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.partition("#")[0].strip()
            if line:
                requirement = Requirement(line)
                self.assertNotEqual(canonicalize_name(requirement.name), "vllm",
                                    f"{constraints.name}:{number} pins the generated vLLM release")

    def test_doctor_accepts_canonical_local_version_spellings(self):
        RUN_PATH(ROOT / "scripts/load_source_package.py")["load"]()
        runtime = importlib.import_module("_rtx_spark.runtime")
        doctor = importlib.import_module("_rtx_spark.runtime.check")
        version_module = importlib.import_module("_rtx_spark.version")
        raw = self.current["WHEEL_VERSION"]
        canonical = str(Version(raw))
        empty_resources = SimpleNamespace(rglob=lambda pattern: ())
        cases = ((raw, canonical, True), (canonical, raw, True),
                 (raw, self.future["WHEEL_VERSION"], False), (raw, "missing", False))
        for expected, installed, matches in cases:
            with self.subTest(expected=expected, installed=installed):
                def installed_version(name):
                    if name == "vllm":
                        return installed
                    raise importlib.metadata.PackageNotFoundError(name)

                # Inspect only the version row; no native assets or GPU modules.
                with patch.object(version_module, "wheel_version", return_value=expected), \
                        patch.object(runtime, "resources_root", return_value=empty_resources), \
                        patch.object(doctor.importlib.metadata, "version", side_effect=installed_version):
                    report = doctor.inspect_installation()
                row = next(item for item in report["checks"] if item["name"] == "vllm")
                self.assertEqual(row["ok"], matches)
                self.assertEqual(row["detail"], installed)


if __name__ == "__main__":
    unittest.main()
