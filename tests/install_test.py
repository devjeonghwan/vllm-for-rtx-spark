"""Run the installer in PowerShell with package processes replaced by functions."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")

# The actual installer runs unchanged. Full-path functions stand in for Python
# executables, including paths containing spaces; no package manager is launched.
HARNESS = r"""
param([string]$Fixture)
$ErrorActionPreference = 'Stop'
$global:scenario = Get-Content -LiteralPath $Fixture -Raw | ConvertFrom-Json
$global:calls = [Collections.Generic.List[object]]::new()
$global:LASTEXITCODE = 0
$env:PATH = Join-Path $scenario.root 'empty path'
$env:VIRTUAL_ENV = $scenario.active
$before = @{PATH=$env:PATH; VIRTUAL_ENV=$env:VIRTUAL_ENV}
foreach ($entry in $scenario.uv_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($entry.Name, [string]$entry.Value)
}
$uvNames = @('UV_EXTRA_INDEX_URL')
foreach ($entry in $scenario.uv_environment.PSObject.Properties) { $uvNames += $entry.Name }
foreach ($name in $uvNames) {
    $before[$name] = @{exists=(Test-Path -LiteralPath ('Env:' + $name)); value=[Environment]::GetEnvironmentVariable($name)}
}

function global:Register-Python([string]$Executable) {
    Set-Item -LiteralPath ('Function:global:' + $Executable) -Value {
        $name = $MyInvocation.MyCommand.Name
        Invoke-FixturePython $name @args
    }
}
function global:Add-Call([string]$Kind, [string]$Executable, [object[]]$Arguments) {
    $call = @{kind=$Kind; executable=$Executable; args=@($Arguments | ForEach-Object { [string]$_ })}
    if ($Executable -eq 'uv') {
        $call.environment = @{}
        foreach ($entry in $scenario.uv_environment.PSObject.Properties) {
            $call.environment[$entry.Name] = [Environment]::GetEnvironmentVariable($entry.Name)
        }
    }
    $global:calls.Add($call)
    $global:LASTEXITCODE = 0
    if ($scenario.fail -eq $Kind) { $global:LASTEXITCODE = 29 }
}
function global:New-FixtureEnvironment([string]$Directory) {
    $scripts = Join-Path $Directory 'Scripts'
    New-Item -ItemType Directory -Path $scripts -Force | Out-Null
    $executable = Join-Path $scripts 'python.exe'
    Set-Content -LiteralPath $executable -Value 'inert executable fixture'
    Register-Python $executable
}
function global:Invoke-FixturePython([string]$Executable) {
    $arguments = @($args)
    if ($arguments[0] -eq '-c') {
        $code = $arguments[1]
        if ($code.Contains('version_info')) {
            Add-Call 'identity' $Executable $arguments
            if ($scenario.identity_error) { $global:LASTEXITCODE = 31; return }
            $isVenv = $Executable -ne 'python' -and $Executable -ne $scenario.base_python
            if ($scenario.not_venv) { $isVenv = $false }
            $identity = @('win32', 'arm64', @(3,13), $isVenv)
            if ($Executable -eq 'python' -and $scenario.wrong_path) { $identity[1] = 'amd64' }
            if ($scenario.machine) { $identity[1] = $scenario.machine }
            if ($scenario.version) { $identity[2] = @($scenario.version) }
            ConvertTo-Json -InputObject $identity -Compress
            return
        }
        if ($code.Contains('distributions')) {
            Add-Call 'legacy' $Executable $arguments
            if ($scenario.legacy) { 'True' } else { 'False' }
            return
        }
        throw "Unexpected Python code: $code"
    }
    if ($arguments[0] -eq '-m' -and $arguments[1] -eq 'venv') {
        Add-Call 'create' $Executable $arguments
        if (-not $LASTEXITCODE) { New-FixtureEnvironment $arguments[-1] }
        return
    }
    if ($arguments[0] -eq '-m' -and $arguments[1] -eq 'pip') {
        if ($arguments -contains '--version') {
            Add-Call 'pip-version' $Executable $arguments
            if ($scenario.no_pip) { $global:LASTEXITCODE = 32 }
            return
        }
        if ($scenario.no_pip) { throw 'The uv fixture has no pip module.' }
        $kind = 'install'
        if ($arguments -contains 'uninstall') { $kind = 'uninstall' }
        if ($arguments -contains '--force-reinstall') { $kind = 'reinstall' }
        Add-Call $kind $Executable $arguments
        return
    }
    if ($arguments[0] -eq '-m' -and $arguments[1] -eq 'vllm._rtx_spark.runtime.check') {
        Add-Call 'doctor' $Executable $arguments
        return
    }
    throw "Unexpected Python arguments: $($arguments -join ' ')"
}
Register-Python 'python'
Register-Python $scenario.base_python
foreach ($existing in @($scenario.existing)) { Register-Python $existing }
if (-not $scenario.missing_uv) {
    function global:uv {
        $arguments = @($args)
        if ($arguments[0] -eq '--version') { Add-Call 'uv-version' 'uv' $arguments; 'uv fixture'; return }
        if ($arguments[0] -eq 'python' -and $arguments[1] -eq 'find') {
            Add-Call 'resolve' 'uv' $arguments
            $scenario.base_python
            return
        }
        if ($arguments[0] -eq 'venv') {
            Add-Call 'create' 'uv' $arguments
            if (-not $LASTEXITCODE) {
                $directory = $arguments | Where-Object { $_ -eq $scenario.target }
                if (-not $directory) { throw 'uv did not receive the target environment.' }
                New-FixtureEnvironment $directory
            }
            return
        }
        if ($arguments[0] -eq 'pip') {
            $kind = 'install'
            if ($arguments -contains '-r') { $kind = 'vendor' }
            if ($arguments -contains 'uninstall') { $kind = 'uninstall' }
            if ($arguments -contains '--force-reinstall' -or $arguments -contains '--reinstall') { $kind = 'reinstall' }
            Add-Call $kind 'uv' $arguments
            return
        }
        throw "Unexpected uv arguments: $($arguments -join ' ')"
    }
}
$result = @{error=$null; calls=$global:calls; before=$before}
try {
    $options = @{}
    if ($scenario.explicit) { $options.Environment = $scenario.environment_argument }
    if ($scenario.use_uv) { $options.UseUv = $true }
    if ($scenario.explicit_python) { $options.Python = $scenario.base_python }
    & (Join-Path $scenario.root 'scripts/install.ps1') @options
} catch { $result.error = $_.Exception.Message }
$result.after = @{PATH=$env:PATH; VIRTUAL_ENV=$env:VIRTUAL_ENV}
foreach ($name in $uvNames) {
    $result.after[$name] = @{exists=(Test-Path -LiteralPath ('Env:' + $name)); value=[Environment]::GetEnvironmentVariable($name)}
}
ConvertTo-Json -InputObject $result -Depth 12 | Set-Content -LiteralPath $scenario.result -Encoding utf8
"""


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Requires Windows PowerShell")
class InstallerTests(unittest.TestCase):
    @contextmanager
    def installation(self, **options):
        parent = ROOT / ".build/install-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rtx spark install ", dir=parent) as directory:
            root = Path(directory) / "release with spaces"
            (root / "scripts").mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/install.ps1", root / "scripts/install.ps1")
            (root / "empty path").mkdir()
            (root / "requirements").mkdir()
            constraints = root / "requirements/constraints.txt"
            constraints.write_text("fixture==1\n", encoding="utf-8")
            (root / "requirements/runtime-vendor.txt").write_text("torch==2.14.0+cu134\n", encoding="utf-8")
            manifest = {
                "platform": {"os": "windows", "cpu": "arm64", "python": "3.13"},
                "requirements": {
                    "constraints": "requirements/constraints.txt",
                    "runtime_vendor": "requirements/runtime-vendor.txt",
                },
            }
            manifest_json = json.dumps(manifest)
            (root / "manifest.json").write_text(manifest_json, encoding="utf-8")
            wheel = root / "wheels/vllm-fixture-cp313-cp313-win_arm64.whl"
            wheel.parent.mkdir()
            wheel.write_bytes(b"not a wheel; package processes are intercepted")
            target = root / "selected env"
            active = root / "active env"
            default = root / ".venv"
            config = dict(
                root=str(root), target=str(target), active=None, explicit=True,
                use_uv=False, base_python=str(root / "base Python/python.exe"),
                explicit_python=False, existing=[], result=str(root / "result.json"), uv_environment={},
            )
            config.update(options)
            base_python = Path(config["base_python"])
            base_python.parent.mkdir(parents=True)
            base_python.write_text("inert executable fixture", encoding="utf-8")
            if config.pop("relative_environment", False):
                target = root / "relative selected env"
                config["target"] = str(target)
                config["environment_argument"] = "relative selected env"
            else:
                config["environment_argument"] = str(target)
            if config.pop("source_checkout", False):
                (root / "manifest.json").unlink()
                source_manifest = root / "patches/vllm/manifest.json"
                source_manifest.parent.mkdir(parents=True)
                source_manifest.write_text(manifest_json, encoding="utf-8")
            if config.pop("active_environment", False):
                config["active"] = str(active)
            if not config["explicit"]:
                target = active if config["active"] and config["use_uv"] else default
                config["target"] = str(target)
            existing = []
            if config.pop("target_exists", False):
                existing.append(target)
            if config["active"]:
                existing.append(active)
            for environment in existing:
                (environment / "Scripts").mkdir(parents=True, exist_ok=True)
                (environment / "Scripts/python.exe").write_text("inert", encoding="utf-8")
                (environment / "keep.txt").write_text("user environment", encoding="utf-8")
                config["existing"].append(str(environment / "Scripts/python.exe"))
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps(config), encoding="utf-8")
            harness = root / "harness.ps1"
            harness.write_text(HARNESS, encoding="utf-8")
            process = subprocess.run(
                [POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(harness), "-Fixture", str(fixture)],
                capture_output=True, text=True, encoding="utf-8", timeout=25,
                env={key: value for key, value in os.environ.items() if not key.upper().startswith("UV_")},
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads(Path(config["result"]).read_text(encoding="utf-8-sig"))
            self.assertEqual(result["before"], result["after"], "Installer must not activate or alter PATH")
            for environment in existing:
                self.assertEqual((environment / "keep.txt").read_text(), "user environment")
            yield root, target, wheel, constraints, result

    @staticmethod
    def calls(result, kind):
        return [call for call in result["calls"] if call["kind"] == kind]

    def assert_success(self, result):
        self.assertIsNone(result["error"], result["error"])
        self.assertEqual(len(self.calls(result, "doctor")), 1)

    def assert_package_arguments(self, result, target, wheel, constraints, uv=False):
        install, = self.calls(result, "install")
        reinstall, = self.calls(result, "reinstall")
        self.assertEqual(install["args"][-1], str(wheel))
        self.assertIn("--only-binary=:all:", install["args"])
        self.assertIn("https://pypi.org/simple" if uv else "https://pypi.nvidia.com/nvtorch_oot/", install["args"])
        self.assertEqual(install["args"][install["args"].index("-c") + 1], str(constraints))
        self.assertIn("--no-deps", reinstall["args"])
        self.assertEqual(reinstall["args"][-1], str(wheel))
        vendor = self.calls(result, "vendor")
        if uv:
            vendor_install, = vendor
            self.assertIn("--no-deps", vendor_install["args"])
            self.assertIn(str(constraints.parent / "runtime-vendor.txt"), vendor_install["args"])
            self.assertIn("https://pypi.nvidia.com/nvtorch_oot/", vendor_install["args"])
            self.assertIn("--no-index", reinstall["args"])
            for call in (vendor_install, install):
                self.assertEqual(call["args"][call["args"].index("--index-strategy") + 1], "first-index")
                self.assertIn(str(constraints), call["args"])
        else:
            self.assertEqual(vendor, [])
        for call in (install, reinstall, *vendor, *self.calls(result, "uninstall")):
            if uv:
                self.assertEqual(call["executable"], "uv")
                self.assertEqual(call["args"][call["args"].index("--python") + 1], str(target / "Scripts/python.exe"))
            else:
                self.assertEqual(call["executable"], str(target / "Scripts/python.exe"))
                self.assertEqual(call["args"][:2], ["-m", "pip"])
        doctor, = self.calls(result, "doctor")
        self.assertEqual(doctor["executable"], str(target / "Scripts/python.exe"))
        self.assertEqual(doctor["args"], ["-m", "vllm._rtx_spark.runtime.check", "doctor"])

    def test_default_pip_creates_requested_environment_and_installs_one_wheel(self):
        with self.installation(explicit_python=True) as (_, target, wheel, constraints, result):
            self.assert_success(result)
            create, = self.calls(result, "create")
            self.assertEqual(create["args"], ["-m", "venv", str(target)])
            self.assert_package_arguments(result, target, wheel, constraints)
            self.assertEqual(self.calls(result, "uninstall"), [])

    def test_uv_creates_unseeded_environment_and_removes_legacy_package(self):
        with self.installation(use_uv=True, no_pip=True, legacy=True, explicit_python=True) as (_, target, wheel, constraints, result):
            self.assert_success(result)
            create, = self.calls(result, "create")
            self.assertEqual(create["executable"], "uv")
            self.assertNotIn("--seed", create["args"])
            self.assertIn(str(target), create["args"])
            self.assert_package_arguments(result, target, wheel, constraints, uv=True)
            uninstall, = self.calls(result, "uninstall")
            self.assertIn("vllm-rtx-spark", uninstall["args"])

    def test_existing_target_wins_over_wrong_path_python(self):
        for uv in (False, True):
            with self.subTest(uv=uv), self.installation(target_exists=True, wrong_path=True, use_uv=uv, no_pip=uv) as (_, target, wheel, constraints, result):
                self.assert_success(result)
                self.assertEqual(self.calls(result, "create"), [])
                self.assertTrue(all(call["executable"] != "python" for call in result["calls"]))
                self.assert_package_arguments(result, target, wheel, constraints, uv)

    def test_omitted_environment_reuses_active_environment(self):
        with self.installation(explicit=False, active_environment=True, wrong_path=True, use_uv=True, no_pip=True) as (root, target, wheel, constraints, result):
            self.assert_success(result)
            self.assertEqual(target, root / "active env")
            self.assertFalse((root / ".venv").exists())
            self.assertEqual(self.calls(result, "create"), [])
            self.assert_package_arguments(result, target, wheel, constraints, uv=True)

    def test_explicit_target_wins_over_active_environment(self):
        with self.installation(active_environment=True, use_uv=True, no_pip=True) as (root, target, wheel, constraints, result):
            self.assert_success(result)
            self.assertEqual(len(self.calls(result, "create")), 1)
            self.assertNotEqual(target, root / "active env")
            self.assert_package_arguments(result, target, wheel, constraints, uv=True)

    def test_default_environment_is_release_relative(self):
        for uv, active in ((False, False), (False, True), (True, False)):
            with self.subTest(uv=uv, active=active), self.installation(explicit=False, use_uv=uv, active_environment=active) as (root, target, wheel, constraints, result):
                self.assert_success(result)
                self.assertEqual(target, root / ".venv")
                self.assert_package_arguments(result, target, wheel, constraints, uv)

    def test_relative_target_is_anchored_at_release_directory(self):
        with self.installation(relative_environment=True, use_uv=True) as (root, target, wheel, constraints, result):
            self.assert_success(result)
            self.assertEqual(target, root / "relative selected env")
            self.assert_package_arguments(result, target, wheel, constraints, uv=True)
            resolve, = self.calls(result, "resolve")
            self.assertIn("cpython-3.13-windows-aarch64-none", resolve["args"])
            self.assertIn("--no-python-downloads", resolve["args"])

    def test_existing_target_can_read_source_manifest_without_path_python(self):
        with self.installation(source_checkout=True, target_exists=True, wrong_path=True, use_uv=True) as (_, target, _, _, result):
            self.assert_success(result)
            self.assertTrue(all(call["executable"] != "python" for call in result["calls"]))

    def test_missing_uv_fails_before_environment_or_package_changes(self):
        with self.installation(use_uv=True, missing_uv=True, target_exists=True) as (_, _, _, _, result):
            self.assertIsNotNone(result["error"])
            self.assertIn("uv", result["error"].lower())
            self.assertFalse(any(call["kind"] in {"create", "install", "uninstall", "reinstall", "doctor"} for call in result["calls"]))

    def test_incompatible_existing_interpreter_is_rejected_before_packages(self):
        for identity in ({"machine": "amd64"}, {"version": [3, 12]}, {"identity_error": True}, {"not_venv": True}):
            with self.subTest(identity=identity), self.installation(target_exists=True, **identity) as (_, _, _, _, result):
                self.assertIsNotNone(result["error"])
                self.assertFalse(any(call["kind"] in {"create", "install", "uninstall", "reinstall", "doctor"} for call in result["calls"]))

    def test_pip_requires_pip_module_in_selected_environment(self):
        with self.installation(target_exists=True, no_pip=True) as (_, _, _, _, result):
            self.assertIsNotNone(result["error"])
            self.assertIn("UseUv", result["error"])
            self.assertEqual(len(self.calls(result, "pip-version")), 1)
            self.assertEqual(self.calls(result, "install"), [])

    def test_creation_failure_stops_before_install(self):
        for uv in (False, True):
            with self.subTest(uv=uv), self.installation(use_uv=uv, fail="create", active_environment=True) as (_, _, _, _, result):
                self.assertIsNotNone(result["error"])
                self.assertEqual(len(self.calls(result, "create")), 1)
                self.assertEqual(self.calls(result, "install"), [])

    def test_package_and_doctor_failures_stop_at_the_failed_stage(self):
        for uv in (False, True):
            stages = (["vendor"] if uv else []) + ["install", "legacy", "uninstall", "reinstall", "doctor"]
            for failed in stages:
                with self.subTest(uv=uv, failed=failed), self.installation(use_uv=uv, no_pip=uv, legacy=True, target_exists=True, fail=failed) as (_, _, _, _, result):
                    self.assertIsNotNone(result["error"])
                    observed = [call["kind"] for call in result["calls"] if call["kind"] in stages]
                    self.assertEqual(observed, stages[:stages.index(failed) + 1])

    def test_uv_ignores_inherited_install_defaults_and_restores_them_on_failure(self):
        defaults = {
            "UV_DEFAULT_INDEX": "https://fixture.invalid/simple",
            "UV_INDEX": "https://fixture.invalid/extra",
            "UV_FIND_LINKS": "C:\\fixture local wheels",
            "UV_CONFIG_FILE": "C:\\fixture config\\uv.toml",
            "UV_SYSTEM_PYTHON": "1", "UV_VENV_CLEAR": "1", "UV_VENV_SEED": "1",
        }
        for failure in (None, "vendor", "doctor"):
            with self.subTest(failure=failure), self.installation(use_uv=True, uv_environment=defaults, fail=failure) as (_, _, _, _, result):
                if failure is None:
                    self.assert_success(result)
                else:
                    self.assertIsNotNone(result["error"])
                uv_calls = [call for call in result["calls"] if call["executable"] == "uv"]
                self.assertTrue(uv_calls)
                for call in uv_calls:
                    self.assertTrue(all(value is None for value in call["environment"].values()), call)


if __name__ == "__main__":
    unittest.main()
