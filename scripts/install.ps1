param(
    [string]$Python = 'python',
    [string]$Environment = '.venv',
    [string]$WheelDirectory = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$manifestPath = Join-Path $root 'manifest.json'
if (Test-Path -LiteralPath $manifestPath) {
    $manifestSpec = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
} else {
    $manifestJson = & $Python -c 'import json,runpy,sys; v=runpy.run_path(sys.argv[1]); print(json.dumps(v["load_manifest"]()))' `
        (Join-Path $root 'src/version.py')
    if ($LASTEXITCODE) { throw 'Cannot read the release manifest.' }
    $manifestSpec = $manifestJson | ConvertFrom-Json
}
if ($manifestSpec.platform.os -ne 'windows' -or $manifestSpec.platform.cpu -ne 'arm64') {
    throw 'This release requires native Windows ARM64 installation.'
}
$pythonVersion = $manifestSpec.platform.python
if (-not $WheelDirectory) {
    $WheelDirectory = Join-Path $root 'wheels'
    if (-not (Test-Path -LiteralPath $WheelDirectory)) { $WheelDirectory = Join-Path $root 'dist' }
}
function Assert-NativePython([string]$Executable) {
    $identity = & $Executable -c 'import json,platform,sys; print(json.dumps([sys.platform,platform.machine().lower(),list(sys.version_info[:2])]))'
    if ($LASTEXITCODE) { throw "Cannot run Python. Supply -Python with a Python $pythonVersion ARM64 executable." }
    $identity = $identity | ConvertFrom-Json
    if ($identity[0] -ne 'win32' -or $identity[1] -notin @('arm64','aarch64') -or ($identity[2] -join '.') -ne $pythonVersion) {
        throw "This release requires native Windows ARM64 Python $pythonVersion."
    }
}
Assert-NativePython $Python
$wheelRoot = (Resolve-Path -LiteralPath $WheelDirectory).Path
$wheelPaths = @(Get-ChildItem -LiteralPath $wheelRoot -Filter 'vllm-*-win_arm64.whl' -File | ForEach-Object FullName)
if ($wheelPaths.Count -ne 1) {
    throw 'Expected one Windows ARM64 vLLM wheel.'
}
if (-not [IO.Path]::IsPathRooted($Environment)) { $Environment = Join-Path $root $Environment }
$envPython = Join-Path $Environment 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $envPython)) {
    & $Python -m venv $Environment
    if ($LASTEXITCODE) { throw 'Virtual environment creation failed.' }
}
Assert-NativePython $envPython
& $envPython -m pip install --only-binary=:all: --extra-index-url https://pypi.nvidia.com/nvtorch_oot/ `
    -c (Join-Path $root $manifestSpec.requirements.constraints) @wheelPaths
if ($LASTEXITCODE) { throw 'Package installation failed.' }
$legacyInstalled = & $envPython -c 'import importlib.metadata as m; print(any(d.metadata["Name"] == "vllm-rtx-spark" for d in m.distributions()))'
if ($LASTEXITCODE) { throw 'Cannot inspect the virtual environment.' }
if ($legacyInstalled -eq 'True') {
    & $envPython -m pip uninstall -y vllm-rtx-spark
    if ($LASTEXITCODE) { throw 'Previous companion package removal failed.' }
}
& $envPython -m pip install --force-reinstall --no-deps @wheelPaths
if ($LASTEXITCODE) { throw 'vLLM wheel installation failed.' }
& $envPython -m vllm._rtx_spark.runtime.check doctor
if ($LASTEXITCODE) { throw 'Installation checks failed.' }
Write-Host "Ready: $Environment\Scripts\vllm.exe"
