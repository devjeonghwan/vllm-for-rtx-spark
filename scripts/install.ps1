param(
    [string]$Python = 'python',
    [string]$Environment = '.venv',
    [string]$WheelDirectory = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $WheelDirectory) {
    $WheelDirectory = Join-Path $root 'wheels'
    if (-not (Test-Path -LiteralPath $WheelDirectory)) { $WheelDirectory = Join-Path $root 'dist' }
}
function Assert-NativePython([string]$Executable) {
    $identity = & $Executable -c 'import json,platform,sys; print(json.dumps([sys.platform,platform.machine().lower(),list(sys.version_info[:2])]))'
    if ($LASTEXITCODE) { throw 'Cannot run Python. Supply -Python with a Python 3.13 ARM64 executable.' }
    $identity = $identity | ConvertFrom-Json
    if ($identity[0] -ne 'win32' -or $identity[1] -notin @('arm64','aarch64') -or ($identity[2] -join '.') -ne '3.13') {
        throw 'This release requires native Windows ARM64 Python 3.13.'
    }
}
Assert-NativePython $Python
$wheelRoot = (Resolve-Path -LiteralPath $WheelDirectory).Path
$checksums = Get-Content (Join-Path $wheelRoot 'wheel-sha256.json') -Raw | ConvertFrom-Json
$wheelPaths = @()
foreach ($property in $checksums.PSObject.Properties) {
    $name = $property.Name
    if ([IO.Path]::GetFileName($name) -ne $name -or -not $name.EndsWith('.whl')) { throw 'Invalid wheel manifest.' }
    $path = Join-Path $wheelRoot $name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) {
        throw "Wheel checksum mismatch: $name"
    }
    $wheelPaths += $path
}
if ($wheelPaths.Count -ne 1 -or [IO.Path]::GetFileName($wheelPaths[0]) -notlike 'vllm-*-win_arm64.whl') {
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
    -c (Join-Path $root 'requirements/constraints.txt') @wheelPaths
if ($LASTEXITCODE) { throw 'Package installation failed.' }
$legacyInstalled = & $envPython -c 'import importlib.metadata as m; print(any(d.metadata["Name"] == "vllm-rtx-spark" for d in m.distributions()))'
if ($LASTEXITCODE) { throw 'Cannot inspect the virtual environment.' }
if ($legacyInstalled -eq 'True') {
    & $envPython -m pip uninstall -y vllm-rtx-spark
    if ($LASTEXITCODE) { throw 'Previous companion package removal failed.' }
}
& $envPython -m pip install --force-reinstall --no-deps @wheelPaths
if ($LASTEXITCODE) { throw 'vLLM wheel installation failed.' }
& $envPython -m vllm._spark.runtime.check doctor
if ($LASTEXITCODE) { throw 'Installation checks failed.' }
Write-Host "Ready: $Environment\Scripts\vllm.exe"
