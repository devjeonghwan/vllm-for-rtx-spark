param(
    [string]$Python = 'python',
    [string]$Environment = '.venv',
    [string]$WheelDirectory = '',
    [switch]$UseUv
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$manifestPath = Join-Path $root 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    $manifestPath = Join-Path $root 'patches/vllm/manifest.json'
}
$manifestSpec = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifestSpec.platform.os -ne 'windows' -or $manifestSpec.platform.cpu -ne 'arm64') {
    throw 'This release requires native Windows ARM64 installation.'
}
$pythonVersion = $manifestSpec.platform.python
$uvCommand = $null
if (-not $WheelDirectory) {
    $WheelDirectory = Join-Path $root 'wheels'
    if (-not (Test-Path -LiteralPath $WheelDirectory)) { $WheelDirectory = Join-Path $root 'dist' }
}
function Assert-NativePython([string]$Executable, [switch]$RequireEnvironment) {
    if ([string]::IsNullOrWhiteSpace($Executable)) { throw 'Supply a native Windows ARM64 Python executable.' }
    $identity = & $Executable -c 'import json,platform,sys; print(json.dumps([sys.platform,platform.machine().lower(),list(sys.version_info[:2]),sys.prefix != sys.base_prefix]))'
    if ($LASTEXITCODE) { throw "Cannot run Python. Supply -Python with a Python $pythonVersion ARM64 executable." }
    $identity = $identity | ConvertFrom-Json
    if ($identity[0] -ne 'win32' -or $identity[1] -notin @('arm64','aarch64') -or ($identity[2] -join '.') -ne $pythonVersion) {
        throw "This release requires native Windows ARM64 Python $pythonVersion."
    }
    if ($RequireEnvironment -and ($identity.Count -lt 4 -or $identity[3] -ne $true)) {
        throw 'The target Python must belong to a virtual environment.'
    }
}
function Invoke-Uv([string[]]$UvArguments, [string]$FailureMessage) {
    # Release indexes/target and non-destructive venv defaults take precedence
    # over caller uv configuration. Restore process values even on failure.
    $names = @('UV_INDEX', 'UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_EXTRA_INDEX_URL',
        'UV_FIND_LINKS', 'UV_NO_INDEX', 'UV_INDEX_STRATEGY', 'UV_CONFIG_FILE',
        'UV_SYSTEM_PYTHON', 'UV_VENV_CLEAR', 'UV_VENV_SEED')
    $saved = @{}
    foreach ($name in $names) {
        $saved[$name] = @{ Exists = Test-Path -LiteralPath ('Env:' + $name)
            Value = [Environment]::GetEnvironmentVariable($name, 'Process') }
        Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
    }
    try {
        & $uvCommand @UvArguments
        if ($LASTEXITCODE) { throw $FailureMessage }
    } finally {
        foreach ($name in $names) {
            if ($saved[$name].Exists) {
                Set-Item -LiteralPath ('Env:' + $name) -Value $saved[$name].Value
            } else {
                Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
            }
        }
    }
}
$wheelRoot = (Resolve-Path -LiteralPath $WheelDirectory).Path
$wheelPaths = @(Get-ChildItem -LiteralPath $wheelRoot -Filter 'vllm-*-win_arm64.whl' -File | ForEach-Object FullName)
if ($wheelPaths.Count -ne 1) {
    throw 'Expected one Windows ARM64 vLLM wheel.'
}
if ($UseUv -and -not $PSBoundParameters.ContainsKey('Environment') -and
    -not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
    $Environment = $env:VIRTUAL_ENV
}
if ([string]::IsNullOrWhiteSpace($Environment)) { throw 'The target -Environment cannot be empty.' }
if (-not [IO.Path]::IsPathRooted($Environment)) { $Environment = Join-Path $root $Environment }
$Environment = [IO.Path]::GetFullPath($Environment)
$envPython = Join-Path $Environment 'Scripts/python.exe'
if (Test-Path -LiteralPath $envPython -PathType Leaf) {
    Assert-NativePython $envPython -RequireEnvironment
} else {
    if (Test-Path -LiteralPath $Environment) {
        if (-not (Test-Path -LiteralPath $Environment -PathType Container) -or
            @(Get-ChildItem -LiteralPath $Environment -Force).Count) {
            throw 'Existing target environment is incomplete; repair it or choose a new -Environment.'
        }
    }
    if ($UseUv) {
        $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCommand) { throw 'uv was not found. Install uv separately or omit -UseUv.' }
        if (-not $PSBoundParameters.ContainsKey('Python')) {
            $pythonRequest = "cpython-$pythonVersion-windows-aarch64-none"
            $discovered = @(Invoke-Uv -UvArguments @('python', 'find', '--no-project', '--no-config',
                '--no-python-downloads', $pythonRequest) -FailureMessage (
                "Native Python $pythonVersion was not found locally. Run 'uv python install $pythonRequest' or supply -Python with its ARM64 executable."))
            if ($discovered.Count -ne 1) { throw 'uv did not return one Python executable; supply -Python explicitly.' }
            $Python = ([string]$discovered[0]).Trim()
        }
    }
    Assert-NativePython $Python
    if ($UseUv) {
        Invoke-Uv -UvArguments @('venv', '--python', $Python, '--no-project', '--no-config',
            '--no-python-downloads', $Environment) -FailureMessage 'uv virtual environment creation failed.'
    } else {
        & $Python -m venv $Environment
        if ($LASTEXITCODE) { throw 'Virtual environment creation failed.' }
    }
    Assert-NativePython $envPython -RequireEnvironment
}
if ($UseUv) {
    if (-not $uvCommand) {
        $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCommand) { throw 'uv was not found. Install uv separately or omit -UseUv.' }
    }
    $uvInstall = @('pip', 'install', '--python', $envPython, '--no-config', '--no-python-downloads',
        '--only-binary=:all:', '-c', (Join-Path $root $manifestSpec.requirements.constraints))
    # Pin vendor provenance before resolving ordinary dependencies from PyPI.
    # --no-deps prevents a first-index NVIDIA entry from shadowing PyPI deps.
    Invoke-Uv -UvArguments ($uvInstall + @('--default-index', 'https://pypi.nvidia.com/nvtorch_oot/',
        '--index-strategy', 'first-index', '--no-deps', '-r',
        (Join-Path $root $manifestSpec.requirements.runtime_vendor))) -FailureMessage 'NVIDIA vendor package installation failed.'
    Invoke-Uv -UvArguments ($uvInstall + @('--default-index', 'https://pypi.org/simple',
        '--index-strategy', 'first-index') + $wheelPaths) -FailureMessage 'Package installation failed.'
} else {
    & $envPython -m pip --version
    if ($LASTEXITCODE) { throw 'pip is missing from the target environment. Use -UseUv or create an environment with pip.' }
    & $envPython -m pip install --only-binary=:all: --extra-index-url https://pypi.nvidia.com/nvtorch_oot/ `
        -c (Join-Path $root $manifestSpec.requirements.constraints) @wheelPaths
    if ($LASTEXITCODE) { throw 'Package installation failed.' }
}
$legacyInstalled = & $envPython -c "import importlib.metadata as m; print(any(d.metadata['Name'] == 'vllm-rtx-spark' for d in m.distributions()))"
if ($LASTEXITCODE) { throw 'Cannot inspect the virtual environment.' }
if ($legacyInstalled -eq 'True') {
    if ($UseUv) {
        Invoke-Uv -UvArguments @('pip', 'uninstall', '--python', $envPython, '--no-config',
            '--no-python-downloads', 'vllm-rtx-spark') -FailureMessage 'Previous companion package removal failed.'
    } else {
        & $envPython -m pip uninstall -y vllm-rtx-spark
        if ($LASTEXITCODE) { throw 'Previous companion package removal failed.' }
    }
}
if ($UseUv) {
    Invoke-Uv -UvArguments ($uvInstall + @('--reinstall', '--no-deps', '--no-index') + $wheelPaths) `
        -FailureMessage 'vLLM wheel installation failed.'
} else {
    & $envPython -m pip install --force-reinstall --no-deps @wheelPaths
    if ($LASTEXITCODE) { throw 'vLLM wheel installation failed.' }
}
& $envPython -m vllm._rtx_spark.runtime.check doctor
if ($LASTEXITCODE) { throw 'Installation checks failed.' }
Write-Host "Ready: $Environment\Scripts\vllm.exe"
