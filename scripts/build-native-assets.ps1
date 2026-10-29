[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$OutputDirectory = '',
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$Components = 'ple,depth,csf,sampling,hc_upmix,tiktoken,llguidance',
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
if ([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne 'Arm64') {
    throw 'Use native ARM64 PowerShell for native host libraries.'
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio C++ ARM64 build tools are required.' }
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
    if (-not $vs) { throw 'Visual Studio ARM64 tools were not found.' }
    Import-Module (Join-Path $vs 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=arm64 -host_arch=arm64' | Out-Null
}
if (-not $CudaRoot) { throw 'Set CUDA_PATH or pass -CudaRoot.' }
$arguments = @((Join-Path $PSScriptRoot 'build_native_assets.py'), '--cuda-root', $CudaRoot, '--components', $Components)
if ($OutputDirectory) { $arguments += @('--output', $OutputDirectory) }
if ($Install) { $arguments += '--install' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Native build failed with exit code $LASTEXITCODE" }
