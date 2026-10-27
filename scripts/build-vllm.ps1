[CmdletBinding()]
param(
    [string]$Python = 'python',
    [ValidateRange(1,8)][int]$Jobs = 2,
    [string]$VisualStudioPath,
    [string]$SourceDirectory,
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne 'Arm64') {
    throw 'Run the core build from native ARM64 PowerShell.'
}
if (-not $VisualStudioPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio Installer was not found.' }
    $VisualStudioPath = & $vswhere -latest -products '*' -property installationPath
}
Import-Module (Join-Path $VisualStudioPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $VisualStudioPath -SkipAutomaticLocation -DevCmdArguments '-arch=arm64 -host_arch=arm64'
$arguments = @((Join-Path $PSScriptRoot 'build_vllm.py'), 'build', '--jobs', $Jobs)
if ($SourceDirectory) { $arguments += @('--source-dir', $SourceDirectory) }
if ($OutputDirectory) { $arguments += @('--output', $OutputDirectory) }
& $Python @arguments
if ($LASTEXITCODE) { throw 'vLLM source build failed.' }
