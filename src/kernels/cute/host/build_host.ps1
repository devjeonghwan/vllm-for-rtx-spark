[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Request
)
$ErrorActionPreference = 'Stop'
if ([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne 'Arm64') {
    throw 'Use native ARM64 PowerShell to link the inference DLL.'
}
$settings = Get-Content -LiteralPath $Request -Raw | ConvertFrom-Json -AsHashtable
$destination = [IO.Path]::GetFullPath($settings.output)
if (Test-Path -LiteralPath $destination) { throw 'Host build output must be new.' }
New-Item -ItemType Directory -Path $destination | Out-Null
$symbols = @()
$objects = foreach ($object in $settings.objects) {
    $resolved = (Resolve-Path -LiteralPath $object).Path
    $bytes = [IO.File]::ReadAllBytes($resolved)
    if ([BitConverter]::ToUInt16($bytes, 0) -ne 0xaa64) { throw "Expected ARM64 COFF: $resolved" }
    foreach ($line in (& $settings.llvm_nm --defined-only --extern-only $resolved)) {
        if ($line -match '^\S+\s+T\s+((?:_mlir_|original_|b12x_)\S+)$') { $symbols += $Matches[1] }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Host symbol inventory failed.' }
    $resolved
}
$symbols = @($symbols | Sort-Object -Unique)
if ($symbols.Count -eq 0) { throw 'No exported host functions found.' }
$definition = Join-Path $destination 'exports.def'
@('EXPORTS') + $symbols | Set-Content -LiteralPath $definition -Encoding ascii
$before = @{}
Get-ChildItem Env: | ForEach-Object { $before[$_.Name] = $_.Value }
try {
    $vs = $settings.visual_studio
    Import-Module (Join-Path $vs 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=arm64 -host_arch=arm64' | Out-Null
    $compiler = (Get-Command cl.exe -ErrorAction Stop).Source
    $linker = (Get-Command link.exe -ErrorAction Stop).Source
    $shim = Join-Path $PSScriptRoot 'cuda_runtime_shim.cpp'
    $shimObject = Join-Path $destination 'runtime.obj'
    & $compiler /nologo /c /O2 /std:c++17 /EHsc /MD ('/I' + (Join-Path $settings.cuda_root 'include')) ('/Fo' + $shimObject) $shim
    if ($LASTEXITCODE -ne 0) { throw 'CUDA host wrapper compilation failed.' }
    $library = Join-Path $destination ($settings.name + '.dll')
    $linkArgs = @('/nologo','/DLL','/MACHINE:ARM64',('/OUT:'+$library),('/DEF:'+$definition),$shimObject) + @($objects) + @((Join-Path $settings.cuda_root 'lib/arm64/cudart.lib'))
    & $linker @linkArgs
    if ($LASTEXITCODE -ne 0) { throw 'ARM64 host library link failed.' }
    @{library=$library;sha256=(Get-FileHash -LiteralPath $library).Hash.ToLowerInvariant();cpu='win_arm64'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destination 'result.json') -Encoding utf8
} finally {
    foreach ($entry in @(Get-ChildItem Env:)) {
        if (-not $before.ContainsKey($entry.Name)) { Remove-Item -LiteralPath ('Env:'+$entry.Name) }
    }
    foreach ($name in $before.Keys) { Set-Item -LiteralPath ('Env:'+$name) -Value $before[$name] }
}
