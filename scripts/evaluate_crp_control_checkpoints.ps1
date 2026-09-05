[CmdletBinding()]
param(
    [string]$Config = "$PSScriptRoot/run_crp_controls.conf",
    [int[]]$Epochs = @(50),
    [int[]]$Seeds,
    [string[]]$Arms,
    [string]$CondaEnvironment = "grgrie-train",
    [switch]$ValidateOnly,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $env:USERPROFILE "miniconda3/envs/$CondaEnvironment/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python was not found at $Python" }
$Config = (Resolve-Path -LiteralPath $Config).Path
Push-Location $RepoRoot
try {
    $Arguments = @("-u", "-m", "scripts.tools.evaluate_crp_control_checkpoints", "--config", $Config, "--epochs")
    $Arguments += @($Epochs | ForEach-Object { [string]$_ })
    if ($Seeds) { $Arguments += @("--seeds") + @($Seeds | ForEach-Object { [string]$_ }) }
    if ($Arms) { $Arguments += @("--arms") + $Arms }
    if ($ValidateOnly) { $Arguments += "--validate-only" }
    if ($Force) { $Arguments += "--force" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint evaluation stopped with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
