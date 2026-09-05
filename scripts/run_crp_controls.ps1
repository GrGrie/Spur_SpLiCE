[CmdletBinding()]
param(
    [string]$Config = "$PSScriptRoot/run_crp_controls.conf",
    [string]$CondaEnvironment = "grgrie-train",
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $env:USERPROFILE "miniconda3/envs/$CondaEnvironment/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python was not found at $Python" }
$Config = (Resolve-Path -LiteralPath $Config).Path
Push-Location $RepoRoot
try {
    $Arguments = @("-u", "-m", "scripts.tools.run_crp_controls", "--config", $Config)
    if ($ValidateOnly) { $Arguments += "--validate-only" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Control experiment stopped with exit code $LASTEXITCODE. See its log." }
}
finally { Pop-Location }
