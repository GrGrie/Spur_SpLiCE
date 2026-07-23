[CmdletBinding()]
param(
    [ValidateSet("baseline", "localized", "both")]
    [string]$Variant = "both",
    [ValidateSet("waterbirds", "celeba", "spur_cifar10")]
    [string]$Dataset = "waterbirds",
    [Parameter(Mandatory = $true)]
    [string]$DataFolder,
    [int[]]$Seeds = @(0),
    [int]$BatchSize = 256,
    [int]$NumWorkers = 5,
    [string]$PythonExe = "",
    [switch]$NoWandb,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "HomeTraining.Common.ps1")

$projectRoot = Get-ProjectRoot -WindowsScriptsDirectory $PSScriptRoot
$python = Get-TrainingPython -ProjectRoot $projectRoot -PythonExe $PythonExe
$dataPath = if ([System.IO.Path]::IsPathRooted($DataFolder)) {
    $DataFolder
} else {
    Join-Path $projectRoot $DataFolder
}

foreach ($seed in $Seeds) {
    $arguments = @(
        "scripts\run_layer_localized_500.py",
        "--variant", $Variant,
        "--dataset", $Dataset,
        "--data_folder", $dataPath,
        "--seed", [string]$seed,
        "--batch_size", [string]$BatchSize,
        "--num_workers", [string]$NumWorkers,
        "--python", $python
    )
    if ($NoWandb) { $arguments += "--no-use_wandb" }
    if ($DryRun) { $arguments += "--dry_run" }

    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Layer-localized launcher failed for seed $seed with exit code $LASTEXITCODE."
    }
}
