<#
.SYNOPSIS
    Frozen-CLIP SpLiCE concept-ablation sanity check (reproduces the SpLiCE
    appendix Waterbirds intervention: worst-group accuracy ~0.48 -> ~0.60).

.DESCRIPTION
    Trains an L1 logistic-regression probe on frozen SpLiCE concept codes, then
    zeroes the background-concept coefficients and re-measures worst-group
    accuracy. No SSL training is involved, so this runs in minutes on CPU/GPU.

    The intervention concept list is fixed to the SpLiCE paper's Waterbirds set
    resolved in the bundled 10k LAION vocab:
        bamboo, forest, forests, hiking, rainforest
    This is deliberately NOT the automatic discovery list (which omits
    'forest'/'hiking' and adds the noise concepts 'whale'/'raven').

.EXAMPLE
    .\scripts\Run-SpliceSanityCheck.ps1 -DataFolder "D:\Datasets\waterbirds"

.EXAMPLE
    # Match the paper number exactly by evaluating on the test split.
    .\scripts\Run-SpliceSanityCheck.ps1 -DataFolder "D:\Datasets\waterbirds" -FinalTest
#>
[CmdletBinding()]
param(
    [ValidateSet("waterbirds", "spur_cifar10", "celeba")]
    [string]$Dataset = "waterbirds",
    [string]$DataFolder = ".\datasets",
    [string]$PythonExe = "",
    # Paper-faithful Waterbirds background concepts (see .DESCRIPTION).
    [string]$Concepts = "bamboo,forest,forests,hiking,rainforest",
    [double]$L1Penalty = 0.25,
    [double]$ProbeC = 1.0,
    [ValidateSet("train", "ds_train", "val")]
    [string]$TrainSplit = "train",
    [int]$Seed = 0,
    [int]$BatchSize = 128,
    [int]$NumWorkers = 2,
    [int]$GpuIndex = 0,
    # Development default is the validation split; -FinalTest matches the paper's
    # reported test-split number.
    [switch]$FinalTest
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

if (-not (Test-Path -LiteralPath $dataPath)) {
    throw "Data folder not found: $dataPath. Point -DataFolder at the directory containing the dataset."
}

$outputsDir = Join-Path $projectRoot "outputs"
New-Item -ItemType Directory -Force -Path $outputsDir | Out-Null
$resultPath = Join-Path $outputsDir "${Dataset}_splice_cbm_sanity.json"

$env:CUDA_VISIBLE_DEVICES = [string]$GpuIndex

$arguments = @(
    "splice_cbm.py",
    "--dataset", $Dataset,
    "--data_folder", $dataPath,
    "--train_split", $TrainSplit,
    "--intervention_concepts", $Concepts,
    "--splice_l1_penalty", ($L1Penalty.ToString([System.Globalization.CultureInfo]::InvariantCulture)),
    "--probe_c", ($ProbeC.ToString([System.Globalization.CultureInfo]::InvariantCulture)),
    "--seed", [string]$Seed,
    "--batch_size", [string]$BatchSize,
    "--num_workers", [string]$NumWorkers,
    "--out_path", $resultPath
)
if ($FinalTest) {
    $arguments += "--final_test"
    $evalSplitLabel = "test"
} else {
    $evalSplitLabel = "val"
}

Write-Host ""
Write-Host "=== SpLiCE frozen-CLIP sanity check ===" -ForegroundColor Cyan
Write-Host "Dataset      : $Dataset"
Write-Host "Concepts     : $Concepts"
Write-Host "Train / Eval : $TrainSplit / $evalSplitLabel"
Write-Host "l1_penalty=$L1Penalty  probe_C=$ProbeC  seed=$Seed"
Write-Host "Python       : $python"
Write-Host ""

Push-Location $projectRoot
try {
    & $python @arguments
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($exitCode -ne 0) {
    throw "splice_cbm.py exited with code $exitCode."
}

if (-not (Test-Path -LiteralPath $resultPath)) {
    throw "Expected result file was not written: $resultPath"
}

$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
$baselineWg = [math]::Round($result.baseline.worst_group_accuracy * 100, 2)
$baselineAvg = [math]::Round($result.baseline.average_accuracy * 100, 2)
$probeWg = [math]::Round($result.probe_intervention.worst_group_accuracy * 100, 2)
$probeAvg = [math]::Round($result.probe_intervention.average_accuracy * 100, 2)
$reprWg = [math]::Round($result.representation_intervention.worst_group_accuracy * 100, 2)
$delta = [math]::Round($probeWg - $baselineWg, 2)

Write-Host ""
Write-Host "================ SANITY-CHECK SUMMARY ================" -ForegroundColor Green
Write-Host ("Baseline (no ablation)          WG={0,6:N2}%  Avg={1,6:N2}%" -f $baselineWg, $baselineAvg)
Write-Host ("Probe-coefficient ablation      WG={0,6:N2}%  Avg={1,6:N2}%   <-- paper's intervention" -f $probeWg, $probeAvg)
Write-Host ("Representation ablation          WG={0,6:N2}%" -f $reprWg)
Write-Host ("Worst-group change (probe)      {0:+0.00;-0.00}%" -f $delta) -ForegroundColor Yellow
Write-Host "====================================================="
Write-Host "Paper reference (Waterbirds, test): WG 48% -> 60%."
Write-Host "A clear positive WG change confirms the pipeline reproduces the effect."
Write-Host "Full JSON: $resultPath"
