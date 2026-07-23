[CmdletBinding()]
param(
    [string]$DataFolder = ".\datasets",
    [string]$PythonExe = "",
    [int]$GpuIndex = 0
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

$arguments = @(
    "spur_splice.py",
    "--dataset", "waterbirds",
    "--data_folder", $dataPath,
    "--model", "resnet18_large",
    "--epochs", "1",
    "--batch_size", "256",
    "--num_workers", "0",
    "--learning_rate", "0.01",
    "--temp", "0.05",
    "--linear_probe_mode", "none",
    "--rank_eval_freq", "0",
    "--print_freq", "1",
    "--amp", "true",
    "--channels_last", "true",
    "--cudnn_enabled", "true",
    "--splice_mode", "none",
    "--checkpoint_dir", (Join-Path $projectRoot "outputs\home_smoke_checkpoints"),
    "--keep_checkpoints",
    "--save_freq", "1",
    "--wandb_tags", "machine_home,smoke_test"
)

Invoke-ManagedPython `
    -PythonExe $python `
    -PythonArguments $arguments `
    -WorkingDirectory $projectRoot `
    -RunLabel "waterbirds_smoke_test" `
    -LogDirectory (Join-Path $projectRoot "outputs\home_logs") `
    -GpuIndex $GpuIndex `
    -CpuThreads 2 | Out-Null
Write-Host "Smoke test passed. The dataset, CUDA, AMP, and checkpoint path are usable."
