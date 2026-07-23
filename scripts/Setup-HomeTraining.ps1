[CmdletBinding()]
param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.11",
    [string]$VenvDirectory = ".venv",
    [switch]$SkipDependencies
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "HomeTraining.Common.ps1")

$projectRoot = Get-ProjectRoot -WindowsScriptsDirectory $PSScriptRoot
$nvidiaSmi = Get-NvidiaSmiPath
Write-Host "[CHECK] NVIDIA driver and GPU"
& $nvidiaSmi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed. Update the NVIDIA driver before creating the environment."
}

$venvPath = if ([System.IO.Path]::IsPathRooted($VenvDirectory)) {
    $VenvDirectory
} else {
    Join-Path $projectRoot $VenvDirectory
}
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "[SETUP] Creating Python $PythonVersion environment at $venvPath"
    & $PythonLauncher "-$PythonVersion" -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment. Install 64-bit Python $PythonVersion and retry."
    }
}

if (-not $SkipDependencies) {
    Write-Host "[SETUP] Updating pip/setuptools/wheel"
    & $venvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Failed to update Python packaging tools." }

    # Current stable PyTorch wheels use CUDA 13 on Blackwell. Installing from
    # PyPI avoids accidentally selecting an old cu118/cu124 wheel without sm_120.
    Write-Host "[SETUP] Installing current stable PyTorch for RTX 50-series"
    & $venvPython -m pip install --upgrade torch torchvision
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch/torchvision." }

    Write-Host "[SETUP] Installing this project and the remaining dependencies"
    & $venvPython -m pip install --editable $projectRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the project dependencies." }
}

Write-Host "[CHECK] PyTorch CUDA support"
$checkCode = @'
import sys
import torch
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in PyTorch")
p = torch.cuda.get_device_properties(0)
print("GPU:", p.name)
print("Compute capability:", f"{p.major}.{p.minor}")
print("VRAM GiB:", round(p.total_memory / 2**30, 2))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("CUDA matrix test:", float(y[0, 0]))
'@
& $venvPython -c $checkCode
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch could not execute the CUDA test. Update the NVIDIA driver and rerun this script."
}

Write-Host ""
Write-Host "Environment is ready: $venvPython"
Write-Host "Next: .\scripts\Test-HomeTraining.ps1 -DataFolder D:\Datasets\waterbirds"
