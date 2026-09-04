[CmdletBinding()]
param(
    [ValidateSet("waterbirds", "celeba")]
    [string]$Dataset = "waterbirds",
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$CachePath = "",
    [string]$SpatialRoot = "",
    [string]$OutputRoot = "",
    [string[]]$VariantRecords = @(
        "current_t070_c020|0.01|0.95|0.70|0.20|none|0.25|0.0",
        "semantic_t070|0.01|0.95|0.70|0.00|none|0.25|0.0",
        "semantic_t065|0.01|0.95|0.65|0.00|none|0.25|0.0"
    ),
    [ValidateRange(0.5, 1.0)]
    [double]$FidelityThreshold = 0.95,
    [ValidateRange(0.5, 1.0)]
    [double]$TargetCoverage = 0.99,
    [ValidateRange(32, 10000)]
    [int]$MiniSamples = 1024,
    [ValidateRange(1, 128)]
    [int]$MiniMaxGroups = 24,
    [ValidateRange(1, 128)]
    [int]$MiniNullTrials = 4,
    [switch]$SkipMiniAudit,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DriveDataRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "Datasets"
    $DataRoot = if (Test-Path -LiteralPath $DriveDataRoot) { $DriveDataRoot } else { Join-Path $RepoRoot "datasets" }
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\crpv4_group_screen"
}
if ([string]::IsNullOrWhiteSpace($SpatialRoot)) {
    $SpatialRoot = Join-Path $RepoRoot "outputs\windows_crpv4_ablation\artifacts"
}
if ([string]::IsNullOrWhiteSpace($CachePath)) {
    $Candidates = @(
        (Join-Path $RepoRoot "outputs\crp\${Dataset}_train_features.pt"),
        (Join-Path $RepoRoot "outputs\windows_splice_only_ablation\${Dataset}_train_features_oi_v7_no_dino.pt")
    )
    $CachePath = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($CachePath)) {
        if ($ValidateOnly) {
            $CachePath = $Candidates[0]
        }
        else {
            throw "No CRP cache was found. Pass -CachePath explicitly."
        }
    }
}

$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$CachePath = [System.IO.Path]::GetFullPath($CachePath)
$SpatialRoot = [System.IO.Path]::GetFullPath($SpatialRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

function Find-CondaExecutable {
    $Command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $Command) { return $Command.Source }
    foreach ($Candidate in @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )) {
        if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    }
    throw "conda.exe was not found. Add Conda to PATH or launch from an Anaconda terminal."
}

if ($ValidateOnly) {
    Write-Host "Repository: $RepoRoot"
    Write-Host "Dataset:    $Dataset"
    Write-Host "Data:       $DataRoot"
    Write-Host "Cache:      $CachePath"
    Write-Host "Spatial:    $SpatialRoot"
    Write-Host "Output:     $OutputRoot"
    Write-Host "Variants:   $($VariantRecords -join ', ')"
    return
}

$CondaExe = Find-CondaExecutable
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Push-Location $RepoRoot
try {
    foreach ($Record in $VariantRecords) {
        $Fields = $Record.Split("|")
        if ($Fields.Count -ne 8) {
            throw "Variant record must contain 8 pipe-separated fields: $Record"
        }
        $Name, $MinFrequency, $MaxFrequency, $TextSimilarity, $Coactivation, $SpatialVariant, $SpatialFloor, $FrequencyPower = $Fields
        $VariantOutput = Join-Path $OutputRoot "$Dataset\$Name"
        New-Item -ItemType Directory -Force -Path $VariantOutput | Out-Null
        $Arguments = @(
            "run", "--no-capture-output", "-n", $CondaEnvironment, "python", "-u", "-m", "splice.crp_group_screen",
            "--dataset", $Dataset,
            "--data-folder", $DataRoot,
            "--cache", $CachePath,
            "--output-json", (Join-Path $VariantOutput "group_screen.json"),
            "--output-html", (Join-Path $VariantOutput "group_screen.html"),
            "--min-concept-frequency", $MinFrequency,
            "--max-concept-frequency", $MaxFrequency,
            "--text-similarity-threshold", $TextSimilarity,
            "--coactivation-threshold", $Coactivation,
            "--spatial-floor", $SpatialFloor,
            "--spatial-frequency-power", $FrequencyPower,
            "--fidelity-threshold", $FidelityThreshold.ToString($Invariant),
            "--target-image-coverage", $TargetCoverage.ToString($Invariant),
            "--mini-audit", $(if ($SkipMiniAudit) { "false" } else { "true" }),
            "--mini-samples", $MiniSamples.ToString($Invariant),
            "--mini-max-groups", $MiniMaxGroups.ToString($Invariant),
            "--mini-null-trials", $MiniNullTrials.ToString($Invariant)
        )
        if ($SpatialVariant -ne "none") {
            $VariantDirectory = Join-Path $SpatialRoot $SpatialVariant
            $BalancePath = Get-ChildItem -LiteralPath $VariantDirectory -Filter "balance*.pt" -File | Select-Object -First 1
            if ($null -eq $BalancePath) {
                throw "No balance artifact was found under $VariantDirectory"
            }
            $Arguments += @("--spatial-variant", $SpatialVariant, "--spatial-balance-artifact", $BalancePath.FullName)
        }
        Write-Host "[CRPv4 group screen] $Name"
        & $CondaExe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Group screen $Name failed with exit code $LASTEXITCODE."
        }
        Write-Host "Report: $(Join-Path $VariantOutput 'group_screen.html')"
    }
}
finally {
    Pop-Location
}
