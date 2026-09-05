[CmdletBinding()]
param(
    [ValidateSet("waterbirds", "celeba")]
    [string]$Dataset = "waterbirds",
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$CachePath = "",
    [string]$InputRoot = "",
    [string]$OutputRoot = "",
    [Parameter(Mandatory = $true)]
    [string[]]$RunNames,
    [ValidateRange(32, 10000)]
    [int]$MiniSamples = 1024,
    [ValidateRange(1, 128)]
    [int]$MiniMaxGroups = 24,
    [ValidateRange(1, 128)]
    [int]$MiniNullTrials = 4,
    [ValidateRange(0.0, 1.0)]
    [double]$MiniNullQuantile = 0.95,
    [ValidateRange(0.0, 1.0)]
    [double]$MiniMinCoverage = 0.01,
    [double]$MiniMinInterventionGain = 5e-4,
    [ValidateRange(0.0, 1.0)]
    [double]$MiniActivationDifferenceQuantile = 0.85,
    [double]$MiniResidualThreshold = 0.25,
    [int]$Seed = 0,
    [switch]$Force,
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
if ([string]::IsNullOrWhiteSpace($InputRoot)) {
    $InputRoot = Join-Path $RepoRoot "outputs\crpv4_group_protocol_v1"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\crpv4_group_mini_audit"
}
if ([string]::IsNullOrWhiteSpace($CachePath)) {
    $Candidates = @(
        (Join-Path $RepoRoot "outputs\crp\${Dataset}_train_features.pt"),
        (Join-Path $RepoRoot "outputs\windows_splice_only_ablation\${Dataset}_train_features_oi_v7_no_dino.pt")
    )
    $CachePath = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($CachePath)) {
        if ($ValidateOnly) { $CachePath = $Candidates[0] }
        else { throw "No CRP cache was found. Pass -CachePath explicitly." }
    }
}

$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$CachePath = [System.IO.Path]::GetFullPath($CachePath)
$InputRoot = [System.IO.Path]::GetFullPath($InputRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$InputDatasetRoot = Join-Path $InputRoot $Dataset
$OutputDatasetRoot = Join-Path $OutputRoot $Dataset
$SummaryPath = Join-Path $OutputDatasetRoot "mini_audit_summary.csv"

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

function Expand-RunNames([string[]]$Values) {
    $Expanded = @()
    foreach ($Value in $Values) {
        if ([string]::IsNullOrWhiteSpace($Value)) { continue }
        $Expanded += $Value.Split(",") |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }
    return @($Expanded | Select-Object -Unique)
}

function Find-BalanceArtifact([string]$Variant) {
    $SpatialRoot = Join-Path $RepoRoot "outputs\windows_crpv4_ablation\artifacts"
    $VariantDirectory = Join-Path $SpatialRoot $Variant
    $Artifact = Get-ChildItem -LiteralPath $VariantDirectory -Filter "balance*.pt" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $Artifact) {
        throw "No balance artifact was found under $VariantDirectory"
    }
    return $Artifact.FullName
}

function Read-MiniSummary([string]$JsonPath, [string]$RunName, [string]$Status, [string]$ErrorMessage = "") {
    if (-not (Test-Path -LiteralPath $JsonPath)) {
        return [pscustomobject]@{
            run_name = $RunName
            status = $Status
            decision = ""
            source_coverage = ""
            median_source_fidelity = ""
            p01_source_fidelity = ""
            candidate_groups = ""
            selected_groups = ""
            selected_concepts = ""
            audited_groups = ""
            requested_groups = ""
            null_pass_fraction = ""
            median_top1_turnover = ""
            median_neighbor_turnover = ""
            median_jaccard_at_k = ""
            geometry_changed = ""
            mini_passed = ""
            reason = $ErrorMessage
            json = $JsonPath
            html = [System.IO.Path]::ChangeExtension($JsonPath, ".html")
        }
    }

    $Report = Get-Content -Raw -LiteralPath $JsonPath | ConvertFrom-Json
    $Metrics = $Report.metrics
    $Mini = $Report.mini_intervention
    return [pscustomobject]@{
        run_name = $RunName
        status = $Status
        decision = [string]$Report.decision.status
        source_coverage = $Metrics.source_coverage
        median_source_fidelity = $Metrics.median_source_fidelity
        p01_source_fidelity = $Metrics.p01_source_fidelity
        candidate_groups = $Metrics.candidate_group_count
        selected_groups = $Metrics.selected_group_count
        selected_concepts = $Metrics.selected_concept_count
        audited_groups = if ($null -ne $Mini) { $Mini.audited_group_count } else { "" }
        requested_groups = if ($null -ne $Mini) { $Mini.requested_group_count } else { "" }
        null_pass_fraction = if ($null -ne $Mini) { $Mini.null_pass_fraction } else { "" }
        median_top1_turnover = if ($null -ne $Mini) { $Mini.median_top1_neighbor_turnover } else { "" }
        median_neighbor_turnover = ""
        median_jaccard_at_k = if ($null -ne $Mini) { $Mini.median_jaccard_at_k } else { "" }
        geometry_changed = if ($null -ne $Mini) { $Mini.geometry_changed } else { "" }
        mini_passed = if ($null -ne $Mini) { $Mini.passed } else { "" }
        reason = [string]$Report.decision.reason
        json = $JsonPath
        html = [System.IO.Path]::ChangeExtension($JsonPath, ".html")
    }
}

$Names = @(Expand-RunNames $RunNames)
if ($Names.Count -eq 0) {
    throw "At least one run name is required through -RunNames."
}

$Jobs = @()
foreach ($Name in $Names) {
    $InputJson = Join-Path (Join-Path $InputDatasetRoot $Name) "group_screen.json"
    if (-not (Test-Path -LiteralPath $InputJson)) {
        throw "Could not find report for '$Name': $InputJson"
    }
    $Report = Get-Content -Raw -LiteralPath $InputJson | ConvertFrom-Json
    $GroupConfig = $Report.group_config
    $ScreenConfig = $Report.screen_config
    $Jobs += [pscustomobject]@{
        Name = $Name
        InputJson = $InputJson
        MinFrequency = [double]$GroupConfig.min_concept_frequency
        MaxFrequency = [double]$GroupConfig.max_concept_frequency
        TextSimilarity = [double]$GroupConfig.text_similarity_threshold
        Coactivation = [double]$GroupConfig.coactivation_threshold
        SpatialBalance = [bool]$GroupConfig.spatial_balance
        SpatialVariant = [string]$GroupConfig.spatial_balance_variant
        SpatialFloor = [double]$GroupConfig.spatial_balance_floor
        FrequencyPower = [double]$GroupConfig.spatial_frequency_power
        FidelityThreshold = [double]$ScreenConfig.fidelity_threshold
        TargetCoverage = [double]$ScreenConfig.target_image_coverage
        CoherenceWarning = [double]$ScreenConfig.coherence_warning_threshold
        OriginalSeed = [int]$GroupConfig.seed
    }
}

Write-Host "Input root:       $InputDatasetRoot"
Write-Host "Output root:      $OutputDatasetRoot"
Write-Host "Runs:             $($Jobs.Count)"
Write-Host "Mini samples:     $MiniSamples"
Write-Host "Mini max groups:  $MiniMaxGroups"
Write-Host "Mini null trials: $MiniNullTrials"

if ($ValidateOnly) {
    Write-Host "Validation only; no audits were started."
    $Jobs | Select-Object Name, MinFrequency, MaxFrequency, TextSimilarity, Coactivation, SpatialBalance | Format-Table -AutoSize
    return
}

$CondaExe = Find-CondaExecutable
New-Item -ItemType Directory -Force -Path $OutputDatasetRoot | Out-Null
Push-Location $RepoRoot
try {
    $Rows = @()
    $Index = 0
    foreach ($Job in $Jobs) {
        $Index++
        $RunOutput = Join-Path $OutputDatasetRoot $Job.Name
        $JsonPath = Join-Path $RunOutput "group_screen.json"
        $HtmlPath = Join-Path $RunOutput "group_screen.html"

        if ((-not $Force) -and (Test-Path -LiteralPath $JsonPath)) {
            Write-Host "[SKIP] [$Index/$($Jobs.Count)] $($Job.Name) (report exists)"
            $Rows += Read-MiniSummary $JsonPath $Job.Name "reused"
            continue
        }

        New-Item -ItemType Directory -Force -Path $RunOutput | Out-Null
        $Arguments = @(
            "run", "--no-capture-output", "-n", $CondaEnvironment, "python", "-u", "-m", "splice.crp_group_screen",
            "--dataset", $Dataset,
            "--data-folder", $DataRoot,
            "--cache", $CachePath,
            "--output-json", $JsonPath,
            "--output-html", $HtmlPath,
            "--min-concept-frequency", $Job.MinFrequency.ToString($Invariant),
            "--max-concept-frequency", $Job.MaxFrequency.ToString($Invariant),
            "--text-similarity-threshold", $Job.TextSimilarity.ToString($Invariant),
            "--coactivation-threshold", $Job.Coactivation.ToString($Invariant),
            "--fidelity-threshold", $Job.FidelityThreshold.ToString($Invariant),
            "--target-image-coverage", $Job.TargetCoverage.ToString($Invariant),
            "--coherence-warning-threshold", $Job.CoherenceWarning.ToString($Invariant),
            "--mini-audit", "true",
            "--mini-samples", $MiniSamples.ToString($Invariant),
            "--mini-max-groups", $MiniMaxGroups.ToString($Invariant),
            "--mini-null-trials", $MiniNullTrials.ToString($Invariant),
            "--mini-null-quantile", $MiniNullQuantile.ToString($Invariant),
            "--mini-min-coverage", $MiniMinCoverage.ToString($Invariant),
            "--mini-min-intervention-gain", $MiniMinInterventionGain.ToString($Invariant),
            "--mini-activation-difference-quantile", $MiniActivationDifferenceQuantile.ToString($Invariant),
            "--mini-residual-threshold", $MiniResidualThreshold.ToString($Invariant),
            "--seed", $(if ($Seed -ne 0) { $Seed } else { $Job.OriginalSeed })
        )
        if ($Job.SpatialBalance) {
            $BalancePath = Find-BalanceArtifact $Job.SpatialVariant
            $Arguments += @(
                "--spatial-variant", $Job.SpatialVariant,
                "--spatial-floor", $Job.SpatialFloor.ToString($Invariant),
                "--spatial-frequency-power", $Job.FrequencyPower.ToString($Invariant),
                "--spatial-balance-artifact", $BalancePath
            )
        }

        Write-Host "[$Index/$($Jobs.Count)] $($Job.Name)"
        & $CondaExe @Arguments
        $ExitCode = $LASTEXITCODE
        if ($ExitCode -ne 0) {
            $Message = "group screen exited with code $ExitCode"
            Write-Warning "$($Job.Name): $Message"
            $Rows += Read-MiniSummary $JsonPath $Job.Name "failed" $Message
        }
        else {
            $Rows += Read-MiniSummary $JsonPath $Job.Name "completed"
        }
    }
}
finally {
    Pop-Location
}

if ($Rows.Count -gt 0) {
    $Rows |
        Sort-Object @{ Expression = { [double]($_.null_pass_fraction -as [double]) }; Descending = $true },
                    @{ Expression = { [double]($_.median_top1_turnover -as [double]) }; Descending = $true } |
        Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $SummaryPath
}

Write-Host ""
Write-Host "Top mini-audit results:"
$Rows |
    Sort-Object @{ Expression = { [double]($_.null_pass_fraction -as [double]) }; Descending = $true },
                @{ Expression = { [double]($_.median_top1_turnover -as [double]) }; Descending = $true } |
    Select-Object -First 20 run_name, decision, source_coverage, median_source_fidelity, null_pass_fraction, median_top1_turnover, median_jaccard_at_k, geometry_changed, mini_passed |
    Format-Table -AutoSize

Write-Host "Summary: $SummaryPath"
