[CmdletBinding()]
param(
    [ValidateSet("waterbirds", "celeba")]
    [string]$Dataset = "waterbirds",
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$CachePath = "",
    [string]$OutputRoot = "",
    [double[]]$MinConceptFrequencies = @(0.005, 0.01, 0.02),
    [double[]]$MaxConceptFrequencies = @(0.85, 0.95, 1.00),
    [double[]]$TextSimilarityThresholds = @(0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
    [double[]]$CoactivationThresholds = @(0.00, 0.10, 0.20, 0.30, 0.40, 0.50),
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
    [switch]$RunMiniAudit,
    [ValidateRange(0, 100000)]
    [int]$MaxRuns = 0,
    [switch]$Force,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DriveDataRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "Datasets"
    $DataRoot = if (Test-Path -LiteralPath $DriveDataRoot) {
        $DriveDataRoot
    }
    else {
        Join-Path $RepoRoot "datasets"
    }
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\crpv4_group_sweep"
}
if ([string]::IsNullOrWhiteSpace($CachePath)) {
    $Candidates = @(
        (Join-Path $RepoRoot "outputs\crp\${Dataset}_train_features.pt"),
        (Join-Path $RepoRoot "outputs\windows_splice_only_ablation\${Dataset}_train_features_oi_v7_no_dino.pt")
    )
    $CachePath = $Candidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
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
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$DatasetOutputRoot = Join-Path $OutputRoot $Dataset
$SummaryPath = Join-Path $DatasetOutputRoot "sweep_summary.csv"

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

function Format-RunValue([double]$Value) {
    return $Value.ToString("0.##########", $Invariant).Replace("-", "m").Replace(".", "p")
}

function Read-ReportSummary([string]$JsonPath, [string]$RunName, [string]$Status, [string]$ErrorMessage = "") {
    if (-not (Test-Path -LiteralPath $JsonPath)) {
        return [pscustomobject]@{
            run_name = $RunName
            status = $Status
            decision = ""
            reason = $ErrorMessage
            min_frequency = ""
            max_frequency = ""
            text_similarity = ""
            coactivation = ""
            candidate_groups = ""
            selected_groups = ""
            selected_concepts = ""
            source_coverage = ""
            median_source_fidelity = ""
            p01_source_fidelity = ""
            coherence_warnings = ""
            json = $JsonPath
            html = [System.IO.Path]::ChangeExtension($JsonPath, ".html")
        }
    }

    $Report = Get-Content -Raw -LiteralPath $JsonPath | ConvertFrom-Json
    $GroupConfig = $Report.group_config
    $Metrics = $Report.metrics
    return [pscustomobject]@{
        run_name = $RunName
        status = $Status
        decision = [string]$Report.decision.status
        reason = [string]$Report.decision.reason
        min_frequency = $GroupConfig.min_concept_frequency
        max_frequency = $GroupConfig.max_concept_frequency
        text_similarity = $GroupConfig.text_similarity_threshold
        coactivation = $GroupConfig.coactivation_threshold
        candidate_groups = $Metrics.candidate_group_count
        selected_groups = $Metrics.selected_group_count
        selected_concepts = $Metrics.selected_concept_count
        source_coverage = $Metrics.source_coverage
        median_source_fidelity = $Metrics.median_source_fidelity
        p01_source_fidelity = $Metrics.p01_source_fidelity
        coherence_warnings = $Metrics.coherence_warning_count
        json = $JsonPath
        html = [System.IO.Path]::ChangeExtension($JsonPath, ".html")
    }
}

$Combinations = @(
    foreach ($MinFrequency in $MinConceptFrequencies) {
        foreach ($MaxFrequency in $MaxConceptFrequencies) {
            if ($MaxFrequency -lt $MinFrequency) {
                continue
            }
            foreach ($TextSimilarity in $TextSimilarityThresholds) {
                foreach ($Coactivation in $CoactivationThresholds) {
                    $RunName = "min$(Format-RunValue $MinFrequency)_max$(Format-RunValue $MaxFrequency)_t$(Format-RunValue $TextSimilarity)_c$(Format-RunValue $Coactivation)"
                    [pscustomobject]@{
                        Name = $RunName
                        MinFrequency = $MinFrequency
                        MaxFrequency = $MaxFrequency
                        TextSimilarity = $TextSimilarity
                        Coactivation = $Coactivation
                    }
                }
            }
        }
    }
)

if ($MaxRuns -gt 0) {
    $Combinations = @($Combinations | Select-Object -First $MaxRuns)
}

Write-Host "Repository:       $RepoRoot"
Write-Host "Dataset:          $Dataset"
Write-Host "Cache:            $CachePath"
Write-Host "Output:           $DatasetOutputRoot"
Write-Host "Combinations:     $($Combinations.Count)"
Write-Host "Mini intervention: $(if ($RunMiniAudit) { 'enabled' } else { 'disabled' })"

if ($ValidateOnly) {
    Write-Host "Validation only; no runs were started."
    $Combinations | Select-Object Name, MinFrequency, MaxFrequency, TextSimilarity, Coactivation | Format-Table -AutoSize
    return
}

$CondaExe = Find-CondaExecutable
New-Item -ItemType Directory -Force -Path $DatasetOutputRoot | Out-Null
Push-Location $RepoRoot
try {
    $Rows = @()
    $Completed = 0
    foreach ($Combination in $Combinations) {
        $RunOutput = Join-Path $DatasetOutputRoot $Combination.Name
        $JsonPath = Join-Path $RunOutput "group_screen.json"
        $HtmlPath = Join-Path $RunOutput "group_screen.html"

        if ((-not $Force) -and (Test-Path -LiteralPath $JsonPath)) {
            Write-Host "[SKIP] $($Combination.Name) (report exists)"
            $Rows += Read-ReportSummary $JsonPath $Combination.Name "reused"
            $Completed++
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
            "--min-concept-frequency", $Combination.MinFrequency.ToString($Invariant),
            "--max-concept-frequency", $Combination.MaxFrequency.ToString($Invariant),
            "--text-similarity-threshold", $Combination.TextSimilarity.ToString($Invariant),
            "--coactivation-threshold", $Combination.Coactivation.ToString($Invariant),
            "--fidelity-threshold", $FidelityThreshold.ToString($Invariant),
            "--target-image-coverage", $TargetCoverage.ToString($Invariant),
            "--mini-audit", $(if ($RunMiniAudit) { "true" } else { "false" }),
            "--mini-samples", $MiniSamples.ToString($Invariant),
            "--mini-max-groups", $MiniMaxGroups.ToString($Invariant),
            "--mini-null-trials", $MiniNullTrials.ToString($Invariant)
        )

        $Index = $Completed + 1
        Write-Host "[$Index/$($Combinations.Count)] $($Combination.Name)"
        & $CondaExe @Arguments
        $ExitCode = $LASTEXITCODE
        if ($ExitCode -ne 0) {
            $Message = "group screen exited with code $ExitCode"
            Write-Warning "$($Combination.Name): $Message"
            $Rows += Read-ReportSummary $JsonPath $Combination.Name "failed" $Message
        }
        else {
            $Rows += Read-ReportSummary $JsonPath $Combination.Name "completed"
        }
        $Completed++

        $Rows |
            Sort-Object @{ Expression = { [double]($_.source_coverage -as [double]) }; Descending = $true },
                        @{ Expression = { [double]($_.median_source_fidelity -as [double]) }; Descending = $true } |
            Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $SummaryPath
    }

    if ($Rows.Count -gt 0) {
        $Rows |
            Sort-Object @{ Expression = { [double]($_.source_coverage -as [double]) }; Descending = $true },
                        @{ Expression = { [double]($_.median_source_fidelity -as [double]) }; Descending = $true } |
            Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $SummaryPath
    }
}
finally {
    Pop-Location
}

$ValidRows = @($Rows | Where-Object { $_.status -ne "failed" -and $_.decision })
Write-Host ""
Write-Host "Top configurations by source coverage:"
$ValidRows |
    Sort-Object @{ Expression = { [double]$_.source_coverage }; Descending = $true },
                @{ Expression = { [double]$_.median_source_fidelity }; Descending = $true },
                @{ Expression = { [int]$_.selected_groups }; Descending = $false } |
    Select-Object -First 20 run_name, decision, source_coverage, median_source_fidelity, p01_source_fidelity, candidate_groups, selected_groups |
    Format-Table -AutoSize

Write-Host "Summary: $SummaryPath"
