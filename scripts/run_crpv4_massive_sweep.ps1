[CmdletBinding()]
param(
    [ValidateSet("waterbirds", "celeba")]
    [string]$Dataset = "waterbirds",
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$CachePath = "",
    [string]$OutputRoot = "",
    [string]$SpatialRoot = "",

    # Phase 1: broad group-construction grid.
    [double[]]$GroupMinConceptFrequencies = @(0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.005),
    [double[]]$GroupMaxConceptFrequencies = @(0.45, 1.0),
    [double[]]$GroupTextSimilarityThresholds = @(0.60, 0.70, 0.80, 0.85),
    [double[]]$GroupCoactivationThresholds = @(0.00, 0.10, 0.20, 0.30, 0.50),
    [ValidateRange(0.5, 1.0)]
    [double]$ReferenceFidelityThreshold = 0.90,
    [ValidateRange(0.5, 1.0)]
    [double]$ReferenceTargetCoverage = 0.99,

    # Phase 2: refine selected groupings across the reconstruction trade-off.
    [double[]]$RefineFidelityThresholds = @(0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99),
    [double[]]$RefineTargetCoverages = @(0.90, 0.95, 0.99),
    [ValidateRange(1, 100000)]
    [int]$Phase2TopGroupConfigs = 64,

    # Phase 3: three deliberately different mini-audit profiles by default.
    [int[]]$MiniSamplesGrid = @(1024, 1536, 2048),
    [int[]]$MiniMaxGroupsGrid = @(24, 48, 64),
    [int[]]$MiniNullTrialsGrid = @(4, 8, 12),
    [double[]]$MiniNullQuantilesGrid = @(0.90, 0.95, 0.975),
    [double[]]$MiniMinCoveragesGrid = @(0.005, 0.01, 0.02),
    [double[]]$MiniMinInterventionGainsGrid = @(0.0001, 0.0005, 0.001),
    [double[]]$MiniActivationDifferenceQuantilesGrid = @(0.75, 0.85, 0.90),
    [double[]]$MiniResidualThresholdsGrid = @(0.25, 0.30, 0.35),
    [ValidateRange(0.01, 1.0)]
    [double]$Phase3TopFraction = 0.50,

    # Spatial Concept Balancing is an explicit optional axis, not part of the default baseline.
    [string[]]$SpatialVariants = @("none"),
    [double[]]$SpatialFloors = @(0.25),
    [double[]]$SpatialFrequencyPowers = @(0.0),

    [ValidateRange(0.1, 72.0)]
    [double]$MaxHours = 12.0,
    [ValidateRange(0, 100000)]
    [int]$Phase1MaxRuns = 0,
    [ValidateRange(0, 100000)]
    [int]$MaxMiniAudits = 0,
    [int]$Seed = 0,
    [switch]$Force,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Invariant = [Globalization.CultureInfo]::InvariantCulture
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DriveDataRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "Datasets"
    $DataRoot = if (Test-Path -LiteralPath $DriveDataRoot) { $DriveDataRoot } else { Join-Path $RepoRoot "datasets" }
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\crpv4_massive_sweep"
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
        if ($ValidateOnly) { $CachePath = $Candidates[0] }
        else { throw "No CRP cache was found. Pass -CachePath explicitly." }
    }
}

$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$CachePath = [System.IO.Path]::GetFullPath($CachePath)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$SpatialRoot = [System.IO.Path]::GetFullPath($SpatialRoot)
$Phase1Root = Join-Path $OutputRoot "phase1_group_grid\$Dataset"
$Phase2Root = Join-Path $OutputRoot "phase2_fidelity_refinement\$Dataset"
$Phase3Root = Join-Path $OutputRoot "phase3_mini_audit\$Dataset"
$Phase1SummaryPath = Join-Path $Phase1Root "phase1_summary.csv"
$Phase2SummaryPath = Join-Path $Phase2Root "phase2_summary.csv"
$Phase3SummaryPath = Join-Path $Phase3Root "phase3_summary.csv"

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

function Test-TimeBudget {
    if ($Stopwatch.Elapsed.TotalHours -ge $MaxHours) {
        Write-Warning ("Time budget reached: {0:N2} hours. Stopping before the next run." -f $Stopwatch.Elapsed.TotalHours)
        return $false
    }
    return $true
}

function Find-BalanceArtifact([string]$Variant) {
    $VariantDirectory = Join-Path $SpatialRoot $Variant
    $Artifact = Get-ChildItem -LiteralPath $VariantDirectory -Filter "balance*.pt" -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $Artifact) { throw "No balance artifact was found under $VariantDirectory" }
    return $Artifact.FullName
}

function Assert-EqualGridLengths {
    param([object[]]$Grids)
    $Expected = @($Grids[0]).Count
    foreach ($Grid in $Grids) {
        if (@($Grid).Count -ne $Expected) {
            throw "All mini-audit profile grids must have the same length ($Expected expected)."
        }
    }
    if ($Expected -eq 0) { throw "At least one mini-audit profile is required." }
}

Assert-EqualGridLengths @(
    $MiniSamplesGrid,
    $MiniMaxGroupsGrid,
    $MiniNullTrialsGrid,
    $MiniNullQuantilesGrid,
    $MiniMinCoveragesGrid,
    $MiniMinInterventionGainsGrid,
    $MiniActivationDifferenceQuantilesGrid,
    $MiniResidualThresholdsGrid
)

foreach ($Variant in $SpatialVariants) {
    if ($Variant -ne "none") {
        [void](Find-BalanceArtifact $Variant)
    }
}

function New-SpatialJobs {
    foreach ($Variant in $SpatialVariants) {
        if ($Variant -eq "none") {
            [pscustomobject]@{ Variant = "none"; Floor = 0.25; FrequencyPower = 0.0 }
            continue
        }
        foreach ($Floor in $SpatialFloors) {
            foreach ($Power in $SpatialFrequencyPowers) {
                [pscustomobject]@{ Variant = $Variant; Floor = $Floor; FrequencyPower = $Power }
            }
        }
    }
}

function New-GroupJobs {
    $SpatialJobs = @(New-SpatialJobs)
    foreach ($MinFrequency in $GroupMinConceptFrequencies) {
        foreach ($MaxFrequency in $GroupMaxConceptFrequencies) {
            if ($MaxFrequency -lt $MinFrequency) { continue }
            foreach ($TextSimilarity in $GroupTextSimilarityThresholds) {
                foreach ($Coactivation in $GroupCoactivationThresholds) {
                    foreach ($Spatial in $SpatialJobs) {
                        $SpatialToken = if ($Spatial.Variant -eq "none") {
                            "plain"
                        }
                        else {
                            "s$($Spatial.Variant)_f$(Format-RunValue $Spatial.Floor)_p$(Format-RunValue $Spatial.FrequencyPower)"
                        }
                        $Name = "g_min$(Format-RunValue $MinFrequency)_max$(Format-RunValue $MaxFrequency)_t$(Format-RunValue $TextSimilarity)_c$(Format-RunValue $Coactivation)_$SpatialToken"
                        [pscustomobject]@{
                            Name = $Name
                            MinFrequency = [double]$MinFrequency
                            MaxFrequency = [double]$MaxFrequency
                            TextSimilarity = [double]$TextSimilarity
                            Coactivation = [double]$Coactivation
                            SpatialVariant = [string]$Spatial.Variant
                            SpatialFloor = [double]$Spatial.Floor
                            SpatialFrequencyPower = [double]$Spatial.FrequencyPower
                        }
                    }
                }
            }
        }
    }
}

function New-ReportRow {
    param(
        [string]$JsonPath,
        [string]$RunName,
        [string]$Stage,
        [string]$Status,
        [string]$Profile = "",
        [string]$ErrorMessage = ""
    )
    if (-not (Test-Path -LiteralPath $JsonPath)) {
        return [pscustomobject]@{
            stage = $Stage
            run_name = $RunName
            profile = $Profile
            status = $Status
            decision = ""
            reason = $ErrorMessage
            min_frequency = ""
            max_frequency = ""
            text_similarity = ""
            coactivation = ""
            spatial_variant = ""
            spatial_floor = ""
            spatial_frequency_power = ""
            fidelity_threshold = ""
            target_coverage = ""
            candidate_groups = ""
            selected_groups = ""
            selected_concepts = ""
            compression_ratio = ""
            source_coverage = ""
            median_source_fidelity = ""
            p01_source_fidelity = ""
            coherence_warnings = ""
            audited_groups = ""
            requested_groups = ""
            null_pass_fraction = ""
            median_top1_turnover = ""
            median_jaccard_at_k = ""
            geometry_changed = ""
            mini_passed = ""
            balanced_score = ""
            json = $JsonPath
            html = [System.IO.Path]::ChangeExtension($JsonPath, ".html")
        }
    }
    $Report = Get-Content -Raw -LiteralPath $JsonPath | ConvertFrom-Json
    $Group = $Report.group_config
    $Screen = $Report.screen_config
    $Metrics = $Report.metrics
    $Mini = $Report.mini_intervention
    return [pscustomobject]@{
        stage = $Stage
        run_name = $RunName
        profile = $Profile
        status = $Status
        decision = [string]$Report.decision.status
        reason = [string]$Report.decision.reason
        min_frequency = $Group.min_concept_frequency
        max_frequency = $Group.max_concept_frequency
        text_similarity = $Group.text_similarity_threshold
        coactivation = $Group.coactivation_threshold
        spatial_variant = $Group.spatial_balance_variant
        spatial_floor = $Group.spatial_balance_floor
        spatial_frequency_power = $Group.spatial_frequency_power
        fidelity_threshold = $Screen.fidelity_threshold
        target_coverage = $Screen.target_image_coverage
        candidate_groups = $Metrics.candidate_group_count
        selected_groups = $Metrics.selected_group_count
        selected_concepts = $Metrics.selected_concept_count
        compression_ratio = $Metrics.compression_ratio
        source_coverage = $Metrics.source_coverage
        median_source_fidelity = $Metrics.median_source_fidelity
        p01_source_fidelity = $Metrics.p01_source_fidelity
        coherence_warnings = $Metrics.coherence_warning_count
        audited_groups = if ($null -ne $Mini) { $Mini.audited_group_count } else { "" }
        requested_groups = if ($null -ne $Mini) { $Mini.requested_group_count } else { "" }
        null_pass_fraction = if ($null -ne $Mini) { $Mini.null_pass_fraction } else { "" }
        median_top1_turnover = if ($null -ne $Mini) { $Mini.median_top1_neighbor_turnover } else { "" }
        median_jaccard_at_k = if ($null -ne $Mini) { $Mini.median_jaccard_at_k } else { "" }
        geometry_changed = if ($null -ne $Mini) { $Mini.geometry_changed } else { "" }
        mini_passed = if ($null -ne $Mini) { $Mini.passed } else { "" }
        balanced_score = ""
        json = $JsonPath
        html = [System.IO.Path]::ChangeExtension($JsonPath, ".html")
    }
}

function Add-BalancedScores([object[]]$Rows) {
    $Valid = @($Rows | Where-Object {
        $_.status -ne "failed" -and $_.decision -and $_.source_coverage -ne ""
    })
    if ($Valid.Count -eq 0) { return @($Rows) }
    $Columns = @("source_coverage", "median_source_fidelity", "p01_source_fidelity", "compression_ratio")
    $Ranges = @{}
    foreach ($Column in $Columns) {
        $Values = @($Valid | ForEach-Object { [double]($_.$Column) })
        $Ranges[$Column] = @(
            ($Values | Measure-Object -Minimum).Minimum,
            ($Values | Measure-Object -Maximum).Maximum
        )
    }
    foreach ($Row in $Rows) {
        if ($Row.status -eq "failed" -or -not $Row.decision) {
            $Row.balanced_score = 0.0
            continue
        }
        $Score = 0.0
        foreach ($Spec in @(
            @{ Name = "source_coverage"; Weight = 0.40 },
            @{ Name = "median_source_fidelity"; Weight = 0.25 },
            @{ Name = "p01_source_fidelity"; Weight = 0.20 },
            @{ Name = "compression_ratio"; Weight = 0.15 }
        )) {
            $Low = [double]$Ranges[$Spec.Name][0]
            $High = [double]$Ranges[$Spec.Name][1]
            $Value = [double]$Row.($Spec.Name)
            $Normalized = if ($High -gt $Low) { ($Value - $Low) / ($High - $Low) } else { 0.5 }
            $Score += $Spec.Weight * $Normalized
        }
        $Row.balanced_score = $Score
    }
    return @($Rows)
}

function Export-Summary([object[]]$Rows, [string]$Path) {
    if ($Rows.Count -gt 0) {
        $Rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $Path
    }
}

function Invoke-GroupScreen {
    param(
        [pscustomobject]$Job,
        [string]$Stage,
        [string]$OutputDirectory,
        [double]$FidelityThreshold,
        [double]$TargetCoverage,
        [bool]$MiniAudit,
        [int]$MiniSamples = 1024,
        [int]$MiniMaxGroups = 24,
        [int]$MiniNullTrials = 4,
        [double]$MiniNullQuantile = 0.95,
        [double]$MiniMinCoverage = 0.01,
        [double]$MiniMinInterventionGain = 5e-4,
        [double]$MiniActivationDifferenceQuantile = 0.85,
        [double]$MiniResidualThreshold = 0.25,
        [string]$Profile = ""
    )
    $JsonPath = Join-Path $OutputDirectory "group_screen.json"
    $HtmlPath = Join-Path $OutputDirectory "group_screen.html"
    if ((-not $Force) -and (Test-Path -LiteralPath $JsonPath)) {
        Write-Host "[SKIP] $Stage $($Job.Name) $Profile (report exists)"
        return New-ReportRow $JsonPath $Job.Name $Stage "reused" $Profile
    }
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
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
        "--fidelity-threshold", $FidelityThreshold.ToString($Invariant),
        "--target-image-coverage", $TargetCoverage.ToString($Invariant),
        "--mini-audit", $(if ($MiniAudit) { "true" } else { "false" }),
        "--mini-samples", $MiniSamples.ToString($Invariant),
        "--mini-max-groups", $MiniMaxGroups.ToString($Invariant),
        "--mini-null-trials", $MiniNullTrials.ToString($Invariant),
        "--mini-null-quantile", $MiniNullQuantile.ToString($Invariant),
        "--mini-min-coverage", $MiniMinCoverage.ToString($Invariant),
        "--mini-min-intervention-gain", $MiniMinInterventionGain.ToString($Invariant),
        "--mini-activation-difference-quantile", $MiniActivationDifferenceQuantile.ToString($Invariant),
        "--mini-residual-threshold", $MiniResidualThreshold.ToString($Invariant),
        "--seed", $Seed.ToString($Invariant)
    )
    if ($Job.SpatialVariant -ne "none") {
        $BalancePath = Find-BalanceArtifact $Job.SpatialVariant
        $Arguments += @(
            "--spatial-variant", $Job.SpatialVariant,
            "--spatial-floor", $Job.SpatialFloor.ToString($Invariant),
            "--spatial-frequency-power", $Job.SpatialFrequencyPower.ToString($Invariant),
            "--spatial-balance-artifact", $BalancePath
        )
    }
    Write-Host "[RUN] $Stage $($Job.Name) $Profile"
    & $CondaExe @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        $Message = "group screen exited with code $ExitCode"
        Write-Warning "$($Job.Name): $Message"
        return New-ReportRow $JsonPath $Job.Name $Stage "failed" $Profile $Message
    }
    return New-ReportRow $JsonPath $Job.Name $Stage "completed" $Profile
}

$GroupJobs = @(New-GroupJobs)
$Phase1Jobs = if ($Phase1MaxRuns -gt 0) { @($GroupJobs | Select-Object -First $Phase1MaxRuns) } else { $GroupJobs }
$MiniProfileCount = @($MiniSamplesGrid).Count
$Phase1Count = $Phase1Jobs.Count
$Phase2Planned = [math]::Min($Phase2TopGroupConfigs, $Phase1Count) * @($RefineFidelityThresholds).Count * @($RefineTargetCoverages).Count
$Phase3PlannedUpperBound = [math]::Ceiling($Phase2Planned * $Phase3TopFraction) * $MiniProfileCount

Write-Host "Repository:                 $RepoRoot"
Write-Host "Dataset:                    $Dataset"
Write-Host "Cache:                      $CachePath"
Write-Host "Output root:                $OutputRoot"
Write-Host "Phase 1 group configurations: $Phase1Count"
Write-Host "Phase 2 planned reports:      $Phase2Planned"
Write-Host "Phase 3 mini-audits upper bound: $Phase3PlannedUpperBound"
Write-Host "Time budget:                $MaxHours hours"
Write-Host "Spatial variants:           $($SpatialVariants -join ', ')"

if ($ValidateOnly) {
    Write-Host "Validation only; no runs were started."
    $Phase1Jobs | Select-Object -First 10 Name, MinFrequency, MaxFrequency, TextSimilarity, Coactivation, SpatialVariant | Format-Table -AutoSize
    return
}

$CondaExe = Find-CondaExecutable
New-Item -ItemType Directory -Force -Path $Phase1Root, $Phase2Root, $Phase3Root | Out-Null

# Phase 1: broad group-construction grid at a neutral reference gate.
$Phase1Rows = @()
$Phase1Stopped = $false
foreach ($Job in $Phase1Jobs) {
    if (-not (Test-TimeBudget)) { $Phase1Stopped = $true; break }
    $OutputDirectory = Join-Path $Phase1Root $Job.Name
    $Phase1Rows += Invoke-GroupScreen $Job "phase1" $OutputDirectory $ReferenceFidelityThreshold $ReferenceTargetCoverage $false
    Export-Summary $Phase1Rows $Phase1SummaryPath
}
if ($Phase1Rows.Count -eq 0 -and (Test-Path -LiteralPath $Phase1SummaryPath)) {
    $Phase1Rows = @(Import-Csv -LiteralPath $Phase1SummaryPath)
}
$Phase1Rows = @(Add-BalancedScores $Phase1Rows)
Export-Summary $Phase1Rows $Phase1SummaryPath

if ($Phase1Stopped) {
    Write-Warning "Phase 1 stopped by time budget; refining the completed subset."
}

# Select a diverse refinement set: balanced score plus separate coverage/tail/compression leaders.
$Phase1Valid = @($Phase1Rows | Where-Object { $_.status -ne "failed" -and $_.decision })
$RefineNames = @{}
$RefineRows = @()
foreach ($Row in ($Phase1Valid | Sort-Object {[double]$_.balanced_score} -Descending | Select-Object -First $Phase2TopGroupConfigs)) {
    if (-not $RefineNames.ContainsKey($Row.run_name) -and $RefineRows.Count -lt $Phase2TopGroupConfigs) {
        $RefineNames[$Row.run_name] = $true
        $RefineRows += $Row
    }
}
foreach ($LeaderSet in @(
    ($Phase1Valid | Sort-Object {[double]$_.source_coverage} -Descending | Select-Object -First 10),
    ($Phase1Valid | Sort-Object {[double]$_.p01_source_fidelity} -Descending | Select-Object -First 10),
    ($Phase1Valid | Sort-Object {[double]$_.compression_ratio} -Descending | Select-Object -First 10)
)) {
    foreach ($Row in $LeaderSet) {
        if (-not $RefineNames.ContainsKey($Row.run_name) -and $RefineRows.Count -lt $Phase2TopGroupConfigs) {
            $RefineNames[$Row.run_name] = $true
            $RefineRows += $Row
        }
    }
}
$RefineRows = @($RefineRows)
$RefineJobsByName = @{}
foreach ($Job in $GroupJobs) { $RefineJobsByName[$Job.Name] = $Job }

# Phase 2: vary fidelity and target coverage for the selected groupings.
$Phase2Rows = @()
$Phase2Stopped = $false
foreach ($BaseRow in $RefineRows) {
    $BaseJob = $RefineJobsByName[$BaseRow.run_name]
    foreach ($Fidelity in $RefineFidelityThresholds) {
        foreach ($Target in $RefineTargetCoverages) {
            if (-not (Test-TimeBudget)) { $Phase2Stopped = $true; break }
            $RefinedName = "$($BaseJob.Name)_f$(Format-RunValue $Fidelity)_cov$(Format-RunValue $Target)"
            $RefinedJob = [pscustomobject]@{
                Name = $RefinedName
                MinFrequency = $BaseJob.MinFrequency
                MaxFrequency = $BaseJob.MaxFrequency
                TextSimilarity = $BaseJob.TextSimilarity
                Coactivation = $BaseJob.Coactivation
                SpatialVariant = $BaseJob.SpatialVariant
                SpatialFloor = $BaseJob.SpatialFloor
                SpatialFrequencyPower = $BaseJob.SpatialFrequencyPower
            }
            $OutputDirectory = Join-Path $Phase2Root $RefinedName
            $Phase2Rows += Invoke-GroupScreen $RefinedJob "phase2" $OutputDirectory $Fidelity $Target $false
            Export-Summary $Phase2Rows $Phase2SummaryPath
        }
        if ($Phase2Stopped) { break }
    }
    if ($Phase2Stopped) { break }
}
if ($Phase2Rows.Count -eq 0 -and (Test-Path -LiteralPath $Phase2SummaryPath)) {
    $Phase2Rows = @(Import-Csv -LiteralPath $Phase2SummaryPath)
}
$Phase2Rows = @(Add-BalancedScores $Phase2Rows)
Export-Summary $Phase2Rows $Phase2SummaryPath

if ($Phase2Stopped) {
    Write-Warning "Phase 2 stopped by time budget; running mini-audits on the completed subset."
}

# Phase 3: mini-audit the top fraction under the balanced reconstruction trade-off.
$Phase2Valid = @($Phase2Rows | Where-Object { $_.status -ne "failed" -and $_.decision })
$Phase2Ranked = @($Phase2Valid | Sort-Object {[double]$_.balanced_score}, {[double]$_.source_coverage}, {[double]$_.median_source_fidelity} -Descending)
$Phase3CandidateCount = [math]::Max(1, [math]::Ceiling($Phase2Ranked.Count * $Phase3TopFraction))
$Phase3Candidates = @($Phase2Ranked | Select-Object -First $Phase3CandidateCount)

$Phase3Rows = @()
$Phase3Stopped = $false
$ProfileCount = @($MiniSamplesGrid).Count
for ($CandidateIndex = 0; $CandidateIndex -lt $Phase3Candidates.Count; $CandidateIndex++) {
    $Candidate = $Phase3Candidates[$CandidateIndex]
    $CandidateJob = [pscustomobject]@{
        Name = $Candidate.run_name
        MinFrequency = [double]$Candidate.min_frequency
        MaxFrequency = [double]$Candidate.max_frequency
        TextSimilarity = [double]$Candidate.text_similarity
        Coactivation = [double]$Candidate.coactivation
        SpatialVariant = if ([string]::IsNullOrWhiteSpace([string]$Candidate.spatial_variant)) { "none" } else { [string]$Candidate.spatial_variant }
        SpatialFloor = [double]$Candidate.spatial_floor
        SpatialFrequencyPower = [double]$Candidate.spatial_frequency_power
    }
    for ($ProfileIndex = 0; $ProfileIndex -lt $ProfileCount; $ProfileIndex++) {
        if (-not (Test-TimeBudget)) { $Phase3Stopped = $true; break }
        $ProfileName = "mini$('{0:D2}' -f ($ProfileIndex + 1))"
        $MiniName = "$($Candidate.run_name)_$ProfileName"
        $MiniJob = [pscustomobject]@{
            Name = $MiniName
            MinFrequency = $CandidateJob.MinFrequency
            MaxFrequency = $CandidateJob.MaxFrequency
            TextSimilarity = $CandidateJob.TextSimilarity
            Coactivation = $CandidateJob.Coactivation
            SpatialVariant = $CandidateJob.SpatialVariant
            SpatialFloor = $CandidateJob.SpatialFloor
            SpatialFrequencyPower = $CandidateJob.SpatialFrequencyPower
        }
        $OutputDirectory = Join-Path $Phase3Root $MiniName
        $Phase3Rows += Invoke-GroupScreen $MiniJob "phase3" $OutputDirectory ([double]$Candidate.fidelity_threshold) ([double]$Candidate.target_coverage) $true `
            $MiniSamplesGrid[$ProfileIndex] $MiniMaxGroupsGrid[$ProfileIndex] $MiniNullTrialsGrid[$ProfileIndex] `
            $MiniNullQuantilesGrid[$ProfileIndex] $MiniMinCoveragesGrid[$ProfileIndex] $MiniMinInterventionGainsGrid[$ProfileIndex] `
            $MiniActivationDifferenceQuantilesGrid[$ProfileIndex] $MiniResidualThresholdsGrid[$ProfileIndex] $ProfileName
        Export-Summary $Phase3Rows $Phase3SummaryPath
        if ($MaxMiniAudits -gt 0 -and $Phase3Rows.Count -ge $MaxMiniAudits) {
            $Phase3Stopped = $true
            break
        }
    }
    if ($Phase3Stopped) { break }
}
if ($Phase3Rows.Count -eq 0 -and (Test-Path -LiteralPath $Phase3SummaryPath)) {
    $Phase3Rows = @(Import-Csv -LiteralPath $Phase3SummaryPath)
}

Export-Summary $Phase3Rows $Phase3SummaryPath

Write-Host ""
Write-Host "Mega-sweep finished or reached its time budget."
Write-Host ("Elapsed: {0:N2} hours" -f $Stopwatch.Elapsed.TotalHours)
Write-Host "Phase 1 summary: $Phase1SummaryPath"
Write-Host "Phase 2 summary: $Phase2SummaryPath"
Write-Host "Phase 3 summary: $Phase3SummaryPath"

if ($Phase3Rows.Count -gt 0) {
    Write-Host ""
    Write-Host "Top mini-audit results:"
    $Phase3Rows |
        Sort-Object @{ Expression = { [double]($_.null_pass_fraction -as [double]) }; Descending = $true },
                    @{ Expression = { [double]($_.median_top1_turnover -as [double]) }; Descending = $true },
                    @{ Expression = { [double]($_.median_jaccard_at_k -as [double]) }; Descending = $false } |
        Select-Object -First 30 run_name, profile, decision, source_coverage, median_source_fidelity, null_pass_fraction, median_top1_turnover, median_jaccard_at_k, geometry_changed, mini_passed |
        Format-Table -AutoSize
}
