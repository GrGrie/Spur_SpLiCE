[CmdletBinding()]
param(
    [ValidateSet("all", "baseline", "vanilla_patchwise", "vanilla_slots", "sclip_patchwise", "sclip_slots")]
    [string]$Variant = "all",
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$SourceAblationRoot = "",
    [string]$OutputRoot = "",
    [string]$CachePath = "",
    [string]$BalancePath = "",
    [ValidateRange(1, 500)]
    [int]$Epochs = 100,
    [ValidateRange(1, 500)]
    [int]$SpatialEpochs = 50,
    [ValidateRange(1, 64)]
    [int]$ConceptsPerRegion = 4,
    [ValidateRange(1, 200)]
    [int]$ProbeEpochs = 30,
    [ValidateRange(1, 500)]
    [int]$LinearProbeFrequency = 25,
    [ValidateRange(1, 1024)]
    [int]$BatchSize = 128,
    [ValidateRange(0, 32)]
    [int]$Workers = 4,
    [ValidateRange(1, 128)]
    [int]$NullTrials = 16,
    [ValidateSet("offline", "online")]
    [string]$WandbMode = "online",
    [string]$WandbProject = "Spur_SpLiCE",
    [string]$WandbEntity = "gsgrechkin-rptu",
    [switch]$PrepareOnly,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
if ([string]::IsNullOrWhiteSpace($SourceAblationRoot)) {
    $SourceAblationRoot = Join-Path $RepoRoot "outputs\windows_crpv4_ablation"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\windows_crpv4_diverse"
}
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$SourceAblationRoot = [System.IO.Path]::GetFullPath($SourceAblationRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if ($Variant -eq "all") {
    if (-not [string]::IsNullOrWhiteSpace($BalancePath)) {
        throw "-BalancePath can only be used with one explicit -Variant."
    }
    $Arms = if ($PrepareOnly) {
        @("vanilla_slots", "vanilla_patchwise")
    }
    else {
        @("vanilla_slots", "vanilla_patchwise", "baseline")
    }
    Write-Host "Diverse experiment arms: $($Arms -join ', ')"
    foreach ($Arm in $Arms) {
        Write-Host ""
        Write-Host "=== Diverse experiment arm: $Arm ==="
        $Forwarded = @{
            Variant = $Arm
            CondaEnvironment = $CondaEnvironment
            DataRoot = $DataRoot
            SourceAblationRoot = $SourceAblationRoot
            OutputRoot = $OutputRoot
            CachePath = $CachePath
            Epochs = $Epochs
            SpatialEpochs = $SpatialEpochs
            ConceptsPerRegion = $ConceptsPerRegion
            ProbeEpochs = $ProbeEpochs
            LinearProbeFrequency = $LinearProbeFrequency
            BatchSize = $BatchSize
            Workers = $Workers
            NullTrials = $NullTrials
            WandbMode = $WandbMode
            WandbProject = $WandbProject
            WandbEntity = $WandbEntity
            PrepareOnly = $PrepareOnly
            ValidateOnly = $ValidateOnly
        }
        & $PSCommandPath @Forwarded
    }
    if (-not $ValidateOnly -and -not $PrepareOnly) {
        $CombinedResults = foreach ($Arm in $Arms) {
            $ArmResultsPath = Join-Path $OutputRoot "$Arm\results.csv"
            if (-not (Test-Path -LiteralPath $ArmResultsPath)) {
                throw "Expected arm results at $ArmResultsPath"
            }
            Import-Csv -LiteralPath $ArmResultsPath
        }
        $CombinedResultsPath = Join-Path $OutputRoot "results.csv"
        $CombinedResults | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $CombinedResultsPath
        Write-Host ""
        $CombinedResults | Format-Table -AutoSize
        Write-Host "Combined diverse results: $CombinedResultsPath"
    }
    return
}

$IsBaseline = $Variant -eq "baseline"
$UsesSlots = -not $IsBaseline -and $Variant.EndsWith("_slots")
$SpatialIdentity = if ($UsesSlots) { "slotse$SpatialEpochs" } else { "patchwise" }
if ($IsBaseline) {
    if (-not [string]::IsNullOrWhiteSpace($BalancePath)) {
        throw "The SimCLR baseline does not accept -BalancePath."
    }
    $BalancePath = ""
}
elseif ([string]::IsNullOrWhiteSpace($BalancePath)) {
    $BalancePath = Join-Path $SourceAblationRoot `
        "artifacts\$Variant\balance_${SpatialIdentity}_k$ConceptsPerRegion.pt"
}
if (-not [string]::IsNullOrWhiteSpace($BalancePath)) {
    $BalancePath = [System.IO.Path]::GetFullPath($BalancePath)
}

if ([string]::IsNullOrWhiteSpace($CachePath)) {
    $CacheCandidates = @(
        (Join-Path $RepoRoot "outputs\crp\waterbirds_train_features_oi_v7.pt"),
        (Join-Path $RepoRoot "outputs\windows_splice_only_ablation\waterbirds_train_features_oi_v7.pt"),
        (Join-Path $SourceAblationRoot "waterbirds_train_features_oi_v7.pt")
    )
    $CachePath = $CacheCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not [string]::IsNullOrWhiteSpace($CachePath)) {
    $CachePath = [System.IO.Path]::GetFullPath($CachePath)
}

Write-Host "Repository:       $RepoRoot"
Write-Host "Variant:          $Variant"
Write-Host "Source balance:   $(if ($IsBaseline) { 'not applicable' } else { $BalancePath })"
Write-Host "Cache:            $(if ($IsBaseline) { 'not applicable' } else { $CachePath })"
Write-Host "Diverse output:   $OutputRoot"
Write-Host "SSL epochs/seed:  $Epochs / 0"
Write-Host "Linear probe:     every $LinearProbeFrequency SSL epochs"
if ($ValidateOnly) {
    return
}

if (-not $IsBaseline -and ([string]::IsNullOrWhiteSpace($CachePath) -or -not (Test-Path -LiteralPath $CachePath))) {
    throw "A matching frozen cache was not found. Pass -CachePath explicitly."
}
if (-not $IsBaseline -and -not (Test-Path -LiteralPath $BalancePath)) {
    throw "Spatial evidence is not ready at $BalancePath. Let the four-way launcher finish this variant or pass -BalancePath explicitly."
}

function Find-CondaExecutable {
    $Command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        return $Command.Source
    }
    foreach ($Candidate in @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )) {
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    throw "conda.exe was not found."
}

$CondaExe = Find-CondaExecutable

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$LogPath = "",
        [switch]$AllowFailure,
        [switch]$ReturnExitCode
    )
    $CondaArguments = @("run", "--no-capture-output", "-n", $CondaEnvironment, "python") + $Arguments
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ([string]::IsNullOrWhiteSpace($LogPath)) {
            & $CondaExe @CondaArguments
        }
        else {
            & $CondaExe @CondaArguments 2>&1 | Tee-Object -FilePath $LogPath
        }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0 -and -not $AllowFailure) {
        throw "Python stage failed with exit code $ExitCode. See $LogPath"
    }
    if ($ReturnExitCode) {
        return $ExitCode
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$VariantRoot = Join-Path $OutputRoot $Variant
$LogsRoot = Join-Path $VariantRoot "logs"
$TrainingRoot = Join-Path $VariantRoot "training"
$WandbRoot = Join-Path $VariantRoot "wandb"
New-Item -ItemType Directory -Force -Path $VariantRoot, $LogsRoot, $TrainingRoot, $WandbRoot | Out-Null
$env:PYTHONUTF8 = "1"
$env:WANDB_MODE = $WandbMode
$env:WANDB_DIR = $WandbRoot
Set-Location $RepoRoot

Invoke-CondaPython -Arguments @(
    "-c",
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU',torch.cuda.get_device_name(0))"
)

if ($IsBaseline) {
    $TrainingLog = Join-Path $LogsRoot `
        "ssl_simclr_e${Epochs}_periodic${LinearProbeFrequency}_probe${ProbeEpochs}.log"
    $FinalPattern = "Last accuracy: ([0-9.]+), Last worst-group accuracy: ([0-9.]+), Last best-group accuracy: ([0-9.]+)"
    $Completed = (Test-Path -LiteralPath $TrainingLog) -and $null -ne (
        Select-String -LiteralPath $TrainingLog -Pattern $FinalPattern | Select-Object -Last 1
    )
    if ($Completed) {
        Write-Host "[REUSE] Completed matched SimCLR baseline: $TrainingLog"
    }
    else {
        Write-Host "=== Training matched SimCLR baseline ($Epochs epochs, seed 0) ==="
        Invoke-CondaPython -Arguments @(
            "-u", "spur_splice.py",
            "--dataset", "waterbirds",
            "--data_folder", $DataRoot,
            "--model", "resnet18_large",
            "--seed", "0",
            "--trial", "windows_crpv4_diverse_baseline",
            "--epochs", $Epochs.ToString(),
            "--batch_size", $BatchSize.ToString(),
            "--num_workers", $Workers.ToString(),
            "--optimizer", "SGD",
            "--learning_rate", "0.01",
            "--lr_decay_epochs", "auto",
            "--lr_decay_rate", "0.1",
            "--weight_decay", "0.0001",
            "--momentum", "0.9",
            "--temp", "0.05",
            "--simclr_weight", "1.0",
            "--ssl_crop_min", "0.2",
            "--head", "mlp",
            "--feat_dim", "128",
            "--amp", "true",
            "--channels_last", "true",
            "--cudnn_enabled", "true",
            "--cudnn_benchmark", "false",
            "--print_freq", "5",
            "--rank_eval_freq", "0",
            "--train_set_linear_layer", "ds_train",
            "--linear_eval_split", "val",
            "--linear_probe_mode", "periodic",
            "--linear_probe_freq", $LinearProbeFrequency.ToString(),
            "--linear_probe_epochs", $ProbeEpochs.ToString(),
            "--linear_lr_decay_epochs", "auto",
            "--linear_spurious_probe", "false",
            "--checkpoint_dir", $TrainingRoot,
            "--save_freq", $Epochs.ToString(),
            "--checkpoint_keep_count", "1",
            "--delete_checkpoints_after_training", "false",
            "--keep_checkpoints",
            "--splice_mode", "none",
            "--use_wandb",
            "--wandb_name", $WandbProject,
            "--entity", $WandbEntity,
            "--wandb_run_name", "windows_crpv4_diverse_baseline_s0_e${Epochs}_lp$LinearProbeFrequency",
            "--wandb_group", "waterbirds_crpv4_diverse_seed0_e$Epochs",
            "--wandb_tags", "windows_local,crpv4,diverse_screen,baseline,simclr,seed_0,linear_probe_every_$LinearProbeFrequency"
        ) -LogPath $TrainingLog
    }

    $MetricMatch = Select-String -LiteralPath $TrainingLog -Pattern $FinalPattern | Select-Object -Last 1
    if ($null -eq $MetricMatch -or -not $MetricMatch.Matches[0].Success) {
        throw "Could not read final probe metrics from $TrainingLog"
    }
    $Result = [pscustomobject]@{
        variant = $Variant
        seed = 0
        ssl_epochs = $Epochs
        linear_probe_frequency = $LinearProbeFrequency
        source_groups = 0
        audited_groups = 0
        quality_gate_groups = 0
        selected_groups = 0
        graph_edges = 0
        graph_coverage = 0.0
        validation_accuracy = [double]::Parse(
            $MetricMatch.Matches[0].Groups[1].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        validation_worst_group_accuracy = [double]::Parse(
            $MetricMatch.Matches[0].Groups[2].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        validation_best_group_accuracy = [double]::Parse(
            $MetricMatch.Matches[0].Groups[3].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        selected_concepts = ""
        graph_path = ""
        log_path = $TrainingLog
    }
    $ResultsPath = Join-Path $VariantRoot "results.csv"
    $Result | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ResultsPath
    $Result | Format-List
    Write-Host "Results: $ResultsPath"
    return
}

$AuditConfig = [ordered]@{
    min_concept_frequency = 0.005
    max_concept_frequency = 0.80
    text_similarity_threshold = 0.72
    coactivation_threshold = 0.10
    min_group_size = 1
    max_selected_groups = 0
    projected_neighbors = 20
    activation_difference_quantile = 0.85
    min_intervention_gain = 0.0005
    min_coverage = 0.01
    graph_top_k = 3
    max_indegree = 10
    indegree_factor = 3.0
    null_trials = $NullTrials
    null_quantile = 0.95
    similarity_chunk_size = 512
    orthogonal_tolerance = 0.000001
    use_residual_splice_gate = $true
    residual_splice_similarity_threshold = 0.25
    use_cobalt_confidence = $false
    spatial_balance = $true
    spatial_balance_variant = $Variant
    spatial_balance_floor = 0.25
    spatial_frequency_power = 0.50
    seed = 0
    cobalt = $false
}
$DiversityConfig = [ordered]@{
    semantic_cluster_count = 12
    candidates_per_cluster = 4
    candidate_budget = 48
    selected_group_count = 8
    max_selected_per_cluster = 1
    semantic_similarity_ceiling = 0.75
    semantic_redundancy_penalty = 0.75
    edge_overlap_penalty = 0.25
    preaudit_quality_weight = 0.25
    spatial_support_weight = 0.15
    activation_entropy_weight = 0.65
    spatial_agreement_weight = 0.25
    preaudit_spatial_support_weight = 0.10
    kmeans_iterations = 12
}
$AuditConfigJson = $AuditConfig | ConvertTo-Json -Compress
$DiversityConfigJson = $DiversityConfig | ConvertTo-Json -Compress
$AuditConfigNative = $AuditConfigJson -replace '"', '\"'
$DiversityConfigNative = $DiversityConfigJson -replace '"', '\"'
$GraphIdentity = "diverse_v1_${SpatialIdentity}_regions${ConceptsPerRegion}_mf005_t072_c010_k48_s8_null$NullTrials"
$GraphPath = Join-Path $VariantRoot "teacher_graph_${GraphIdentity}.json"

$GraphReusable = $false
if (Test-Path -LiteralPath $GraphPath) {
    $GraphReuseCheck = "import json,sys; g=json.load(open(sys.argv[1],encoding='utf-8')); a=json.loads(sys.argv[2]); d=json.loads(sys.argv[3]); ok=g.get('artifact')=='splice_crp_v4_teacher_graph' and g.get('config')==a and g.get('diverse_selection',{}).get('config')==d; raise SystemExit(0 if ok else 1)"
    $ReuseExit = Invoke-CondaPython -Arguments @(
        "-c", $GraphReuseCheck, $GraphPath, $AuditConfigNative, $DiversityConfigNative
    ) -AllowFailure -ReturnExitCode
    $GraphReusable = $ReuseExit -eq 0
}
if ($GraphReusable) {
    Write-Host "[REUSE] Matching diverse graph: $GraphPath"
}
else {
    Write-Host "=== Building isolated diversity-constrained CRPv4 graph ==="
    Invoke-CondaPython -Arguments @(
        "-u", "-m", "splice.crp_diverse",
        "--cache", $CachePath,
        "--spatial-balance-artifact", $BalancePath,
        "--output", $GraphPath,
        "--config", $AuditConfigNative,
        "--diversity-config", $DiversityConfigNative
    ) -LogPath (Join-Path $LogsRoot "graph_${GraphIdentity}.log")
}

$GraphValidation = "import json,sys,torch; from splice.crp import validate_feature_cache; from splice.crp_training import validate_teacher_graph; c=validate_feature_cache(torch.load(sys.argv[2],map_location='cpu',weights_only=True)); g=json.load(open(sys.argv[1],encoding='utf-8')); validate_teacher_graph(g,c['sample_ids']); d=g['diverse_selection']; print('[OK] source',d['source_group_count'],'audited',d['preselected_group_count'],'passed',d['quality_gate_count'],'selected',d['selected_group_count']); print('[INFO] concepts',[x['concepts'] for x in g['groups'] if x['selected']]); print('[WARN] empty graph uses SimCLR fallback') if g['degree_stats']['edge_count']==0 else None"
Invoke-CondaPython -Arguments @("-c", $GraphValidation, $GraphPath, $CachePath)
Invoke-CondaPython -Arguments @(
    "-u", "-m", "scripts.tools.summarize_crp_audit", $GraphPath
) -AllowFailure

$AuditHtml = Join-Path $VariantRoot "graph_audit_${GraphIdentity}.html"
if (-not (Test-Path -LiteralPath $AuditHtml)) {
    Invoke-CondaPython -Arguments @(
        "-u", "-m", "scripts.tools.render_concept_ablation_examples",
        "--cache", $CachePath,
        "--graph", $GraphPath,
        "--data-folder", $DataRoot,
        "--scope", "selected",
        "--max-interventions", "8",
        "--edges-per-group", "1",
        "--output", $AuditHtml
    ) -AllowFailure
}

if ($PrepareOnly) {
    Write-Host "[DONE] Diverse graph prepared without student training: $GraphPath"
    return
}

$TrainingLog = Join-Path $LogsRoot `
    "ssl_${GraphIdentity}_e${Epochs}_periodic${LinearProbeFrequency}_probe${ProbeEpochs}.log"
$FinalPattern = "Last accuracy: ([0-9.]+), Last worst-group accuracy: ([0-9.]+), Last best-group accuracy: ([0-9.]+)"
$Completed = (Test-Path -LiteralPath $TrainingLog) -and $null -ne (
    Select-String -LiteralPath $TrainingLog -Pattern $FinalPattern | Select-Object -Last 1
)
if ($Completed) {
    Write-Host "[REUSE] Completed diverse SSL run: $TrainingLog"
}
else {
    Write-Host "=== Training the diverse CRPv4 student ($Epochs epochs, seed 0) ==="
    Invoke-CondaPython -Arguments @(
        "-u", "spur_splice.py",
        "--dataset", "waterbirds",
        "--data_folder", $DataRoot,
        "--model", "resnet18_large",
        "--seed", "0",
        "--trial", "windows_crpv4_diverse",
        "--epochs", $Epochs.ToString(),
        "--batch_size", $BatchSize.ToString(),
        "--num_workers", $Workers.ToString(),
        "--optimizer", "SGD",
        "--learning_rate", "0.01",
        "--lr_decay_epochs", "auto",
        "--lr_decay_rate", "0.1",
        "--weight_decay", "0.0001",
        "--momentum", "0.9",
        "--temp", "0.05",
        "--simclr_weight", "1.0",
        "--ssl_crop_min", "0.2",
        "--head", "mlp",
        "--feat_dim", "128",
        "--amp", "true",
        "--channels_last", "true",
        "--cudnn_enabled", "true",
        "--cudnn_benchmark", "false",
        "--print_freq", "5",
        "--rank_eval_freq", "0",
        "--train_set_linear_layer", "ds_train",
        "--linear_eval_split", "val",
        "--linear_probe_mode", "periodic",
        "--linear_probe_freq", $LinearProbeFrequency.ToString(),
        "--linear_probe_epochs", $ProbeEpochs.ToString(),
        "--linear_lr_decay_epochs", "auto",
        "--linear_spurious_probe", "false",
        "--checkpoint_dir", $TrainingRoot,
        "--save_freq", $Epochs.ToString(),
        "--checkpoint_keep_count", "1",
        "--delete_checkpoints_after_training", "false",
        "--keep_checkpoints",
        "--splice_mode", "crp_relational",
        "--splice_weight", "2.0",
        "--crp_teacher_graph", $GraphPath,
        "--crp_temperature", "0.25",
        "--crp_start_epoch", "10",
        "--crp_warmup_epochs", "10",
        "--crp_decay_start_epoch", "0",
        "--crp_decay_end_epoch", "0",
        "--use_wandb",
        "--wandb_name", $WandbProject,
        "--entity", $WandbEntity,
        "--wandb_run_name", "windows_crpv4_${Variant}_diverse_s0_e${Epochs}_lp$LinearProbeFrequency",
        "--wandb_group", "waterbirds_crpv4_diverse_seed0_e$Epochs",
        "--wandb_tags", "windows_local,crpv4,diverse_selector,${Variant},seed_0,openimages_v7,linear_probe_every_$LinearProbeFrequency"
    ) -LogPath $TrainingLog
}

$MetricMatch = Select-String -LiteralPath $TrainingLog -Pattern $FinalPattern | Select-Object -Last 1
if ($null -eq $MetricMatch -or -not $MetricMatch.Matches[0].Success) {
    throw "Could not read final probe metrics from $TrainingLog"
}
$Graph = Get-Content -LiteralPath $GraphPath -Raw | ConvertFrom-Json
$SelectedConcepts = @(
    $Graph.groups | Where-Object { $_.selected } | ForEach-Object { $_.concepts -join " + " }
) -join " | "
$Result = [pscustomobject]@{
    variant = $Variant
    seed = 0
    ssl_epochs = $Epochs
    linear_probe_frequency = $LinearProbeFrequency
    source_groups = [int]$Graph.diverse_selection.source_group_count
    audited_groups = [int]$Graph.diverse_selection.preselected_group_count
    quality_gate_groups = [int]$Graph.diverse_selection.quality_gate_count
    selected_groups = [int]$Graph.diverse_selection.selected_group_count
    graph_edges = [int]$Graph.degree_stats.edge_count
    graph_coverage = [double]$Graph.degree_stats.coverage
    validation_accuracy = [double]::Parse(
        $MetricMatch.Matches[0].Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture
    )
    validation_worst_group_accuracy = [double]::Parse(
        $MetricMatch.Matches[0].Groups[2].Value,
        [Globalization.CultureInfo]::InvariantCulture
    )
    validation_best_group_accuracy = [double]::Parse(
        $MetricMatch.Matches[0].Groups[3].Value,
        [Globalization.CultureInfo]::InvariantCulture
    )
    selected_concepts = $SelectedConcepts
    graph_path = $GraphPath
    log_path = $TrainingLog
}
$ResultsPath = Join-Path $VariantRoot "results.csv"
$Result | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ResultsPath
$Result | Format-List
Write-Host "Results: $ResultsPath"
Write-Host "Graph audit: $AuditHtml"
