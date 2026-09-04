[CmdletBinding()]
param(
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$OutputRoot = "",
    [string]$CachePath = "",
    [ValidateRange(1, 500)]
    [int]$Epochs = 100,
    [ValidateRange(1, 500)]
    [int]$SpatialEpochs = 50,
    [ValidateRange(1, 200)]
    [int]$ProbeEpochs = 30,
    [ValidateRange(1, 500)]
    [int]$LinearProbeFrequency = 25,
    [ValidateRange(1, 1024)]
    [int]$BatchSize = 128,
    [ValidateRange(1, 1024)]
    [int]$SpatialBatchSize = 64,
    [ValidateRange(0, 32)]
    [int]$Workers = 4,
    [ValidateRange(1, 128)]
    [int]$NullTrials = 16,
    [ValidateRange(1, 64)]
    [int]$ConceptsPerRegion = 4,
    [ValidateRange(0.0, 1.0)]
    [double]$SpatialBalanceFloor = 0.25,
    [ValidateRange(0.0, 4.0)]
    [double]$SpatialFrequencyPower = 0.0,
    [ValidateSet("offline", "online")]
    [string]$WandbMode = "online",
    [string]$WandbProject = "Spur_SpLiCE",
    [string]$WandbEntity = "gsgrechkin-rptu",
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DriveDataRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "Datasets"
    if (Test-Path -LiteralPath $DriveDataRoot) {
        $DataRoot = $DriveDataRoot
    }
    else {
        $DataRoot = Join-Path $RepoRoot "datasets"
    }
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\windows_crpv4_ablation"
}
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$Variants = @(
    [pscustomobject]@{ Name = "vanilla_patchwise"; FeatureSource = "vanilla"; UseSlots = $false },
    [pscustomobject]@{ Name = "vanilla_slots";     FeatureSource = "vanilla"; UseSlots = $true  },
    [pscustomobject]@{ Name = "sclip_patchwise";  FeatureSource = "sclip";   UseSlots = $false },
    [pscustomobject]@{ Name = "sclip_slots";      FeatureSource = "sclip";   UseSlots = $true  }
)

if ($ValidateOnly) {
    Write-Host "Repository: $RepoRoot"
    Write-Host "Data:       $DataRoot"
    Write-Host "Output:     $OutputRoot"
    Write-Host "Seed:       0"
    Write-Host "SSL epochs: $Epochs"
    Write-Host "Slot epochs:$SpatialEpochs"
    Write-Host "Linear probe: every $LinearProbeFrequency SSL epochs"
    Write-Host "Variants:   $($Variants.Name -join ', ')"
    return
}

function Find-CondaExecutable {
    $Command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        return $Command.Source
    }
    $Candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    throw "conda.exe was not found. Add Conda to PATH or launch from an Anaconda terminal."
}

$CondaExe = Find-CondaExecutable

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$LogPath = "",
        [switch]$AllowFailure,
        [switch]$ReturnExitCode
    )

    $CondaArguments = @("run", "--no-capture-output", "-n", $CondaEnvironment, "python") + $Arguments
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 may promote ordinary native stderr to an error.
        # Let the process finish and use its actual exit code instead.
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

function Test-FinalProbeLog {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    return $null -ne (
        Select-String -LiteralPath $Path `
            -Pattern "Last accuracy: ([0-9.]+), Last worst-group accuracy: ([0-9.]+), Last best-group accuracy: ([0-9.]+)" |
            Select-Object -Last 1
    )
}

function Read-FinalProbeResult {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string]$GraphPath
    )
    $Match = Select-String -LiteralPath $LogPath `
        -Pattern "Last accuracy: ([0-9.]+), Last worst-group accuracy: ([0-9.]+), Last best-group accuracy: ([0-9.]+)" |
        Select-Object -Last 1
    if ($null -eq $Match -or -not $Match.Matches[0].Success) {
        throw "Could not find the final probe metrics in $LogPath"
    }
    $Graph = Get-Content -LiteralPath $GraphPath -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        variant = $Variant
        seed = 0
        ssl_epochs = $Epochs
        linear_probe_frequency = $LinearProbeFrequency
        slot_epochs = if ($Variant.EndsWith("_slots")) { $SpatialEpochs } else { 0 }
        selected_groups = @($Graph.selected_group_ids).Count
        supported_anchors = [int]$Graph.degree_stats.supported_anchors
        graph_edges = [int]$Graph.degree_stats.edge_count
        graph_coverage = [double]$Graph.degree_stats.coverage
        validation_accuracy = [double]::Parse(
            $Match.Matches[0].Groups[1].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        validation_worst_group_accuracy = [double]::Parse(
            $Match.Matches[0].Groups[2].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        validation_best_group_accuracy = [double]::Parse(
            $Match.Matches[0].Groups[3].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        graph_path = $GraphPath
        log_path = $LogPath
    }
}

New-Item -ItemType Directory -Force -Path $DataRoot, $OutputRoot | Out-Null
$LogsRoot = Join-Path $OutputRoot "logs"
$ArtifactsRoot = Join-Path $OutputRoot "artifacts"
$TrainingRoot = Join-Path $OutputRoot "training"
$WandbRoot = Join-Path $OutputRoot "wandb"
New-Item -ItemType Directory -Force -Path $LogsRoot, $ArtifactsRoot, $TrainingRoot, $WandbRoot | Out-Null

$env:PYTHONUTF8 = "1"
$env:WANDB_MODE = $WandbMode
$env:WANDB_DIR = $WandbRoot
Set-Location $RepoRoot

Write-Host "=== Checking Conda and CUDA ==="
Invoke-CondaPython -Arguments @(
    "-c",
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.randn(64,64,device='cuda'); (x@x).sum().item(); print('PyTorch',torch.__version__,'GPU',torch.cuda.get_device_name(0))"
)

Write-Host "=== Ensuring Waterbirds and the Open Images vocabulary are available ==="
$DatasetsImportExit = Invoke-CondaPython -Arguments @(
    "-c", "import datasets"
) -AllowFailure -ReturnExitCode
if ($DatasetsImportExit -ne 0) {
    Invoke-CondaPython -Arguments @(
        "-m", "pip", "install", "datasets>=3,<5"
    )
}
Invoke-CondaPython -Arguments @(
    "-u", "scripts/tools/download_waterbirds_hf.py",
    "--output-root", $DataRoot
)
Invoke-CondaPython -Arguments @(
    "-u", "scripts/download_openimages_vocabulary.py",
    "--download-root", (Join-Path $RepoRoot "data")
)

if ([string]::IsNullOrWhiteSpace($CachePath)) {
    $CacheCandidates = @(
        (Join-Path $RepoRoot "outputs\crp\waterbirds_train_features_oi_v7.pt"),
        (Join-Path $RepoRoot "outputs\windows_splice_only_ablation\waterbirds_train_features_oi_v7.pt"),
        (Join-Path $OutputRoot "waterbirds_train_features_oi_v7.pt")
    )
    $ExistingCache = $CacheCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    $CachePath = if ($null -ne $ExistingCache) {
        $ExistingCache
    }
    else {
        $CacheCandidates[-1]
    }
}
$CachePath = [System.IO.Path]::GetFullPath($CachePath)

Write-Host "=== Building or validating the frozen SpLiCE cache ==="
if (-not (Test-Path -LiteralPath $CachePath)) {
    Invoke-CondaPython -Arguments @(
        "-u", "-m", "scripts.tools.cache_crp_features",
        "--dataset", "waterbirds",
        "--data-folder", $DataRoot,
        "--output", $CachePath,
        "--batch-size", "64",
        "--num-workers", $Workers.ToString(),
        "--splice-model", "open_clip:ViT-B-32",
        "--splice-pretrained", "laion2b_s34b_b79k",
        "--splice-vocab", "openimages_v7",
        "--splice-vocab-size", "-1",
        "--splice-l1-penalty", "0.25"
    ) -LogPath (Join-Path $LogsRoot "cache.log")
}
$CacheValidation = "import sys,torch; from splice.crp import validate_feature_cache; c=validate_feature_cache(torch.load(sys.argv[1],map_location='cpu',weights_only=True)); p=c['provenance']; assert p['dataset']=='waterbirds' and p['splice_vocab']=='openimages_v7' and p['splice_vocab_size']==-1; print('[OK] cache samples',len(c['sample_ids']))"
Invoke-CondaPython -Arguments @("-c", $CacheValidation, $CachePath) | Out-Null
Write-Host "[OK] Cache: $CachePath"

$GraphConfigBase = [ordered]@{
    min_concept_frequency = 0.01
    max_concept_frequency = 0.95
    text_similarity_threshold = 0.70
    coactivation_threshold = 0.20
    min_group_size = 2
    max_selected_groups = 12
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
    spatial_balance_variant = ""
    spatial_balance_floor = $SpatialBalanceFloor
    spatial_frequency_power = $SpatialFrequencyPower
    seed = 0
    cobalt = $false
}
$FloorToken = $SpatialBalanceFloor.ToString(
    "0.###", [Globalization.CultureInfo]::InvariantCulture
).Replace(".", "p")
$FrequencyToken = $SpatialFrequencyPower.ToString(
    "0.###", [Globalization.CultureInfo]::InvariantCulture
).Replace(".", "p")

$CommonTrainingArguments = @(
    "-u", "spur_splice.py",
    "--dataset", "waterbirds",
    "--data_folder", $DataRoot,
    "--model", "resnet18_large",
    "--seed", "0",
    "--trial", "windows_crpv4_ablation",
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
    "--save_freq", $Epochs.ToString(),
    "--checkpoint_keep_count", "1",
    "--delete_checkpoints_after_training", "false",
    "--keep_checkpoints",
    "--use_wandb",
    "--wandb_name", $WandbProject,
    "--entity", $WandbEntity,
    "--wandb_group", "waterbirds_crpv4_four_way_seed0_e$Epochs"
)

$Results = @()
foreach ($Variant in $Variants) {
    $VariantName = $Variant.Name
    $VariantRoot = Join-Path $ArtifactsRoot $VariantName
    $VariantLogRoot = Join-Path $LogsRoot $VariantName
    # Keep the previous final-only checkpoints intact and prevent their logs
    # from being mistaken for completed periodic-probe runs.
    $VariantTrainingRoot = Join-Path (Join-Path $TrainingRoot $VariantName) `
        "probe_every_$LinearProbeFrequency"
    New-Item -ItemType Directory -Force -Path $VariantRoot, $VariantLogRoot, $VariantTrainingRoot | Out-Null

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "CRPv4 variant: $VariantName"
    Write-Host "============================================================"

    $SlotCheckpoint = Join-Path $VariantRoot "slots_e$SpatialEpochs.pt"
    if ($Variant.UseSlots) {
        $SlotIsReusable = $false
        if (Test-Path -LiteralPath $SlotCheckpoint) {
            $SlotReuseCheck = "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=True); ok=c.get('artifact')=='splice_spatial_slots_v1' and c.get('dataset')=='waterbirds' and c.get('seed')==0 and c.get('epoch',0)>=int(sys.argv[2]) and c.get('model_config',{}).get('feature_source')==sys.argv[3]; raise SystemExit(0 if ok else 1)"
            $SlotReuseExit = Invoke-CondaPython -Arguments @(
                "-c", $SlotReuseCheck, $SlotCheckpoint, $SpatialEpochs.ToString(), $Variant.FeatureSource
            ) -AllowFailure -ReturnExitCode
            $SlotIsReusable = $SlotReuseExit -eq 0
        }
        if ($SlotIsReusable) {
            Write-Host "[REUSE] Spatial slot checkpoint: $SlotCheckpoint"
        }
        else {
            if (Test-Path -LiteralPath $SlotCheckpoint) {
                Write-Host "[RESTART] Existing slot checkpoint is incomplete or incompatible."
            }
            Write-Host "=== Training the $VariantName spatial slot branch ($SpatialEpochs epochs) ==="
            Invoke-CondaPython -Arguments @(
                "-u", "-m", "CoBalT.train_spatial",
                "--dataset", "waterbirds",
                "--data-root", $DataRoot,
                "--cache", $CachePath,
                "--output", $SlotCheckpoint,
                "--seed", "0",
                "--epochs", $SpatialEpochs.ToString(),
                "--batch-size", $SpatialBatchSize.ToString(),
                "--workers", $Workers.ToString(),
                "--image-size", "224",
                "--crop-min", "0.2",
                "--num-slots", "4",
                "--feature-source", $Variant.FeatureSource,
                "--learning-rate", "0.0002",
                "--weight-decay", "0.0005",
                "--student-temperature", "0.1",
                "--teacher-temperature", "0.07",
                "--teacher-momentum", "0.99",
                "--center-momentum", "0.9",
                "--semantic-weight", "1.0",
                "--amp",
                "--wandb-project", $WandbProject,
                "--wandb-entity", $WandbEntity,
                "--wandb-run-name", "windows_crpv4_${VariantName}_spatial_s0_e$SpatialEpochs",
                "--wandb-group", "waterbirds_crpv4_four_way_seed0_e$Epochs",
                "--wandb-tags", "windows_local,crpv4,spatial,${VariantName},label_free,seed_0"
            ) -LogPath (Join-Path $VariantLogRoot "spatial_train_e$SpatialEpochs.log")
        }
    }

    $SpatialIdentity = if ($Variant.UseSlots) { "slotse$SpatialEpochs" } else { "patchwise" }
    $BalancePath = Join-Path $VariantRoot "balance_${SpatialIdentity}_k$ConceptsPerRegion.pt"
    if (Test-Path -LiteralPath $BalancePath) {
        Write-Host "[REUSE] Spatial balance: $BalancePath"
    }
    else {
        Write-Host "=== Extracting $VariantName spatial SpLiCE evidence ==="
        $ExtractArguments = @(
            "-u", "-m", "CoBalT.extract_spatial_balance",
            "--dataset", "waterbirds",
            "--data-root", $DataRoot,
            "--cache", $CachePath,
            "--output", $BalancePath,
            "--variant", $VariantName,
            "--batch-size", $SpatialBatchSize.ToString(),
            "--workers", $Workers.ToString(),
            "--image-size", "224",
            "--concepts-per-region", $ConceptsPerRegion.ToString()
        )
        if ($Variant.UseSlots) {
            $ExtractArguments += @("--checkpoint", $SlotCheckpoint)
        }
        Invoke-CondaPython -Arguments $ExtractArguments `
            -LogPath (Join-Path $VariantLogRoot "spatial_extract_${SpatialIdentity}_k$ConceptsPerRegion.log")
    }

    $BalanceValidation = "import sys,torch; from splice.crp import validate_feature_cache; from splice.spatial_balance import load_spatial_balance_artifact; c=validate_feature_cache(torch.load(sys.argv[1],map_location='cpu',weights_only=True)); a=load_spatial_balance_artifact(sys.argv[2],'waterbirds',c['sample_ids'],c['vocabulary'],c['provenance']); assert a['variant']==sys.argv[3]; print('[OK] spatial rows',len(a['sample_ids']))"
    Invoke-CondaPython -Arguments @("-c", $BalanceValidation, $CachePath, $BalancePath, $VariantName) | Out-Null

    $GraphConfig = [ordered]@{}
    foreach ($Entry in $GraphConfigBase.GetEnumerator()) {
        $GraphConfig[$Entry.Key] = $Entry.Value
    }
    $GraphConfig["spatial_balance_variant"] = $VariantName
    $GraphConfigJson = $GraphConfig | ConvertTo-Json -Compress
    # Preserve JSON quotes through Windows PowerShell 5.1 -> conda.exe.
    $GraphConfigJsonNative = $GraphConfigJson -replace '"', '\"'
    $GraphIdentity = "${SpatialIdentity}_k${ConceptsPerRegion}_null${NullTrials}_floor${FloorToken}_freq${FrequencyToken}"
    $GraphPath = Join-Path $VariantRoot "teacher_graph_${GraphIdentity}.json"

    $GraphIsReusable = $false
    if (Test-Path -LiteralPath $GraphPath) {
        $GraphReuseCheck = "import json,sys; g=json.load(open(sys.argv[1],encoding='utf-8')); e=json.loads(sys.argv[2]); raise SystemExit(0 if g.get('artifact')=='splice_crp_v4_teacher_graph' and g.get('config')==e else 1)"
        $ReuseExit = Invoke-CondaPython -Arguments @(
            "-c", $GraphReuseCheck, $GraphPath, $GraphConfigJsonNative
        ) -AllowFailure -ReturnExitCode
        $GraphIsReusable = $ReuseExit -eq 0
    }
    if ($GraphIsReusable) {
        Write-Host "[REUSE] Matching teacher graph: $GraphPath"
    }
    else {
        Write-Host "=== Building the $VariantName CRPv4 teacher graph ==="
        Invoke-CondaPython -Arguments @(
            "-u", "-m", "splice.crp",
            "--cache", $CachePath,
            "--output", $GraphPath,
            "--seed", "0",
            "--config", $GraphConfigJsonNative,
            "--cobalt", "false",
            "--spatial-balance", "true",
            "--spatial-balance-artifact", $BalancePath
        ) -LogPath (Join-Path $VariantLogRoot "graph_${SpatialIdentity}_k${ConceptsPerRegion}_null$NullTrials.log")
    }

    $GraphValidation = "import json,sys; g=json.load(open(sys.argv[1],encoding='utf-8')); e=json.loads(sys.argv[2]); assert g.get('artifact')=='splice_crp_v4_teacher_graph'; assert g.get('config')==e; d=g['degree_stats']; print('[OK] groups',len(g.get('selected_group_ids',[])),'edges',d['edge_count'],'coverage',d['coverage']); print('[WARN] empty graph: student will use the documented SimCLR fallback') if d['edge_count']==0 else None"
    Invoke-CondaPython -Arguments @(
        "-c", $GraphValidation, $GraphPath, $GraphConfigJsonNative
    ) | Out-Null

    Invoke-CondaPython -Arguments @(
        "-u", "-m", "scripts.tools.summarize_crp_audit", $GraphPath
    ) -AllowFailure
    $AuditPath = Join-Path $VariantRoot "graph_audit_${GraphIdentity}.html"
    if (-not (Test-Path -LiteralPath $AuditPath)) {
        Invoke-CondaPython -Arguments @(
            "-u", "-m", "scripts.tools.render_concept_ablation_examples",
            "--cache", $CachePath,
            "--graph", $GraphPath,
            "--data-folder", $DataRoot,
            "--scope", "selected",
            "--max-interventions", "12",
            "--edges-per-group", "1",
            "--output", $AuditPath
        ) -AllowFailure
    }

    $TrainingLog = Join-Path $VariantLogRoot `
        "ssl_${GraphIdentity}_e${Epochs}_periodic${LinearProbeFrequency}_probe${ProbeEpochs}.log"
    if (Test-FinalProbeLog -Path $TrainingLog) {
        Write-Host "[REUSE] Completed SSL result: $TrainingLog"
    }
    else {
        Write-Host "=== Training $VariantName CRPv4 student ($Epochs epochs, seed 0) ==="
        Invoke-CondaPython -Arguments ($CommonTrainingArguments + @(
            "--checkpoint_dir", $VariantTrainingRoot,
            "--splice_mode", "crp_relational",
            "--splice_weight", "2.0",
            "--crp_teacher_graph", $GraphPath,
            "--crp_temperature", "0.25",
            "--crp_start_epoch", "10",
            "--crp_warmup_epochs", "10",
            "--crp_decay_start_epoch", "0",
            "--crp_decay_end_epoch", "0",
            "--wandb_run_name", "windows_crpv4_${VariantName}_student_s0_e${Epochs}_lp$LinearProbeFrequency",
            "--wandb_tags", "windows_local,crpv4,student,${VariantName},seed_0,openimages_v7,graph_screen_100ep,linear_probe_every_$LinearProbeFrequency"
        )) -LogPath $TrainingLog
    }

    $Results += Read-FinalProbeResult `
        -Variant $VariantName `
        -LogPath $TrainingLog `
        -GraphPath $GraphPath
    $PartialResultsPath = Join-Path $OutputRoot "results_partial.csv"
    $Results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $PartialResultsPath
}

$ResultsPath = Join-Path $OutputRoot "results.csv"
$Results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ResultsPath
Write-Host ""
Write-Host "=== Completed the four CRPv4 ablations ==="
$Results | Format-Table -AutoSize
Write-Host "Results: $ResultsPath"
Write-Host "Artifacts and graph audits: $ArtifactsRoot"
Write-Host "W&B files: $WandbRoot"
