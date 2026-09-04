[CmdletBinding()]
param(
    [string]$CondaEnvironment = "grgrie-train",
    [string]$DataRoot = "",
    [string]$OutputRoot = "",
    [ValidateRange(1, 500)]
    [int]$Epochs = 50,
    [ValidateRange(1, 200)]
    [int]$ProbeEpochs = 30,
    [ValidateRange(1, 1024)]
    [int]$BatchSize = 128,
    [ValidateRange(0, 32)]
    [int]$Workers = 4,
    [ValidateSet("offline", "online")]
    [string]$WandbMode = "offline",
    [switch]$IncludeCobalt,
    [ValidateRange(1, 200)]
    [int]$CobaltEpochs = 50
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $RepoRoot "datasets"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\windows_splice_only_ablation"
}
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

function Find-CondaExecutable {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    throw "conda.exe was not found. Add Conda to PATH or pass through an Anaconda/Miniconda terminal."
}

$CondaExe = Find-CondaExecutable

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$LogPath = "",
        [switch]$AllowFailure
    )
    $condaArguments = @("run", "--no-capture-output", "-n", $CondaEnvironment, "python") + $Arguments
    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        & $CondaExe @condaArguments
    }
    else {
        & $CondaExe @condaArguments 2>&1 | Tee-Object -FilePath $LogPath
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "Python stage failed with exit code $exitCode."
    }
}

New-Item -ItemType Directory -Force -Path $DataRoot, $OutputRoot | Out-Null
$LogsRoot = Join-Path $OutputRoot "logs"
$GraphRoot = Join-Path $OutputRoot "graph_g2_t070_c020_k12"
$TrainingRoot = Join-Path $OutputRoot "training"
$WandbRoot = Join-Path $OutputRoot "wandb"
New-Item -ItemType Directory -Force -Path $LogsRoot, $GraphRoot, $TrainingRoot, $WandbRoot | Out-Null

$env:PYTHONUTF8 = "1"
$env:WANDB_MODE = $WandbMode
$env:WANDB_DIR = $WandbRoot
Set-Location $RepoRoot

Write-Host "=== Checking the Conda environment and RTX GPU ==="
Invoke-CondaPython -Arguments @(
    "-c",
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.randn(64,64,device='cuda'); y=x@x; torch.cuda.synchronize(); print('PyTorch',torch.__version__,'GPU',torch.cuda.get_device_name(0))"
)

Write-Host "=== Ensuring the Waterbirds download dependency is present ==="
& $CondaExe run --no-capture-output -n $CondaEnvironment python -c "import datasets" *> $null
if ($LASTEXITCODE -ne 0) {
    & $CondaExe run --no-capture-output -n $CondaEnvironment python -m pip install "datasets>=3,<5"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the optional Waterbirds download dependency."
    }
}
Invoke-CondaPython -Arguments @(
    "-u", "scripts/tools/download_waterbirds_hf.py",
    "--output-root", $DataRoot
)
Invoke-CondaPython -Arguments @(
    "-u", "scripts/download_openimages_vocabulary.py",
    "--download-root", (Join-Path $RepoRoot "data")
)

Write-Host "=== Building or validating the frozen SpLiCE cache ==="
$CachePath = Join-Path $OutputRoot "waterbirds_train_features_oi_v7.pt"
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
$CacheValidation = "import sys,torch; from splice.crp import validate_feature_cache; c=validate_feature_cache(torch.load(sys.argv[1],map_location='cpu',weights_only=True)); p=c['provenance']; assert p['splice_vocab']=='openimages_v7' and p['splice_vocab_size']==-1; print('[OK] cache samples',len(c['sample_ids']))"
Invoke-CondaPython -Arguments @("-c", $CacheValidation, $CachePath)

Write-Host "=== Building the SpLiCE-only teacher graph ==="
$GraphPath = Join-Path $GraphRoot "teacher_graph.json"
$GraphConfig = [ordered]@{
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
    null_trials = 16
    null_quantile = 0.95
    similarity_chunk_size = 512
    orthogonal_tolerance = 0.000001
    use_residual_splice_gate = $true
    residual_splice_similarity_threshold = 0.25
    use_cobalt_confidence = $false
    seed = 0
    cobalt = $false
}
$GraphConfigJson = $GraphConfig | ConvertTo-Json -Compress
if (-not (Test-Path -LiteralPath $GraphPath)) {
    Invoke-CondaPython -Arguments @(
        "-u", "-m", "splice.crp",
        "--cache", $CachePath,
        "--output", $GraphPath,
        "--seed", "0",
        "--config", $GraphConfigJson,
        "--cobalt", "false"
    ) -LogPath (Join-Path $LogsRoot "graph.log")
}
$GraphValidation = "import json,sys; g=json.load(open(sys.argv[1],encoding='utf-8')); expected=json.loads(sys.argv[2]); assert g.get('config')==expected, 'existing graph config differs'; assert g.get('degree_stats',{}).get('edge_count',0)>0, 'teacher graph has no edges'; print('[OK] graph groups',len(g.get('selected_group_ids',[])),'edges',g['degree_stats']['edge_count'],'coverage',g['degree_stats']['coverage'])"
Invoke-CondaPython -Arguments @("-c", $GraphValidation, $GraphPath, $GraphConfigJson)
Invoke-CondaPython -Arguments @(
    "-u", "-m", "scripts.tools.summarize_crp_audit", $GraphPath
) -AllowFailure
Invoke-CondaPython -Arguments @(
    "-u", "-m", "scripts.tools.render_concept_ablation_examples",
    "--cache", $CachePath,
    "--graph", $GraphPath,
    "--data-folder", $DataRoot,
    "--scope", "selected",
    "--max-interventions", "12",
    "--edges-per-group", "1",
    "--output", (Join-Path $GraphRoot "graph_audit.html")
)

$CommonTrainingArguments = @(
    "-u", "spur_splice.py",
    "--dataset", "waterbirds",
    "--data_folder", $DataRoot,
    "--model", "resnet18_large",
    "--seed", "0",
    "--trial", "windows_quick",
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
    "--linear_probe_mode", "final",
    "--linear_probe_freq", "0",
    "--linear_probe_epochs", $ProbeEpochs.ToString(),
    "--linear_lr_decay_epochs", "auto",
    "--linear_spurious_probe", "false",
    "--checkpoint_dir", $TrainingRoot,
    "--save_freq", $Epochs.ToString(),
    "--checkpoint_keep_count", "1",
    "--delete_checkpoints_after_training", "false",
    "--keep_checkpoints",
    "--use_wandb",
    "--wandb_name", "Spur_SpLiCE_windows_screen",
    "--wandb_group", "waterbirds_splice_cobalt_ablation_windows_seed0_e$Epochs"
)

function Test-CompletedProbeLog {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    return $null -ne (Select-String -Path $Path -Pattern "best accuracy: ([0-9.]+) and worst-group accuracy: ([0-9.]+) and best-group accuracy: ([0-9.]+)" | Select-Object -Last 1)
}

Write-Host "=== Training the matched SimCLR control ==="
$BaselineLog = Join-Path $LogsRoot "simclr_e${Epochs}_p${ProbeEpochs}.log"
if (Test-CompletedProbeLog $BaselineLog) {
    Write-Host "[INFO] Reusing completed SimCLR result from $BaselineLog"
}
else {
    Invoke-CondaPython -Arguments ($CommonTrainingArguments + @(
        "--splice_mode", "none",
        "--wandb_run_name", "windows_simclr_seed0_e$Epochs",
        "--wandb_tags", "windows_local,screen,seed_0,openimages_v7,baseline"
    )) -LogPath $BaselineLog
}

Write-Host "=== Training SpLiCE-only relational SSL ==="
$SpliceLog = Join-Path $LogsRoot "splice_only_crp_e${Epochs}_p${ProbeEpochs}.log"
if (Test-CompletedProbeLog $SpliceLog) {
    Write-Host "[INFO] Reusing completed SpLiCE-only result from $SpliceLog"
}
else {
    Invoke-CondaPython -Arguments ($CommonTrainingArguments + @(
        "--splice_mode", "crp_relational",
        "--splice_weight", "2.0",
        "--crp_teacher_graph", $GraphPath,
        "--crp_temperature", "0.25",
        "--crp_start_epoch", "5",
        "--crp_warmup_epochs", "5",
        "--crp_decay_start_epoch", "0",
        "--crp_decay_end_epoch", "0",
        "--wandb_run_name", "windows_splice_only_crp_seed0_e$Epochs",
        "--wandb_tags", "windows_local,screen,seed_0,openimages_v7,splice_only,no_cobalt"
    )) -LogPath $SpliceLog
}

$CobaltLog = Join-Path $LogsRoot "cobalt_crp_e${Epochs}_p${ProbeEpochs}_discovery${CobaltEpochs}.log"
$CobaltGraphRoot = Join-Path $OutputRoot "graph_g2_t070_c020_k12_cobalt_e$CobaltEpochs"
if ($IncludeCobalt) {
    Write-Host "=== Training or reusing CoBalT Stage 1 ==="
    $CobaltRoot = Join-Path $OutputRoot "cobalt_e$CobaltEpochs"
    New-Item -ItemType Directory -Force -Path $CobaltRoot, $CobaltGraphRoot | Out-Null
    $CobaltCheckpoint = Join-Path $CobaltRoot "discovery.pt"
    $CobaltConcepts = Join-Path $CobaltRoot "concepts.pt"
    if (-not (Test-Path -LiteralPath $CobaltCheckpoint)) {
        Invoke-CondaPython -Arguments @(
            "-u", "-m", "CoBalT.train_discovery",
            "--dataset", "waterbirds",
            "--data-root", $DataRoot,
            "--output", $CobaltCheckpoint,
            "--seed", "0",
            "--epochs", $CobaltEpochs.ToString(),
            "--batch-size", $BatchSize.ToString(),
            "--workers", $Workers.ToString(),
            "--image-size", "224",
            "--crop-min", "0.2",
            "--num-slots", "4",
            "--codebook-size", "8",
            "--slot-dim", "32",
            "--hidden-dim", "1024",
            "--learning-rate", "0.0002",
            "--weight-decay", "0.0005",
            "--student-temperature", "0.1",
            "--teacher-temperature", "0.07",
            "--contrastive-temperature", "0.2",
            "--teacher-momentum", "0.99",
            "--codebook-momentum", "0.9",
            "--center-momentum", "0.9",
            "--backbone", "resnet18",
            "--no-pretrained",
            "--allow-nonpaper-backbone",
            "--amp",
            "--wandb-project", "Spur_SpLiCE_windows_screen",
            "--wandb-run-name", "windows_cobalt_discovery_seed0_e$CobaltEpochs",
            "--wandb-group", "waterbirds_splice_cobalt_ablation_windows_seed0_e$Epochs",
            "--wandb-tags", "windows_local,screen,seed_0,cobalt_discovery"
        ) -LogPath (Join-Path $LogsRoot "cobalt_discovery.log")
    }
    if (-not (Test-Path -LiteralPath $CobaltConcepts)) {
        Invoke-CondaPython -Arguments @(
            "-u", "-m", "CoBalT.extract_concepts",
            "--checkpoint", $CobaltCheckpoint,
            "--data-root", $DataRoot,
            "--output", $CobaltConcepts,
            "--batch-size", $BatchSize.ToString(),
            "--workers", $Workers.ToString(),
            "--image-size", "224"
        ) -LogPath (Join-Path $LogsRoot "cobalt_extract.log")
    }

    Write-Host "=== Building the matched CoBalT-balanced teacher graph ==="
    $CobaltGraphConfig = [ordered]@{}
    foreach ($entry in $GraphConfig.GetEnumerator()) {
        $CobaltGraphConfig[$entry.Key] = $entry.Value
    }
    $CobaltGraphConfig["use_cobalt_confidence"] = $true
    $CobaltGraphConfig["cobalt"] = $true
    $CobaltGraphConfigJson = $CobaltGraphConfig | ConvertTo-Json -Compress
    $CobaltGraphPath = Join-Path $CobaltGraphRoot "teacher_graph.json"
    if (-not (Test-Path -LiteralPath $CobaltGraphPath)) {
        Invoke-CondaPython -Arguments @(
            "-u", "-m", "splice.crp",
            "--cache", $CachePath,
            "--output", $CobaltGraphPath,
            "--seed", "0",
            "--config", $CobaltGraphConfigJson,
            "--cobalt", "true",
            "--cobalt-concepts", $CobaltConcepts
        ) -LogPath (Join-Path $LogsRoot "cobalt_graph.log")
    }
    Invoke-CondaPython -Arguments @("-c", $GraphValidation, $CobaltGraphPath, $CobaltGraphConfigJson)
    Invoke-CondaPython -Arguments @(
        "-u", "-m", "scripts.tools.summarize_crp_audit", $CobaltGraphPath
    ) -AllowFailure
    Invoke-CondaPython -Arguments @(
        "-u", "-m", "scripts.tools.render_concept_ablation_examples",
        "--cache", $CachePath,
        "--graph", $CobaltGraphPath,
        "--data-folder", $DataRoot,
        "--scope", "selected",
        "--max-interventions", "12",
        "--edges-per-group", "1",
        "--output", (Join-Path $CobaltGraphRoot "graph_audit.html")
    )

    Write-Host "=== Training matched CoBalT-balanced CRPv3 SSL ==="
    if (Test-CompletedProbeLog $CobaltLog) {
        Write-Host "[INFO] Reusing completed CoBalT CRP result from $CobaltLog"
    }
    else {
        Invoke-CondaPython -Arguments ($CommonTrainingArguments + @(
            "--splice_mode", "crp_relational",
            "--splice_weight", "2.0",
            "--crp_teacher_graph", $CobaltGraphPath,
            "--crp_temperature", "0.25",
            "--crp_start_epoch", "5",
            "--crp_warmup_epochs", "5",
            "--crp_decay_start_epoch", "0",
            "--crp_decay_end_epoch", "0",
            "--wandb_run_name", "windows_cobalt_crpv3_seed0_e$Epochs",
            "--wandb_tags", "windows_local,screen,seed_0,openimages_v7,cobalt,crpv3"
        )) -LogPath $CobaltLog
    }
}

function Read-ProbeResult {
    param([string]$Name, [string]$Path)
    $match = Select-String -Path $Path -Pattern "best accuracy: ([0-9.]+) and worst-group accuracy: ([0-9.]+) and best-group accuracy: ([0-9.]+)" | Select-Object -Last 1
    if ($null -eq $match -or -not $match.Matches[0].Success) {
        throw "Could not find final probe metrics in $Path"
    }
    return [pscustomobject]@{
        run = $Name
        validation_accuracy = [double]::Parse($match.Matches[0].Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        validation_worst_group_accuracy = [double]::Parse($match.Matches[0].Groups[2].Value, [Globalization.CultureInfo]::InvariantCulture)
        validation_best_group_accuracy = [double]::Parse($match.Matches[0].Groups[3].Value, [Globalization.CultureInfo]::InvariantCulture)
    }
}

$Results = @(
    Read-ProbeResult -Name "simclr" -Path $BaselineLog
    Read-ProbeResult -Name "splice_only_crp" -Path $SpliceLog
)
if ($IncludeCobalt) {
    $Results += Read-ProbeResult -Name "cobalt_crpv3" -Path $CobaltLog
}
$ResultsPath = Join-Path $OutputRoot "results.csv"
$Results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ResultsPath
Write-Host ""
Write-Host "=== Completed quick SpLiCE-only ablation ==="
$Results | Format-Table -AutoSize
Write-Host "Results: $ResultsPath"
Write-Host "Graph audit: $(Join-Path $GraphRoot 'graph_audit.html')"
if ($IncludeCobalt) {
    Write-Host "CoBalT graph audit: $(Join-Path $CobaltGraphRoot 'graph_audit.html')"
}
Write-Host "Offline/online W&B files: $WandbRoot"
