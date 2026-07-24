[CmdletBinding()]
param(
    [ValidateSet("routing", "augmentation", "sweep", "synthesis")]
    [string]$Family = "routing",
    [int[]]$Seeds = @(4),
    [int[]]$Tasks = @(1, 2),
    [ValidateSet("waterbirds", "spur_cifar10")]
    [string]$Dataset = "waterbirds",
    [string]$DataFolder = ".\datasets",
    [string]$PythonExe = "",
    [int]$Epochs = 500,
    [int]$BatchSize = 256,
    [int]$NumWorkers = 4,
    [int]$GpuIndex = 0,
    [int]$CheckpointFrequency = 25,
    [double]$DistillationWeight = 0.1,
    [double]$InterventionStrength = 1.0,
    [string]$WandbProject = "Spur_SpLiCE",
    [string]$WandbEntity = "gsgrechkin-rptu",
    [switch]$NoWandb,
    [switch]$NoResume,
    [switch]$Force,
    [switch]$KeepGoing
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "HomeTraining.Common.ps1")

if ($Epochs -lt 1) { throw "Epochs must be positive." }
if ($BatchSize -lt 1) { throw "BatchSize must be positive." }
if ($NumWorkers -lt 0) { throw "NumWorkers cannot be negative." }
if ($CheckpointFrequency -lt 1) { throw "CheckpointFrequency must be positive." }
if ($DistillationWeight -le 0) { throw "DistillationWeight must be positive." }
if ($InterventionStrength -lt 0 -or $InterventionStrength -gt 2) {
    throw "InterventionStrength must be in [0, 2]."
}

$validTasks = switch ($Family) {
    "routing"      { 0..4 }
    "augmentation" { 0..4 }
    "sweep"        { 0..8 }
    "synthesis" { 0..7 }
}
foreach ($task in $Tasks) {
    if ($task -notin $validTasks) {
        throw "Task $task is invalid for family '$Family'. Valid tasks: $($validTasks -join ', ')."
    }
}

$projectRoot = Get-ProjectRoot -WindowsScriptsDirectory $PSScriptRoot
$python = Get-TrainingPython -ProjectRoot $projectRoot -PythonExe $PythonExe
$dataPath = if ([System.IO.Path]::IsPathRooted($DataFolder)) {
    $DataFolder
} else {
    Join-Path $projectRoot $DataFolder
}
$model = if ($Dataset -eq "spur_cifar10") { "resnet18" } else { "resnet18_large" }
$logsDirectory = Join-Path $projectRoot "outputs\home_logs"
$cacheDirectory = Join-Path $projectRoot "outputs\splice_score_cache"
$familyAutoDirectory = Join-Path $projectRoot "outputs\home_${Dataset}_${Family}_auto"
New-Item -ItemType Directory -Force -Path $logsDirectory, $cacheDirectory, $familyAutoDirectory | Out-Null

if ($Epochs -ne 1000) {
    Write-Warning (
        "This is a shortened $Epochs-epoch protocol. LR milestones will automatically scale to 70%, 80%, and 90%. " +
        "Every run in one comparison must use the same budget; $Epochs-epoch results are not directly " +
        "comparable with the report's 1000-epoch legacy tables."
    )
}

$needsSplice = $Family -eq "augmentation"
if (-not $needsSplice) {
    foreach ($task in $Tasks) {
        if ($task -ne 0) { $needsSplice = $true }
    }
}

$topK = if ($Family -in @("routing", "synthesis")) { 5 } else { 10 }
$discoveryJson = Join-Path $familyAutoDirectory "${Dataset}_splice_concepts.json"
$conceptsFile = Join-Path $familyAutoDirectory "${Dataset}_splice_concepts.concepts.txt"
try {
    if ($needsSplice -and -not (Test-Path -LiteralPath $conceptsFile -PathType Leaf)) {
        $discoveryArguments = @(
            "scripts\tools\discover_splice_spurious_concepts.py",
            "--dataset", $Dataset,
            "--data_folder", $dataPath,
            "--split", "train",
            "--out_path", $discoveryJson,
            "--top_k", [string]$topK,
            "--per_image_top_k", "0",
            "--batch_size", "128",
            "--num_workers", [string][math]::Min($NumWorkers, 2),
            "--device", "cuda",
            "--disable_cudnn",
            "--splice_model", "open_clip:ViT-B-32",
            "--splice_pretrained", "laion2b_s34b_b79k",
            "--splice_vocab", "laion",
            "--splice_vocab_size", "10000",
            "--splice_l1_penalty", "0.25",
            "--splice_score_cache_dir", $cacheDirectory
        )
        if ($Family -in @("routing", "synthesis")) {
            $discoveryArguments += @(
                "--require_consistent_spurious_direction", "true",
                "--deduplicate_concepts", "true"
            )
        }
        Invoke-ManagedPython `
            -PythonExe $python `
            -PythonArguments $discoveryArguments `
            -WorkingDirectory $projectRoot `
            -RunLabel "${Dataset}_${Family}_concept_discovery" `
            -LogDirectory $logsDirectory `
            -GpuIndex $GpuIndex `
            -CpuThreads ([math]::Max(1, $NumWorkers)) | Out-Null
    }

    $concepts = ""
    if ($needsSplice) {
        if (-not (Test-Path -LiteralPath $conceptsFile -PathType Leaf)) {
            throw "Concept discovery did not create $conceptsFile"
        }
        $concepts = (Get-Content -LiteralPath $conceptsFile -Raw).Trim()
        if (-not $concepts) { throw "Concept list is empty: $conceptsFile" }
        Write-Host "[SpLiCE] Concepts: $concepts"
    }

    foreach ($seed in $Seeds) {
        foreach ($task in $Tasks) {
            $runLabel = ""
            $spliceMode = "none"
            $experimentArguments = @()

            if ($Family -eq "routing") {
                switch ($task) {
                    0 { $runLabel = "baseline" }
                    1 { $runLabel = "top5_semantic_q75"; $spliceMode = "augment"; $routingMode = "semantic" }
                    2 { $runLabel = "top5_shuffled_q75"; $spliceMode = "augment"; $routingMode = "shuffled" }
                    3 { $runLabel = "top5_random_q75"; $spliceMode = "augment"; $routingMode = "random" }
                    4 { $runLabel = "top5_all"; $spliceMode = "augment"; $routingMode = "all" }
                }
                if ($spliceMode -eq "augment") {
                    $experimentArguments += @(
                        "--splice_score_threshold", "auto",
                        "--splice_score_quantile", "0.75",
                        "--splice_routing_mode", $routingMode,
                        "--splice_strong_crop", "true",
                        "--splice_strong_color_jitter", "true",
                        "--splice_strong_grayscale_p", "true",
                        "--splice_strong_blur_p", "true",
                        "--splice_auto_require_consistent_direction", "true",
                        "--splice_auto_deduplicate_concepts", "true"
                    )
                }
            } elseif ($Family -eq "augmentation") {
                $spliceMode = "augment"
                switch ($task) {
                    0 {
                        $runLabel = "all"
                        $experimentArguments += @(
                            "--splice_strong_crop", "true",
                            "--splice_strong_color_jitter", "true",
                            "--splice_strong_grayscale_p", "true",
                            "--splice_strong_blur_p", "true"
                        )
                    }
                    1 { $runLabel = "crop"; $experimentArguments += @("--splice_strong_crop", "true") }
                    2 { $runLabel = "color_jitter"; $experimentArguments += @("--splice_strong_color_jitter", "true") }
                    3 { $runLabel = "grayscale"; $experimentArguments += @("--splice_strong_grayscale_p", "true") }
                    4 { $runLabel = "blur"; $experimentArguments += @("--splice_strong_blur_p", "true") }
                }
                $experimentArguments += @(
                    "--splice_score_threshold", "auto",
                    "--splice_score_quantile", "0.75",
                    "--splice_routing_mode", "semantic"
                )
            } elseif ($Family -eq "synthesis") {
                switch ($task) {
                    0 { $runLabel = "baseline" }
                    1 { $runLabel = "original_clip"; $spliceMode = "synthesis_distill"; $intervention = "original" }
                    2 { $runLabel = "class_neutralize"; $spliceMode = "synthesis_distill"; $intervention = "class_neutralize" }
                    3 { $runLabel = "random_coords"; $spliceMode = "synthesis_distill"; $intervention = "random_coords" }
                    4 { $runLabel = "shuffled_donor"; $spliceMode = "synthesis_distill"; $intervention = "shuffled_donor" }
                    5 { $runLabel = "same_class_random"; $spliceMode = "synthesis_distill"; $intervention = "same_class_random_donor" }
                    6 { $runLabel = "zero_out"; $spliceMode = "synthesis_distill"; $intervention = "zero_out" }
                    7 { $runLabel = "oracle_core_swap"; $spliceMode = "synthesis_distill"; $intervention = "core_matched_swap" }
                }
                if ($spliceMode -eq "synthesis_distill") {
                    $experimentArguments += @(
                        "--splice_weight", [string]$DistillationWeight,
                        "--splice_intervention", $intervention,
                        "--splice_intervention_strength", [string]$InterventionStrength
                    )
                }
            } else {
                switch ($task) {
                    0 { $runLabel = "baseline" }
                    1 { $runLabel = "augment_q50"; $spliceMode = "augment"; $quantile = "0.50" }
                    2 { $runLabel = "augment_q75"; $spliceMode = "augment"; $quantile = "0.75" }
                    3 { $runLabel = "augment_q90"; $spliceMode = "augment"; $quantile = "0.90" }
                    4 { $runLabel = "augment_q95"; $spliceMode = "augment"; $quantile = "0.95" }
                    5 { $runLabel = "corr_w0.001"; $spliceMode = "corr_reg"; $weight = "0.001" }
                    6 { $runLabel = "corr_w0.01"; $spliceMode = "corr_reg"; $weight = "0.01" }
                    7 { $runLabel = "corr_w0.1"; $spliceMode = "corr_reg"; $weight = "0.1" }
                    8 { $runLabel = "corr_w1.0"; $spliceMode = "corr_reg"; $weight = "1.0" }
                }
                if ($spliceMode -eq "augment") {
                    $experimentArguments += @(
                        "--splice_score_threshold", "auto",
                        "--splice_score_quantile", $quantile,
                        "--splice_strong_crop", "true",
                        "--splice_strong_color_jitter", "true",
                        "--splice_strong_grayscale_p", "true",
                        "--splice_strong_blur_p", "true"
                    )
                } elseif ($spliceMode -eq "corr_reg") {
                    $experimentArguments += @(
                        "--splice_weight", $weight,
                        "--splice_conditional_on_target", "true"
                    )
                }
            }

            $checkpointRoot = Join-Path $projectRoot "outputs\home_checkpoints\${Family}_seed${seed}_task${task}_e${Epochs}"
            $checkpoint = Get-LatestCheckpoint -CheckpointRoot $checkpointRoot -TargetEpochs $Epochs
            if ($checkpoint -and $checkpoint.Complete -and -not $Force) {
                Write-Host "[SKIP] ${Family}/seed${seed}/task${task} already has last.pth. Use -Force to rerun."
                continue
            }

            $protocolTag = "protocol_e${Epochs}"
            $linearSpuriousProbe = if ($Family -eq "synthesis") { "true" } else { "false" }
            $wandbGroup = "${Dataset}_${Family}_e${Epochs}_seed${seed}"
            $datasetLabel = if ($Dataset -eq "waterbirds") { "Waterbirds" } else { "SpurCIFAR10" }

            # Short task label; the full reproducibility name stays in the checkpoint directory and args.json.
            $taskLabel = switch ($Family) {
                "routing" {
                    switch ($task) {
                        0 { "Baseline" } 1 { "Semantic" } 2 { "Shuffled" }
                        3 { "Random" }   4 { "AugmentAll" }
                    }
                }
                "augmentation" {
                    switch ($task) {
                        0 { "AugAll" }     1 { "Crop" } 2 { "ColorJitter" }
                        3 { "Grayscale" }  4 { "Blur" }
                    }
                }
                "sweep" {
                    switch ($task) {
                        0 { "Baseline" }  1 { "Q50" }      2 { "Q75" }      3 { "Q90" }  4 { "Q95" }
                        5 { "Corr0.001" } 6 { "Corr0.01" } 7 { "Corr0.1" }  8 { "Corr1.0" }
                    }
                }
                "synthesis" {
                    switch ($task) {
                        0 { "Baseline" }      1 { "OrigCLIP" }      2 { "SynNeutralize" }
                        3 { "SynRandCoords" } 4 { "SynShuffDonor" } 5 { "SynSameClass" }
                        6 { "SynZeroOut" }    7 { "SynOracleSwap" }
                    }
                }
            }

            # Invariant culture keeps decimal points out of locale-dependent comma form.
            $invariant = [System.Globalization.CultureInfo]::InvariantCulture
            $weightText = $DistillationWeight.ToString($invariant)
            $strengthText = $InterventionStrength.ToString($invariant)

            $hyperTag = ""
            if ($spliceMode -eq "synthesis_distill") {
                # Alpha is inert for the unedited-teacher control, so it is omitted there.
                $hyperTag = if ($task -eq 1) { "_w${weightText}" } else { "_w${weightText}a${strengthText}" }
                $wandbTags = "dataset_${Dataset},seed_${seed},family_${Family},task_${runLabel}," +
                    "machine_home_rtx5080,${protocolTag},lambda_${weightText},alpha_${strengthText}"
            } else {
                $wandbTags = "dataset_${Dataset},seed_${seed},family_${Family},task_${runLabel}," +
                    "machine_home_rtx5080,${protocolTag}"
            }
            $wandbRunName = "${datasetLabel}_S${seed}_${taskLabel}${hyperTag}_e${Epochs}"
            $arguments = @(
                "spur_splice.py",
                "--dataset", $Dataset,
                "--data_folder", $dataPath,
                "--model", $model,
                "--num_workers", [string]$NumWorkers,
                "--epochs", [string]$Epochs,
                "--batch_size", [string]$BatchSize,
                "--seed", [string]$seed,
                "--temp", "0.05",
                "--learning_rate", "0.01",
                "--lr_decay_epochs", "auto",
                "--weight_decay", "1e-4",
                "--print_freq", "25",
                "--rank_eval_freq", "0",
                "--train_set_linear_layer", "ds_train",
                "--linear_eval_split", "val",
                "--linear_probe_mode", "periodic",
                "--linear_probe_freq", "25",
                "--linear_probe_epochs", "100",
                "--linear_lr_decay_epochs", "auto",
                "--linear_spurious_probe", $linearSpuriousProbe,
                "--amp", "true",
                "--channels_last", "true",
                "--cudnn_enabled", "true",
                "--splice_mode", $spliceMode,
                "--checkpoint_dir", $checkpointRoot,
                "--keep_checkpoints",
                "--save_freq", [string]$CheckpointFrequency,
                "--wandb_name", $WandbProject,
                "--wandb_run_name", $wandbRunName,
                "--entity", $WandbEntity,
                "--wandb_group", $wandbGroup,
                "--wandb_tags", $wandbTags
            )
            if (-not $NoWandb) {
                $arguments += "--use_wandb"
            }
            if ($spliceMode -ne "none") {
                $arguments += @(
                    "--splice_concepts", $concepts,
                    "--splice_auto_top_k", [string]$topK,
                    "--splice_auto_out_dir", $familyAutoDirectory,
                    "--splice_score_reduction", "max",
                    "--splice_batch_size", "128",
                    "--splice_num_workers", [string][math]::Min($NumWorkers, 2),
                    "--splice_l1_penalty", "0.25",
                    "--splice_model", "open_clip:ViT-B-32",
                    "--splice_pretrained", "laion2b_s34b_b79k",
                    "--splice_vocab", "laion",
                    "--splice_vocab_size", "10000",
                    "--splice_score_cache_dir", $cacheDirectory
                )
            }
            $arguments += $experimentArguments
            if ($checkpoint -and -not $NoResume) {
                Write-Host "[RESUME] epoch $($checkpoint.Epoch): $($checkpoint.Path)"
                $arguments += @("--resume", $checkpoint.Path)
            }

            try {
                Invoke-ManagedPython `
                    -PythonExe $python `
                    -PythonArguments $arguments `
                    -WorkingDirectory $projectRoot `
                    -RunLabel "${Dataset}_${Family}_seed${seed}_task${task}_${runLabel}_e${Epochs}" `
                    -LogDirectory $logsDirectory `
                    -GpuIndex $GpuIndex `
                    -CpuThreads ([math]::Max(1, $NumWorkers)) | Out-Null
            } catch {
                Write-Error -ErrorAction Continue $_
                if (-not $KeepGoing) {
                    throw
                }
            }
        }
    }
} finally {
    # Reserved for future queue-level cleanup.
}
