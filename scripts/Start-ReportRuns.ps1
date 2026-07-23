[CmdletBinding()]
param(
    [string]$DataFolder = ".\datasets",
    [string]$PythonExe = "",
    [int]$Epochs = 1000,
    [int]$NumWorkers = 4,
    [switch]$NoWandb,
    [switch]$KeepGoing
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "Run-HomeExperiments.ps1"

# Highest-information-first queue for the current report:
# 1) complete the third semantic-vs-shuffled pair;
# 2) add a same-RTX-5080 baseline and matched-random control for seed 4;
# 3) complete the remaining random control;
# 4) finish the less diagnostic augment-all controls;
# 5) add missing component-ablation seeds only if time remains.
$queue = @(
    [pscustomobject]@{ Family = "routing"; Seed = 4; Task = 1; Reason = "third semantic run" },
    [pscustomobject]@{ Family = "routing"; Seed = 4; Task = 2; Reason = "third matched shuffled run" },
    [pscustomobject]@{ Family = "routing"; Seed = 4; Task = 0; Reason = "same-hardware baseline for seed 4" },
    [pscustomobject]@{ Family = "routing"; Seed = 4; Task = 3; Reason = "third matched-random run" },
    [pscustomobject]@{ Family = "routing"; Seed = 3; Task = 3; Reason = "second matched-random run" },
    [pscustomobject]@{ Family = "routing"; Seed = 1; Task = 4; Reason = "augment-all control" },
    [pscustomobject]@{ Family = "routing"; Seed = 3; Task = 4; Reason = "augment-all control" },
    [pscustomobject]@{ Family = "routing"; Seed = 4; Task = 4; Reason = "augment-all control" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 3; Task = 1; Reason = "crop ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 4; Task = 1; Reason = "crop ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 3; Task = 2; Reason = "color-jitter ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 4; Task = 2; Reason = "color-jitter ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 3; Task = 3; Reason = "grayscale ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 4; Task = 3; Reason = "grayscale ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 3; Task = 4; Reason = "blur ablation" },
    [pscustomobject]@{ Family = "augmentation"; Seed = 4; Task = 4; Reason = "blur ablation" }
)

Write-Host "Report queue contains $($queue.Count) sequential runs."
Write-Host "The first two runs are the priority pair. Stop with Ctrl+C after the completed runs you need."

foreach ($item in $queue) {
    Write-Host ""
    Write-Host "=== $($item.Family), seed $($item.Seed), task $($item.Task): $($item.Reason) ==="
    $arguments = @{
        Family             = $item.Family
        Seeds              = @($item.Seed)
        Tasks              = @($item.Task)
        DataFolder         = $DataFolder
        PythonExe          = $PythonExe
        Epochs             = $Epochs
        NumWorkers         = $NumWorkers
        KeepGoing          = $KeepGoing
    }
    if ($NoWandb) {
        $arguments.NoWandb = $true
    }
    & $runner @arguments
    if ($LASTEXITCODE -ne 0 -and -not $KeepGoing) {
        throw "Queue stopped after a failed run."
    }
}
