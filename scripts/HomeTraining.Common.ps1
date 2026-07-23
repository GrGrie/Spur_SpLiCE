Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    param([Parameter(Mandatory = $true)][string]$WindowsScriptsDirectory)
    return (Resolve-Path (Join-Path $WindowsScriptsDirectory "..")).Path
}

function Get-TrainingPython {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [string]$PythonExe,
        [string]$CondaEnvironment = "grgrie-train"
    )

    if ($PythonExe) {
        if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
            throw "Python executable does not exist: $PythonExe"
        }
        return (Resolve-Path -LiteralPath $PythonExe).Path
    }

    $activeCondaPython = $null
    if ($env:CONDA_PREFIX) {
        $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $activeCondaPython = (Resolve-Path -LiteralPath $candidate).Path
        }
        if ($activeCondaPython -and $env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne "base") {
            Write-Host "[ENV] Using active Conda environment: $env:CONDA_DEFAULT_ENV"
            return $activeCondaPython
        }
    }

    $conda = Get-Command "conda.exe" -ErrorAction SilentlyContinue
    if (-not $conda) {
        $conda = Get-Command "conda" -ErrorAction SilentlyContinue
    }
    if ($conda) {
        try {
            $environmentList = (& $conda.Source env list --json 2>$null | Out-String | ConvertFrom-Json).envs
            $matchingEnvironment = $environmentList |
                Where-Object { (Split-Path $_ -Leaf) -eq $CondaEnvironment } |
                Select-Object -First 1
            if ($matchingEnvironment) {
                $condaPython = Join-Path $matchingEnvironment "python.exe"
                if (Test-Path -LiteralPath $condaPython -PathType Leaf) {
                    Write-Host "[ENV] Using Conda environment: $CondaEnvironment"
                    return (Resolve-Path -LiteralPath $condaPython).Path
                }
            }
        } catch {
            Write-Warning "Could not inspect Conda environments: $($_.Exception.Message)"
        }
    }

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        Write-Host "[ENV] Using project virtual environment: $venvPython"
        return (Resolve-Path -LiteralPath $venvPython).Path
    }
    if ($activeCondaPython) {
        Write-Warning "Falling back to the active base Conda environment."
        return $activeCondaPython
    }
    throw (
        "Neither Conda environment '$CondaEnvironment' nor $venvPython was found. " +
        "Activate the Conda environment or run scripts\Setup-HomeTraining.ps1."
    )
}

function Get-NvidiaSmiPath {
    $command = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $standardPath = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    if (Test-Path -LiteralPath $standardPath -PathType Leaf) {
        return $standardPath
    }
    throw "nvidia-smi.exe was not found. Install/update the NVIDIA driver first."
}

function ConvertTo-ProcessArgumentString {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)

    $quoted = foreach ($argument in $ArgumentList) {
        if ($null -eq $argument) {
            '""'
            continue
        }
        $value = [string]$argument
        if ($value -notmatch '[\s"]') {
            $value
            continue
        }

        # Windows CommandLineToArgvW-compatible escaping for quoted arguments.
        $builder = New-Object System.Text.StringBuilder
        [void]$builder.Append('"')
        $backslashes = 0
        foreach ($character in $value.ToCharArray()) {
            if ($character -eq '\') {
                $backslashes++
                continue
            }
            if ($character -eq '"') {
                [void]$builder.Append(('\' * (($backslashes * 2) + 1)))
                [void]$builder.Append('"')
                $backslashes = 0
                continue
            }
            if ($backslashes -gt 0) {
                [void]$builder.Append(('\' * $backslashes))
                $backslashes = 0
            }
            [void]$builder.Append($character)
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * ($backslashes * 2)))
        }
        [void]$builder.Append('"')
        $builder.ToString()
    }
    return ($quoted -join " ")
}

function Get-LatestCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [int]$TargetEpochs
    )
    if (-not (Test-Path -LiteralPath $CheckpointRoot -PathType Container)) {
        return $null
    }

    $last = Get-ChildItem -LiteralPath $CheckpointRoot -Filter "last.pth" -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($last) {
        return [pscustomobject]@{ Path = $last.FullName; Epoch = $TargetEpochs; Complete = $true }
    }

    $candidates = foreach ($file in Get-ChildItem -LiteralPath $CheckpointRoot -Filter "epoch_*.pth" -File -Recurse -ErrorAction SilentlyContinue) {
        if ($file.BaseName -match '^epoch_(\d+)$') {
            [pscustomobject]@{ Path = $file.FullName; Epoch = [int]$Matches[1]; Complete = $false }
        }
    }
    return $candidates | Sort-Object Epoch -Descending | Select-Object -First 1
}

function Invoke-ManagedPython {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string[]]$PythonArguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$RunLabel,
        [Parameter(Mandatory = $true)][string]$LogDirectory,
        [int]$GpuIndex = 0,
        [int]$CpuThreads = 4
    )

    New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    $safeLabel = $RunLabel -replace '[^A-Za-z0-9_.-]', '_'
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $LogDirectory "${timestamp}_${safeLabel}.out.log"
    $stderrPath = Join-Path $LogDirectory "${timestamp}_${safeLabel}.err.log"

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $PythonExe
    $startInfo.Arguments = ConvertTo-ProcessArgumentString $PythonArguments
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    Write-Host "[RUN] $RunLabel"
    Write-Host "[LOG] $stdoutPath"

    # ProcessStartInfo.EnvironmentVariables behaves inconsistently under
    # Windows PowerShell 5.1 + StrictMode. Temporarily set the parent process
    # values, start the child (which snapshots them), and immediately restore.
    $childEnvironment = @{
        "PYTHONUNBUFFERED" = "1"
        "CUDA_VISIBLE_DEVICES" = [string]$GpuIndex
        "OMP_NUM_THREADS" = [string]$CpuThreads
        "MKL_NUM_THREADS" = [string]$CpuThreads
    }
    $previousEnvironment = @{}
    foreach ($name in $childEnvironment.Keys) {
        $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        [Environment]::SetEnvironmentVariable($name, $childEnvironment[$name], "Process")
    }
    try {
        if (-not $process.Start()) {
            throw "Failed to start Python for $RunLabel"
        }
    } finally {
        foreach ($name in $childEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], "Process")
        }
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $stdoutWriter = New-Object System.IO.StreamWriter($stdoutPath, $false, $utf8NoBom)
    $stderrWriter = New-Object System.IO.StreamWriter($stderrPath, $false, $utf8NoBom)
    $stdoutWriter.AutoFlush = $true
    $stderrWriter.AutoFlush = $true
    $stdoutTask = $process.StandardOutput.ReadLineAsync()
    $stderrTask = $process.StandardError.ReadLineAsync()
    $exitCode = $null

    try {
        while ($true) {
            if ($null -ne $stdoutTask -and $stdoutTask.IsCompleted) {
                $line = $stdoutTask.GetAwaiter().GetResult()
                if ($null -eq $line) {
                    $stdoutTask = $null
                } else {
                    $stdoutWriter.WriteLine($line)
                    $stdoutTask = $process.StandardOutput.ReadLineAsync()
                }
            }
            if ($null -ne $stderrTask -and $stderrTask.IsCompleted) {
                $line = $stderrTask.GetAwaiter().GetResult()
                if ($null -eq $line) {
                    $stderrTask = $null
                } else {
                    $stderrWriter.WriteLine($line)
                    $stderrTask = $process.StandardError.ReadLineAsync()
                }
            }

            if ($process.HasExited -and $null -eq $stdoutTask -and $null -eq $stderrTask) {
                break
            }

            Start-Sleep -Milliseconds 100
        }
        $process.WaitForExit()
        $exitCode = $process.ExitCode
    } finally {
        if (-not $process.HasExited) {
            & taskkill.exe "/PID" $process.Id "/T" "/F" 2>$null | Out-Null
            $process.WaitForExit()
        }
        $stdoutWriter.Dispose()
        $stderrWriter.Dispose()
        $process.Dispose()
    }

    if ($exitCode -ne 0) {
        Write-Host "[ERROR] Last stderr lines:"
        Get-Content -LiteralPath $stderrPath -Tail 60 -ErrorAction SilentlyContinue
        Write-Host "[ERROR] Last stdout lines:"
        Get-Content -LiteralPath $stdoutPath -Tail 60 -ErrorAction SilentlyContinue
        throw "Python exited with code $exitCode during $RunLabel"
    }
    Write-Host "[DONE] $RunLabel"
    return [pscustomobject]@{
        ExitCode      = $exitCode
        StdoutPath    = $stdoutPath
        StderrPath    = $stderrPath
    }
}
