param(
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$Arch = "sm_120",
    [switch]$RunGrid,
    [switch]$RepeatGrid
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Build = Join-Path $Here "build"
$Baseline = Join-Path $Here "run_baseline.ps1"

if ([string]::IsNullOrWhiteSpace($CudaRoot)) {
    $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
}

& $Baseline -CudaRoot $CudaRoot -Arch $Arch
if ($LASTEXITCODE -ne 0) {
    throw "baseline validation failed with exit code $LASTEXITCODE"
}

$Oracle = Join-Path $Build "slab_oracle.exe"
$Sanitizer = Join-Path $CudaRoot "bin\compute-sanitizer.bat"
if (-not (Test-Path -LiteralPath $Sanitizer -PathType Leaf)) {
    throw "compute-sanitizer was not found at $Sanitizer"
}

$Stage2Arguments = @(
    "--backend", "gpu",
    "--format", "json",
    "--case", "stage2_sanitizer",
    "--tau", "3",
    "--ssa", ".97",
    "--phase-model", "mixture-hg",
    "--phase-lobe1-g", ".85",
    "--phase-lobe2-g", "-.15",
    "--phase-lobe1-weight", ".9",
    "--phase-first-moment", ".75",
    "--surface-albedo", ".6",
    "--lower-boundary", "lambertian",
    "--sun-zenith-deg", "65",
    "--view-zenith-deg", "65",
    "--relative-azimuth-deg", "180",
    "--samples", "32768",
    "--seed", "5999168895323013142",
    "--max-scatters", "384",
    "--batch-samples", "16384",
    "--report-forward-flux", "true",
    "--depth-bins", "32"
)

foreach ($Tool in @("memcheck", "racecheck", "initcheck")) {
    $Output = Join-Path $Build ("stage2-sanitizer-{0}.json" -f $Tool)
    & $Sanitizer --tool $Tool --error-exitcode 99 $Oracle @Stage2Arguments --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "compute-sanitizer $Tool failed with exit code $LASTEXITCODE"
    }
}

if ($RunGrid) {
    $GridArguments = @(
        (Join-Path $Here "run_stage2_grid.py"),
        "--oracle", $Oracle,
        "--force"
    )
    if ($RepeatGrid) {
        $GridArguments += "--repeat"
    }
    & python @GridArguments
    $GridExit = $LASTEXITCODE
    $GridSummaryPath = Join-Path $Here "stage2-results-rtx5090-v1-summary.json"
    $GridSummary = Get-Content -LiteralPath $GridSummaryPath -Raw | ConvertFrom-Json
    if ($GridExit -ne 2 -or $GridSummary.all_checks_passed) {
        throw "immutable Stage-2 grid did not reproduce its recorded order-cap blocker"
    }

    & python (Join-Path $Here "run_stage2_remediation.py") --oracle $Oracle --force
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-2 4096-order remediation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Stage-2 CUDA oracle validation passed"
