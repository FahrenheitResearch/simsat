param(
    [string]$Oracle = ".\experiments\cuda-cloud-oracle\build\slab_oracle.exe",
    [string]$Grid = ".\experiments\cloud-closure-fit\fixtures\stage1-request-grid-v1.csv",
    [string]$OutputDir = ".\experiments\cloud-closure-fit\requested-results"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Oracle -PathType Leaf)) {
    throw "CUDA oracle executable not found: $Oracle"
}
if (-not (Test-Path -LiteralPath $Grid -PathType Leaf)) {
    throw "Request grid not found: $Grid"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Rows = Import-Csv -LiteralPath $Grid

foreach ($Row in $Rows) {
    $Output = Join-Path $OutputDir ($Row.case + ".csv")
    $Arguments = @(
        "--backend", "gpu",
        "--format", "csv",
        "--output", $Output,
        "--case", $Row.case,
        "--tau", $Row.tau,
        "--ssa", $Row.ssa,
        "--g", $Row.hg_g,
        "--sun-zenith-deg", $Row.sun_zenith_deg,
        "--view-zenith-deg", $Row.view_zenith_deg,
        "--relative-azimuth-deg", $Row.relative_azimuth_deg,
        "--samples", $Row.samples,
        "--seed", $Row.seed,
        "--max-scatters", $Row.max_scatters,
        "--batch-samples", $Row.batch_samples
    )
    & $Oracle @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA oracle failed for $($Row.case) with exit code $LASTEXITCODE"
    }
}

Write-Host "Completed $($Rows.Count) requested cases in $OutputDir"
