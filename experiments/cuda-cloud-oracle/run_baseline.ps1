param(
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$Arch = "sm_120",
    [UInt64]$Samples = 1048576,
    [UInt64]$BatchSamples = 65536,
    [UInt64]$Seed = 5999168895323013141
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Build = Join-Path $Here "build"
New-Item -ItemType Directory -Force -Path $Build | Out-Null

if ([string]::IsNullOrWhiteSpace($CudaRoot)) {
    $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
}
$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $Nvcc)) {
    throw "nvcc was not found at $Nvcc"
}

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $VsWhere)) {
    throw "vswhere.exe was not found; install Visual Studio C++ Build Tools"
}
$VsInstall = & $VsWhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ([string]::IsNullOrWhiteSpace($VsInstall)) {
    throw "Visual Studio C++ Build Tools were not found"
}
$VcVars = Join-Path $VsInstall "VC\Auxiliary\Build\vcvars64.bat"
$Source = Join-Path $Here "slab_oracle.cu"
$Exe = Join-Path $Build "slab_oracle.exe"

$Compile = 'call "{0}" >nul && "{1}" -O3 -std=c++17 -arch={2} -lineinfo -Xcompiler=/O2 -o "{3}" "{4}"' -f `
    $VcVars, $Nvcc, $Arch, $Exe, $Source
& cmd.exe /d /s /c $Compile
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

& $Exe --self-test --output (Join-Path $Here "self-test-rtx5090.json")
if ($LASTEXITCODE -ne 0) {
    throw "self-test failed with exit code $LASTEXITCODE"
}

& $Exe --sweep --samples $Samples --batch-samples $BatchSamples --seed $Seed `
    --output (Join-Path $Here "baseline-rtx5090.csv")
if ($LASTEXITCODE -ne 0) {
    throw "baseline sweep failed with exit code $LASTEXITCODE"
}

& $Nvcc --version
Get-FileHash -Algorithm SHA256 -LiteralPath $Source,
    (Join-Path $Here "self-test-rtx5090.json"),
    (Join-Path $Here "baseline-rtx5090.csv") |
    Select-Object Path, Hash
