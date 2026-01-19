param(
  [Parameter(Mandatory=$false)]
  [string]$OutDir = ".\data",

  [Parameter(Mandatory=$false)]
  [string]$Dataset = "paultimothymooney/chest-xray-pneumonia"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}

Write-Host "[download_kaggle] OutDir=$OutDir"
Ensure-Dir $OutDir

$zipPath = Get-ChildItem -Path $OutDir -Filter "*.zip" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $zipPath) {
  # Kaggle token checks (only needed when we actually download)
  $kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
  $kaggleJson = Join-Path $kaggleDir "kaggle.json"
  if (-not (Test-Path $kaggleJson)) {
    throw "Kaggle token not found: $kaggleJson. If you downloaded the dataset manually, place the .zip under $OutDir and rerun."
  }

  try {
    $ver = & kaggle --version
    Write-Host "[download_kaggle] kaggle: $ver"
  } catch {
    throw "kaggle CLI not available. If you downloaded the dataset manually, place the .zip under $OutDir and rerun."
  }

  Write-Host "[download_kaggle] downloading dataset (full archive)..."
  & kaggle datasets download -d $Dataset -p $OutDir
  $zipPath = Get-ChildItem -Path $OutDir -Filter "*.zip" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

if (-not $zipPath) {
  throw "No .zip found under $OutDir after download. Check kaggle CLI output."
}

Write-Host "[download_kaggle] expanding zip..."
Expand-Archive -Path $zipPath.FullName -DestinationPath $OutDir -Force

Write-Host "[download_kaggle] done."
Write-Host "[download_kaggle] expected structure: $OutDir\chest_xray\train|test|val"

