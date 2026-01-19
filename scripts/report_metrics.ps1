$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
cd .. | Out-Null

& .\.venv\Scripts\Activate.ps1

python -m vit_xai_audit.report.summarize_metrics_cli `
  --metrics_dir .\artifacts\metrics `
  --out_dir .\artifacts\figures\summary `
  --seed 42

