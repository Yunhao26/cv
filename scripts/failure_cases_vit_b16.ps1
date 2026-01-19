$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
cd .. | Out-Null

& .\.venv\Scripts\Activate.ps1

python -m vit_xai_audit.viz.failure_cases_cli `
  --model vit_b16 `
  --metrics_csv .\artifacts\metrics\quantus_vit_b16.csv `
  --npz .\artifacts\attributions\vit_b16\attributions_test_subset.npz `
  --out_dir .\artifacts\figures `
  --k 12

