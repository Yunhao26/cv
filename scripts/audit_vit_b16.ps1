$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
cd .. | Out-Null

& .\.venv\Scripts\Activate.ps1

python -m vit_xai_audit.audit.quantus_cli `
  --model vit_b16 `
  --ckpt .\runs\vit_b16\20260118-192106\best.ckpt `
  --npz .\artifacts\attributions\vit_b16\attributions_test_subset.npz `
  --out_dir .\artifacts\metrics `
  --seed 42 `
  --fc_nr_runs 50 `
  --fc_subset_size 224 `
  --ms_nr_samples 10 `
  --ms_lower_bound 0.1 `
  --ig_steps 16

