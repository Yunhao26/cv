$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
cd .. | Out-Null

& .\.venv\Scripts\Activate.ps1

python -m vit_xai_audit.explain.explain_cli `
  --model vit_b16 `
  --ckpt .\runs\vit_b16\20260118-192106\best.ckpt `
  --data_root .\data `
  --out_dir .\artifacts\attributions `
  --subset_size 256 `
  --batch_size 1 `
  --num_workers 4 `
  --seed 42 `
  --ig_steps 16

