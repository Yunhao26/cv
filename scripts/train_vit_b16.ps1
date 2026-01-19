$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
cd .. | Out-Null

& .\.venv\Scripts\Activate.ps1

python -m vit_xai_audit.train.train_cli --config .\configs\train_vit_b16.yaml --model vit_b16 --runs_dir .\runs

