$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
cd .. | Out-Null

& .\.venv\Scripts\Activate.ps1

python -m vit_xai_audit.train.train_cli --config .\configs\train_resnet50.yaml --model resnet50 --runs_dir .\runs

