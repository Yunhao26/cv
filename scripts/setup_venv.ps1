$ErrorActionPreference = "Stop"

Write-Host "[setup_venv] creating .venv ..."
python -m venv .venv

Write-Host "[setup_venv] activating ..."
& .\.venv\Scripts\Activate.ps1

Write-Host "[setup_venv] upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "[setup_venv] installing requirements.txt (CUDA build) ..."
pip install -r .\requirements.txt

Write-Host "[setup_venv] installing package (editable) ..."
pip install -e .

Write-Host "[setup_venv] done."

