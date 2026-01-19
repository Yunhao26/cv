$ErrorActionPreference = "Stop"

Write-Host "[smoke_test] python=$(python --version)"

$code = @'
import torch
import torchvision
import timm
import captum
import quantus

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("captum:", captum.__version__)
print("quantus:", quantus.__version__)

print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
    x = torch.randn(16, 3, 224, 224, device="cuda")
    m = torchvision.models.resnet50(weights=None).cuda().eval()
    with torch.no_grad():
        y = m(x)
    print("forward_ok:", tuple(y.shape))
'@

$tmp = Join-Path $env:TEMP "vit_xai_audit_smoke_test.py"
Set-Content -Path $tmp -Value $code -Encoding UTF8
python $tmp
Remove-Item -Path $tmp -ErrorAction SilentlyContinue

Write-Host "[smoke_test] done."

