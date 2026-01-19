## 项目：Quantitative Auditing of ViT Explainability（两天可复现版）

本仓库用于在 Kaggle Chest X-Ray (Pneumonia) 数据集上：
- 微调 **ResNet-50** 与 **ViT-Base/16**
- 生成 **Grad-CAM**、**Integrated Gradients** 归因
- 用 **Quantus** 量化评估解释的 **Faithfulness** 与 **Robustness**

### 0) 先决条件
- Windows 10/11 + NVIDIA GPU（你这台是 RTX 3070 Ti Laptop）
- 已安装 Miniconda/Anaconda（推荐）
- 可用的 Kaggle API Token（`kaggle.json`）

### 1) 创建环境

#### 方案 A：Conda（推荐，但你当前 PowerShell 可能未配置 conda PATH）

```powershell
conda env create -f environment.yml
conda activate vit-xai-audit
```

如果你的系统提示 `conda : The term 'conda' is not recognized`，请改用下面的 **方案 B（venv/pip）**，两天内更省事。

#### 方案 B：venv/pip（本机已检测到 Python 3.11 可用）

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

### 2) 配置 Kaggle Token（只需一次）
把你的 `kaggle.json` 放到：
- `%USERPROFILE%\.kaggle\kaggle.json`

并确保该文件权限仅自己可读（Windows 通常不强制，但建议不要上传到仓库）。

### 3) 下载并解压数据

#### 方案 B：手动下载（不使用 kaggle.json）
- 在 Kaggle 数据集页面下载压缩包（通常是一个 `.zip`）
- 把该 `.zip` 放到：`.\data\`
- 然后运行解压（脚本会自动找到最新的 zip 并解压）：

```powershell
.\scripts\download_kaggle.ps1 -OutDir .\data
```

解压后预期目录结构：
- `.\data\chest_xray\train\NORMAL|PNEUMONIA`
- `.\data\chest_xray\val\NORMAL|PNEUMONIA`
- `.\data\chest_xray\test\NORMAL|PNEUMONIA`

```powershell
.\scripts\download_kaggle.ps1 -OutDir .\data
```

### 4) 环境自检（可选但推荐）

```powershell
.\scripts\smoke_test.ps1
```

### 参考
- [Barekatain & Glocker, 2025, arXiv:2510.12021](https://doi.org/10.48550/arXiv.2510.12021)
- [Hedström et al., 2023, Quantus, arXiv:2202.06861](https://doi.org/10.48550/arXiv.2202.06861)

