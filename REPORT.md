## 学期项目报告：Quantitative Auditing of ViT Explainability in Medical Diagnostics

成员：Nour AFFES，Yunhao ZHOU  
项目目标：在肺炎胸片二分类任务上，**不只生成解释**，而是用 Quantus 对解释做**量化审计**，比较 CNN（ResNet-50）与 ViT（ViT-B/16）的解释在 **Faithfulness** 与 **Robustness** 维度的差异。

### 1. 数据集
- Kaggle：Chest X-Ray Images (Pneumonia)，二分类：`NORMAL` vs `PNEUMONIA`。
- 本地路径（默认）：`.\data\chest_xray\...`

### 2. 模型与训练设置（两天可复现版）
- 模型：
  - **ResNet-50**（ImageNet 预训练，替换分类头为 2 类）
  - **ViT-B/16**（`timm` 预训练，2 类分类头）
- 训练/推理加速：
  - AMP（混合精度）
  - TF32（RTX 30 系列有效）
- 达到“>90%”的策略：
  - 我们采用 **阈值校准（threshold tuning）**：在从 train 划分出的较大验证集上搜索最优阈值，再应用到 test，避免在 test 上调参。
  - 结果文件：
    - ResNet：`runs/resnet50/20260118-183906/threshold_eval.json`
    - ViT：`runs/vit_b16/20260118-192106/threshold_eval.json`

### 3. 解释方法（Captum）
对 test 子集（默认 256 张）生成两类解释并缓存：
- **Integrated Gradients (IG)**（baseline=0，`ig_steps=16`，为适配 8GB 显存做了分片）
- **Grad-CAM**
  - ResNet：标准 conv Grad-CAM
  - ViT：由于 Captum 0.7.0 `LayerGradCam` 不支持 `reshape_transform`，实现了**token 级 Grad-CAM**（最后 block token 特征 + 梯度加权，reshape 回 14×14 并上采样到 224）

缓存位置：
- `artifacts/attributions/resnet50/attributions_test_subset.npz`
- `artifacts/attributions/vit_b16/attributions_test_subset.npz`

### 4. Quantus 量化审计指标
按项目提案，重点比较：
- **Faithfulness**：`FaithfulnessCorrelation`
- **Robustness**：`MaxSensitivity`

输出（逐样本）：
- `artifacts/metrics/quantus_resnet50.csv`
- `artifacts/metrics/quantus_vit_b16.csv`

### 5. 统计汇总结果（核心交付）
统计表与图自动生成于：
- `artifacts/figures/summary/metrics_summary.csv`
- `artifacts/figures/summary/faithfulness_boxplot.png`
- `artifacts/figures/summary/max_sensitivity_boxplot.png`
- `artifacts/figures/summary/faithfulness_correlation_mean_ci.png`
- `artifacts/figures/summary/max_sensitivity_mean_ci.png`

（建议在最终提交时把上述 5 个文件直接作为结果页核心素材。）

### 6. 失败案例（强调“视觉可信 ≠ 数学可信”）
我们自动筛选两类案例并输出面板图：
- **“看起来合理但 faithfulness 低”**
- **“看起来不合理但 faithfulness 高”**

输出目录：
- `artifacts/figures/failure_cases/resnet50/`
- `artifacts/figures/failure_cases/vit_b16/`

其中每类包含：
- `*_panel_*.png`：原图 + 热图 overlay + 指标
- `*.csv`：被选中的样本清单

### 7. 重要实现说明（报告中需诚实披露）
- 在 ViT 的 `MaxSensitivity` 阶段，扰动输入下可能出现 **Grad-CAM 归因全 0**，Quantus 会断言中断。为保证评估流程能完成，我们对全 0 归因做了**极小的确定性 epsilon tick（1e-8）**，并在运行日志中统计出现次数。该现象本身可以作为“ViT 解释鲁棒性/稳定性问题”的证据之一。

### 8. 一键复现实验（从头到尾）
（默认你已完成环境安装 & 数据集已解压）

1) 训练：
```powershell
.\scripts\train_resnet50.ps1
.\scripts\train_vit_b16.ps1
```

2) 解释生成：
```powershell
.\scripts\explain_resnet50.ps1
.\scripts\explain_vit_b16.ps1
```

3) Quantus 审计：
```powershell
.\scripts\audit_resnet50.ps1
.\scripts\audit_vit_b16.ps1
```

4) 失败案例面板：
```powershell
.\scripts\failure_cases_resnet50.ps1
.\scripts\failure_cases_vit_b16.ps1
```

5) 汇总统计图表：
```powershell
.\scripts\report_metrics.ps1
```

### 参考文献（提案引用）
- Barekatain, L., & Glocker, B. (2025). *Evaluating the Explainability of Vision Transformers in Medical Imaging*. arXiv:2510.12021. `https://doi.org/10.48550/arXiv.2510.12021`
- Hedström, A., et al. (2023). *Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond*. arXiv:2202.06861. `https://doi.org/10.48550/arXiv.2202.06861`

