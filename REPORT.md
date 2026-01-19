## Rapport de projet : Quantitative Auditing of ViT Explainability in Medical Diagnostics

Membres : Nour AFFES, Yunhao ZHOU  
Objectif : sur une tâche de classification binaire (pneumonie vs normal) en radiographies thoraciques, **ne pas seulement produire des explications**, mais les **auditer quantitativement** avec Quantus, en comparant CNN (ResNet-50) et ViT (ViT-B/16) sur **Faithfulness** et **Robustness**.

### 1. Jeu de données
- Kaggle : *Chest X-Ray Images (Pneumonia)*, binaire : `NORMAL` vs `PNEUMONIA`.
- Chemin local (par défaut) : `.\data\chest_xray\...`

### 2. Modèles et réglages d’entraînement (version reproductible en 2 jours)
- Modèles :
  - **ResNet-50** (pré-entraîné ImageNet, tête de classification remplacée pour 2 classes)
  - **ViT-B/16** (pré-entraîné via `timm`, tête 2 classes)
- Accélération entraînement/inférence :
  - AMP (précision mixte)
  - TF32 (efficace sur RTX série 30)
- Stratégie pour atteindre “>90%” :
  - **Calibration de seuil (threshold tuning)** : recherche du meilleur seuil sur un ensemble de validation extrait de `train/`, puis application sur `test/` (évite de régler sur `test/`).
  - Fichiers de résultats :
    - ResNet : `runs/resnet50/20260118-183906/threshold_eval.json`
    - ViT : `runs/vit_b16/20260118-192106/threshold_eval.json`

### 3. Méthodes d’explication (Captum)
Sur un sous-ensemble de test (256 images par défaut), nous générons et mettons en cache :
- **Integrated Gradients (IG)** (baseline=0, `ig_steps=16`, calcul fragmenté pour s’adapter à 8GB de VRAM)
- **Grad-CAM**
  - ResNet : Grad-CAM convolutionnel standard
  - ViT : Captum 0.7.0 `LayerGradCam` ne supporte pas `reshape_transform` ; implémentation d’un **Grad-CAM au niveau des tokens** (dernière couche, pondération par gradients, reshape en 14×14 puis upsample en 224)

Caches :
- `artifacts/attributions/resnet50/attributions_test_subset.npz`
- `artifacts/attributions/vit_b16/attributions_test_subset.npz`

### 4. Indicateurs Quantus (audit quantitatif)
Conformément à la proposition, comparaison principale :
- **Faithfulness** : `FaithfulnessCorrelation`
- **Robustness** : `MaxSensitivity`

Sorties (par échantillon) :
- `artifacts/metrics/quantus_resnet50.csv`
- `artifacts/metrics/quantus_vit_b16.csv`

### 5. Résultats agrégés (livrable principal)
Table et figures générées automatiquement :
- `artifacts/figures/summary/metrics_summary.csv`
- `artifacts/figures/summary/faithfulness_boxplot.png`
- `artifacts/figures/summary/max_sensitivity_boxplot.png`
- `artifacts/figures/summary/faithfulness_correlation_mean_ci.png`
- `artifacts/figures/summary/max_sensitivity_mean_ci.png`

(Recommandation : inclure ces 5 fichiers comme résultats principaux dans le rendu final.)

### 6. Cas d’échec (mettre en avant : “visuellement plausible ≠ mathématiquement fidèle”)
Nous sélectionnons automatiquement et visualisons deux catégories :
- **« plausible visuellement mais faible faithfulness »**
- **« peu plausible visuellement mais forte faithfulness »**

Répertoires :
- `artifacts/figures/failure_cases/resnet50/`
- `artifacts/figures/failure_cases/vit_b16/`

Chaque catégorie contient :
- `*_panel_*.png` : image + heatmap + indicateurs
- `*.csv` : liste des échantillons sélectionnés

### 7. Note importante (à déclarer dans le rapport)
- Dans `MaxSensitivity` pour ViT, certaines perturbations produisent un **Grad-CAM tout à zéro**, ce qui fait échouer Quantus (assertion). Pour terminer l’évaluation, nous appliquons un **epsilon tick déterministe (1e-8)** sur ces cartes nulles et comptons les occurrences dans les logs. Ce phénomène peut être interprété comme un signal de fragilité/stabilité des explications ViT.

### 8. Reproduire l’expérience (de bout en bout)
(En supposant que l’environnement est installé et que le dataset est décompressé.)

1) Entraînement :
```powershell
.\scripts\train_resnet50.ps1
.\scripts\train_vit_b16.ps1
```

2) Génération des explications :
```powershell
.\scripts\explain_resnet50.ps1
.\scripts\explain_vit_b16.ps1
```

3) Audit Quantus :
```powershell
.\scripts\audit_resnet50.ps1
.\scripts\audit_vit_b16.ps1
```

4) Panneaux de cas d’échec :
```powershell
.\scripts\failure_cases_resnet50.ps1
.\scripts\failure_cases_vit_b16.ps1
```

5) Figures et tableau de synthèse :
```powershell
.\scripts\report_metrics.ps1
```

### Références (citées dans la proposition)
- Barekatain, L., & Glocker, B. (2025). *Evaluating the Explainability of Vision Transformers in Medical Imaging*. arXiv:2510.12021. `https://doi.org/10.48550/arXiv.2510.12021`
- Hedström, A., et al. (2023). *Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond*. arXiv:2202.06861. `https://doi.org/10.48550/arXiv.2202.06861`

