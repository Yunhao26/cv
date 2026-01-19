## Projet : Quantitative Auditing of ViT Explainability (version reproductible en 2 jours)

Ce dépôt permet, sur le dataset Kaggle **Chest X-Ray (Pneumonia)** :
- d’affiner **ResNet-50** et **ViT-Base/16** ;
- de générer des attributions **Grad-CAM** et **Integrated Gradients** ;
- d’évaluer quantitativement les explications avec **Quantus** selon **Faithfulness** et **Robustness**.

### 0) Prérequis
- Windows 10/11 + GPU NVIDIA (ici : RTX 3070 Ti Laptop)
- Miniconda/Anaconda installé (recommandé)
- Un token API Kaggle (`kaggle.json`) si vous choisissez le téléchargement automatique

### 1) Créer l’environnement

#### Option A : Conda (recommandé, mais votre PowerShell peut ne pas avoir conda dans le PATH)

```powershell
conda env create -f environment.yml
conda activate vit-xai-audit
```

Si votre système affiche `conda : The term 'conda' is not recognized`, utilisez l’**option B (venv/pip)** ci-dessous (plus simple sur 2 jours).

#### Option B : venv/pip (Python 3.11 est disponible sur la machine)

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

### 2) Configurer le token Kaggle (une seule fois)
Placez votre `kaggle.json` dans :
- `%USERPROFILE%\.kaggle\kaggle.json`

Assurez-vous que ce fichier reste privé (ne pas le pousser sur Git).

### 3) Télécharger et décompresser les données

#### Option B : téléchargement manuel (sans `kaggle.json`)
- Téléchargez l’archive du dataset sur Kaggle (généralement un `.zip`)
- Placez le `.zip` dans : `.\data\`
- Lancez ensuite la décompression (le script détecte le `.zip` le plus récent) :

```powershell
.\scripts\download_kaggle.ps1 -OutDir .\data
```

Structure attendue après décompression :
- `.\data\chest_xray\train\NORMAL|PNEUMONIA`
- `.\data\chest_xray\val\NORMAL|PNEUMONIA`
- `.\data\chest_xray\test\NORMAL|PNEUMONIA`

```powershell
.\scripts\download_kaggle.ps1 -OutDir .\data
```

### 4) Vérification de l’environnement (optionnel mais recommandé)

```powershell
.\scripts\smoke_test.ps1
```

### Références
- [Barekatain & Glocker, 2025, arXiv:2510.12021](https://doi.org/10.48550/arXiv.2510.12021)
- [Hedström et al., 2023, Quantus, arXiv:2202.06861](https://doi.org/10.48550/arXiv.2202.06861)

