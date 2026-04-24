# Chest X-ray Dataset Download Guide

You must download each dataset manually and place them in the correct folders.

## Directory Structure After Download

```
data/raw/
├── covid_qu_ex/          ← Dataset 1
│   ├── COVID-19/
│   ├── Non-COVID/
│   └── Normal/
├── chexpert/             ← Dataset 2
│   ├── train.csv
│   └── train/
├── nih_chestxray14/      ← Dataset 3
│   ├── Data_Entry_2017.csv
│   └── images/
├── vindr_cxr/            ← Dataset 4
│   ├── annotations/
│   └── train/
├── shenzhen_tb/          ← Dataset 5
│   ├── tuberculosis/
│   └── normal/
└── lidc_idri/            ← Dataset 6
    ├── nodule_manifest.csv
    └── slices/
```

---

## Dataset 1: COVID-QU-Ex (33,920 images)
- **Source**: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
- **Size**: ~3 GB
- **Format**: Folders per class (COVID-19, Non-COVID, Normal)
- **Download**: Kaggle → Download ZIP → Extract to `data/raw/covid_qu_ex/`
- **Contains**: Lung segmentation masks too (optional)

## Dataset 2: CheXpert (224,316 images)
- **Source**: https://stanfordmlgroup.github.io/competitions/chexpert/
- **Size**: ~11 GB (small version) or ~439 GB (full)
- **Format**: CSV manifest + image folders
- **Download**: Register at Stanford → Download CheXpert-v1.0-small → Extract to `data/raw/chexpert/`
- **Note**: Use the SMALL version (CheXpert-v1.0-small) for practical training

## Dataset 3: NIH ChestX-ray14 (112,120 images)
- **Source**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Alt Source**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Size**: ~42 GB
- **Format**: CSV labels + flat image directory
- **Download**: Kaggle or NIH Box → Extract to `data/raw/nih_chestxray14/`
- **Key file**: `Data_Entry_2017.csv` (labels), images in `images/` folder

## Dataset 4: VinDr-CXR (18,000 images)
- **Source**: https://physionet.org/content/vindr-cxr/1.0.0/
- **Size**: ~25 GB (DICOM format)
- **Format**: DICOM images + CSV annotations with bounding boxes
- **Download**: Register on PhysioNet → Download → Extract to `data/raw/vindr_cxr/`
- **Note**: You'll need to convert DICOM to PNG during preprocessing

## Dataset 5: Shenzhen TB (662 images)
- **Source**: https://www.kaggle.com/datasets/raddar/tuberculosis-tb-chest-xray-dataset
- **Alt Source**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
- **Size**: ~100 MB
- **Format**: Folder per class (tuberculosis, normal)
- **Download**: Kaggle → Extract to `data/raw/shenzhen_tb/`
- **Smallest dataset** — will be heavily oversampled

## Dataset 6: LIDC-IDRI (1,018 CT scans)
- **Source**: https://www.cancerimagingarchive.net/collection/lidc-idri/
- **Size**: ~125 GB (full CT), but we use 2D projections
- **Format**: DICOM CT volumes + nodule annotations
- **Download**: TCIA → Download → Extract to `data/raw/lidc_idri/`
- **Note**: Requires converting 3D CT → 2D CXR-like slices

---

## Quick Start (Minimum Viable)

For a quick start, download just these 2 smallest datasets:

1. **COVID-QU-Ex** (Kaggle, ~3 GB) → `data/raw/covid_qu_ex/`
2. **Shenzhen TB** (Kaggle, ~100 MB) → `data/raw/shenzhen_tb/`

This gives you 3 classes (Normal, COVID-19, Pneumonia, Tuberculosis) to test the pipeline.

## After Downloading

Run the dataset builder:
```bash
python scripts/01_build_dataset.py --config configs/dataset_config.yaml
```

This will:
1. Scan all `data/raw/` folders
2. Map labels to the unified 6-class schema
3. Create stratified train/val/test splits (70/15/15)
4. Save manifests to `data/manifests/`
