# ABCD Neuroimaging Analysis Pipeline

Config-driven pipeline for analyzing brain imaging data from the ABCD study. Supports multiple research questions (anxiety, psychosis) through dynamic configuration.

## Installation

```bash
# clone repository
git clone <repository-url>
cd abcd

# create virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# install package in editable mode
pip install -e .
```

## Data Requirements

This pipeline requires access to the ABCD Study dataset. Place the following CSV files in `data/raw/`:

- Demographics and clinical assessments
- MRI quality control metrics
- DTI metrics (FA, MD, LD, TD)
- Cortical morphometry (thickness, area, volume, sulcal depth)

See `configs/data.yaml` for the complete list of required files.

## Pipeline Overview

1. **Preprocessing** - QC filtering, missing data handling, train/val/test splits
2. **Harmonization** - ComBat harmonization to remove scanner effects
3. **PCA** - Feature extraction for classification (reduces ~1100 features → 421 components at 95% variance)
4. **Classification** - SVM, Random Forest, MLP models with nested cross-validation
5. **Regression** - Predict continuous CBCL symptom scores

## Quick Start

```bash
# run all models and generate comparison report
python run_all.py

# run specific models only
python run_all.py --models svm randomforest

# run individual pipeline
python run_svm.py

# test mode (first task only, faster)
python run_svm.py --test

# with W&B logging
python run_svm.py --wandb

# or use notebooks for detailed analysis
jupyter notebook notebooks/
```

## Configuration

All settings controlled via YAML files in `configs/`:

- `run.yaml` - Run ID, seed, research question (anxiety/psychosis)
- `data.yaml` - Data sources, QC thresholds, column mappings
- `tsne.yaml` - t-SNE parameters and visualization settings
- `pca.yaml` - PCA configuration (variance threshold, components)

**Change research question:** Edit `run_name` in `configs/run.yaml`

## Output Structure

```
outputs/
└── {research_question}/
    └── {run_id}/
        └── seed_{seed}/
            ├── datasets/           # Preprocessed parquet files
            ├── harmonized/         # ComBat harmonized features (.npy)
            ├── tsne_embeddings/    # t-SNE results and plots
            └── pca/                # PCA models and transformed data
                ├── pca_model.pkl   # Fitted PCA
                ├── scaler.pkl      # StandardScaler
                ├── train_pca.npy   # 421-component features for SVM/MLP
                ├── val_pca.npy
                └── test_pca.npy
```

## Key Features

- **Nested cross-validation** - Rigorous 5-fold outer × 3-fold inner CV prevents overfitting
- **No data leakage** - All preprocessing (harmonization, PCA) fitted only on training data
- **Class imbalance handling** - Downsampling with 100 iterations for robust predictions
- **Dynamic configuration** - Switch research questions by changing one parameter
- **Reproducible** - Single seed controls all randomness
- **Modular** - Each analysis step is independent and reusable

## Requirements

See `requirements.txt`. Main dependencies:
- pandas, numpy, scipy
- scikit-learn
- neuroHarmonize (ComBat harmonization)
- matplotlib, seaborn
- wandb (optional, for experiment tracking)
