# ABCD Neuroimaging Analysis Pipeline

Config-driven pipeline for analyzing brain imaging data from the ABCD study. Supports multiple research questions (anxiety, psychosis) through dynamic configuration.

## Pipeline Overview

1. **Preprocessing** - QC filtering, missing data handling, train/val/test splits
2. **Harmonization** - ComBat harmonization to remove scanner effects
3. **t-SNE** - Dimensionality reduction for visualization
4. **PCA** - Feature extraction for classification (reduces ~1100 features → 421 components at 95% variance)

## Quick Start

```bash
# Run full pipeline
python -m src.core.run_pipeline

# Or use notebooks individually
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

- **Dynamic configuration** - Switch research questions by changing one parameter
- **Reproducible** - Single seed controls all randomness
- **No data leakage** - Strict train/val/test separation for scalers and models
- **Modular** - Each analysis step is independent and reusable

## Next Steps

- SVM classification using PCA features
- MLP deep learning models
- Feature importance analysis

## Requirements

See `requirements.txt`. Main dependencies: pandas, scikit-learn, neuroHarmonize, matplotlib.
