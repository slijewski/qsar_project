# EGFR bioactivity prediction

A QSAR regression project using ChEMBL activity data for EGFR. It converts SMILES strings into 1,024-bit Morgan fingerprints and trains a Random Forest to estimate pIC50.

RDKit also calculates molecular weight, LogP, hydrogen-bond donors and acceptors. These descriptors are displayed in the app; the regression model uses fingerprints.

## Run

Use Python 3.12. From this directory:

```bash
uv sync
uv run python 01_data_collection.py
uv run python 02_eda_descriptors.py
uv run python 03_model_building.py
uv run streamlit run app.py
```

The first script calls the ChEMBL API. Included CSV files let you start from descriptor calculation or training. The Streamlit app can use the saved model and variance selector in `outputs/`.

Enter a SMILES string to see its molecular structure, descriptors and predicted pIC50. The app also converts the prediction to an estimated IC50 in nM. Its activity label is based on a pIC50 threshold of 6, not an experimental measurement.

## Model and evaluation

Training removes low-variance fingerprint bits and fits 100 trees on an 80/20 random split with seed 42. The script prints R² and RMSE and saves a predicted-versus-observed plot to `outputs/model_performance.png`.

## Limits

The variance selector is currently fitted before the split. It should be fitted on training data only. A random split can also place similar structures in both sets; a scaffold split would be a more useful check of generalization.

Cleaning retains one record per SMILES without reconciling assay conditions or measurement qualifiers. The descriptor script uses `pchembl_value` when available and otherwise converts nM measurements, but does not fill individual missing pChEMBL values. These choices need review before drawing conclusions about model quality.

There is no applicability-domain estimate or external validation. A predicted active label does not establish EGFR inhibition, selectivity or usefulness as a drug candidate.
