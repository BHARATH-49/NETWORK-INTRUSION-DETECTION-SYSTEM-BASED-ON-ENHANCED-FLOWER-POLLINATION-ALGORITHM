# UNSW-NB15 Network Intrusion Detection

Machine-learning based binary network intrusion detection using the UNSW-NB15 dataset.

## Project overview

This project loads the UNSW-NB15 training and testing datasets, combines and shuffles them, encodes categorical features, applies feature-selection techniques, and evaluates several supervised machine-learning classifiers.

The original notebook implements:

- Mutual Information feature selection
- Correlation-based feature selection
- EFPA (implemented in the notebook as a population-based feature-selection routine)
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

The primary executed experiment uses **EFPA-selected features** with Logistic Regression, Decision Tree, Random Forest, and KNN. A separate XGBoost experiment is also executed.

## Dataset

The notebook expects the UNSW-NB15 dataset files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

The original notebook was executed in a Kaggle environment and therefore references `/kaggle/input/unsw-nb15/...`.

For a local run, download the dataset from its legitimate source and place the CSV files under `data/`, then update the paths in the notebook.

The dataset itself is not included in this repository.

## Notebook

Open:

`notebooks/UNSW-NB15.ipynb`

The notebook is the recovered source of the project and preserves the original experiment structure and executed outputs.

## Environment

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

## Important reproducibility notes

The original notebook contains stochastic operations in the EFPA routine and does not set a NumPy random seed for that routine. Consequently, EFPA-selected features and downstream model results can vary between executions.

The notebook also uses `fit_transform()` on the test split for `StandardScaler`. This is preserved because this repository is intended to document/recreate the recovered experiment rather than silently alter its methodology.

## Repository structure

```text
LEVEL2-TASK1/
├── README.md
├── requirements.txt
├── PROJECT_REPORT.md
├── notebooks/
│   └── UNSW-NB15.ipynb
├── data/
│   └── README.md
└── results/
    └── README.md
```
