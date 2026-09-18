# Extensive Project Report — UNSW-NB15 Network Intrusion Detection

## 1. Scope of this reconstruction

This report documents the recovered `UNSW-NB15.ipynb` notebook supplied for reconstruction.

The report describes what the notebook actually contains. It does **not** claim that undocumented design decisions, missing files, or unexecuted code were part of the original project.

The notebook contains 14 code cells and no markdown cells. Its recorded execution environment is Kaggle/Python 3.10.13.

---

## 2. File inventory

| File | Purpose |
|---|---|
| `README.md` | Repository-level project overview, dataset instructions, environment notes, and structure |
| `requirements.txt` | Python packages required by the recovered notebook |
| `PROJECT_REPORT.md` | Detailed technical documentation of the recovered project |
| `notebooks/UNSW-NB15.ipynb` | Primary recovered source containing the complete original notebook workflow |
| `data/README.md` | Explains the expected dataset files and why they are not committed |
| `results/README.md` | Explains how the original notebook handled results and reserves a location for future exports |

---

# 3. `notebooks/UNSW-NB15.ipynb`

This is the most important file. It contains the recovered implementation and the executed experiments.

## Cell 1 — Imports

The first cell imports the numerical, data-processing, machine-learning, feature-selection, and evaluation libraries used later.

### Main libraries

- `numpy`: numerical arrays and random-number operations.
- `pandas`: loading and manipulating the UNSW-NB15 tabular data.
- `LabelEncoder`: converts categorical/object columns into integer labels.
- `train_test_split`: creates training and testing partitions.
- `StandardScaler`: standardizes numerical feature values.
- `LogisticRegression`: linear binary classifier.
- `DecisionTreeClassifier`: tree-based classifier.
- `RandomForestClassifier`: ensemble of decision trees.
- `KNeighborsClassifier`: nearest-neighbour classifier.
- `classification_report`: precision, recall, F1-score, and support.
- `accuracy_score`: overall classification accuracy.
- `VotingClassifier`: imported for an ensemble experiment that is currently commented out.
- `cross_val_score`: imported for cross-validation code that is currently commented out.
- `mutual_info_classif` and `SelectKBest`: used for mutual-information feature selection.

The notebook does not import XGBoost until the final cell.

---

## Cell 2 — Dataset loading and preparation

The notebook reads two CSV files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

It then combines them:

```python
unsw = pd.concat([unsw_train, unsw_test]).sample(frac=1)
```

### Purpose

The original training and testing datasets are merged into one DataFrame and shuffled.

The subsequent experiments then create a new 70/30 train-test split using `train_test_split`.

### Important implication

The original supplied train/test partition is therefore **not preserved as the final evaluation split**. The two source files are concatenated before the later 70/30 split.

---

## Cell 3 — Column inspection

```python
unsw.columns
```

This displays the available feature/target columns.

It is an inspection step rather than a transformation.

---

## Cell 4 — Remove `attack_cat`

```python
unsw.drop(columns='attack_cat', axis=1, inplace=True)
```

The `attack_cat` categorical attack-category column is removed.

### Purpose

The project focuses on binary classification using the `label` column rather than multi-class attack-category prediction.

The `label` column remains and is later used as the target.

---

## Cell 5 — Commented-out binary-label conversion

This cell contains code that would convert a textual `attack` field from `normal` versus non-normal into `0` and `1`.

However, the entire cell is commented out.

### Interpretation

This code is inactive and does not affect the executed UNSW-NB15 experiment.

It appears to have been carried over from a related NSL/KDD-style workflow.

---

## Cell 6 — Identify categorical columns

```python
unsw_obj = unsw.select_dtypes(['object']).columns
```

This finds all columns whose pandas data type is `object`.

### Purpose

These columns require encoding before they can be supplied to the numerical machine-learning algorithms used later.

---

## Cell 7 — Label encoding

A `LabelEncoder` is created and applied independently to every object column:

```python
for i in unsw_obj:
    unsw[i] = le.fit_transform(unsw[i])
```

### Purpose

Categorical values such as protocol/service/state-style textual values are converted into integer representations.

### Important implementation detail

The same `LabelEncoder` instance is reused, but `fit_transform()` is called separately for each column. Thus each column gets its own categorical-to-integer mapping.

---

# 4. Feature-selection functions

The notebook contains three feature-selection approaches.

---

## 4.1 `mutual_info_select(dataset)`

This function separates the target:

```python
x = dataset.drop(['label'], axis=1)
y = dataset['label'].copy()
```

It then creates a 70/30 temporary split using `random_state=40`.

Mutual information is calculated on the training portion:

```python
mutual_info = mutual_info_classif(x_train, y_train)
```

The scores are converted to a pandas Series and associated with the feature names.

The function then creates:

```python
SelectKBest(mutual_info_classif, k=10)
```

and fits it to the training data.

### Output

The function returns the names of the 10 selected features.

### Purpose

Mutual information estimates the dependency between each feature and the binary target. `SelectKBest` then retains the 10 highest-scoring features according to that scoring function.

### Execution status

Although this function is defined, the main executed experiment shown in the notebook uses `EFPA(unsw)` rather than `mutual_info_select(unsw)`.

---

## 4.2 `EFPA(dataset)`

This is the notebook's custom feature-selection implementation.

The target column `label` is removed first.

Parameters include:

- `gamma = 0.01`
- randomly selected `lam` between approximately 0.30 and 1.99
- population size = 200
- `switch_probability = 0.8`
- 200 iterations

### Internal `levy_flight()`

The function generates values using a Cauchy-distributed random component and a random denominator raised using `lam`.

This provides stochastic movement for the population.

### Internal `select_features()`

```python
return dataset.columns[pollen > threshold]
```

Features whose final pollen value is greater than `0.5` are selected.

### Internal `fitness_function()`

```python
return np.sum(pollen)
```

The fitness is simply the sum of the pollen vector.

### Population initialization

A population of random values is created:

```python
np.random.rand(population_size, len(dataset.columns))
```

The initial population's best member becomes `abest`.

### Iterative update

For 200 iterations:

1. Levy-flight values are generated.
2. New pollen positions are calculated relative to the current best solution.
3. A mutation term is added.
4. New fitness values are calculated.
5. Solutions with improved fitness replace their previous versions.
6. The current best solution is updated.

### Final result

The best pollen vector is thresholded at `0.5`, and the corresponding feature names are returned.

### Important methodological observation

The fitness function maximizes the **sum of pollen values**, not a classifier metric, prediction error, feature count penalty, or mutual-information score.

Therefore, the recovered function is faithfully documented here as implemented; the notebook does not show an explicit predictive-performance objective inside EFPA.

### Reproducibility

EFPA uses random operations without a fixed NumPy seed. Repeated executions can therefore select different features and produce different downstream results.

---

## 4.3 `correlation(dataset)`

This function removes `label` and computes the absolute feature correlation matrix.

It constructs an upper-triangular matrix and identifies columns with correlation greater than `0.95`.

Those columns are placed in `to_drop`.

The function then returns the original UNSW column list minus those highly correlated columns.

### Purpose

This approach attempts to reduce redundant features by removing features that are highly correlated with another feature.

### Execution status

The function is defined but the displayed executed experiment does not call it.

---

# 5. `train_test(col)`

This is the main classifier evaluation function.

## Data selection

The supplied feature list/array is used:

```python
x = unsw[col]
y = unsw['label'].copy()
```

The dataset is split:

- 70% training
- 30% testing
- `random_state=40`

## Scaling

A `StandardScaler` is fitted to the training data.

The notebook then applies `fit_transform()` to both training and testing data.

The exact recovered implementation is retained in the notebook.

## Models evaluated

### Logistic Regression

The classifier is trained and evaluated.

The function records:

1. training accuracy
2. testing accuracy
3. rounded accuracy
4. complete classification report

### Decision Tree

A default `DecisionTreeClassifier()` is fitted and evaluated in the same way.

### Random Forest

A default `RandomForestClassifier()` is fitted and evaluated.

### KNN

A default `KNeighborsClassifier()` is fitted and evaluated.

### Return value

The function returns a four-element list:

```text
[
    logistic regression results,
    decision tree results,
    random forest results,
    KNN results
]
```

Each model's result is itself a list containing training accuracy, testing accuracy, rounded accuracy, and classification report.

---

# 6. Commented cross-validation implementation

Cell 12 contains an alternative `train_test()` implementation entirely commented out.

It uses `cross_val_score()` with five folds for:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN

The code calculates mean cross-validation scores.

However, those values are not ultimately returned or printed because the cell is commented out.

Therefore, cross-validation should be treated as an experimental/unused version of the evaluation code, not as part of the final executed experiment.

---

# 7. Main EFPA experiment

Cell 13 executes:

```python
normal = train_test(EFPA(unsw))
```

This means:

1. The UNSW-NB15 data is passed to EFPA.
2. EFPA selects a subset of features.
3. Those features are passed into `train_test()`.
4. Four classifiers are trained:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
5. Training/testing accuracy and classification reports are printed.

This is the primary multi-model experiment in the notebook.

---

# 8. XGBoost experiment

Cell 14 imports:

```python
from xgboost import XGBClassifier
```

It then performs another EFPA feature-selection run:

```python
x = unsw[EFPA(unsw)]
y = unsw['label'].copy()
```

The data is split using:

```python
test_size=0.3
random_state=42
```

It scales the features and trains:

```python
XGBClassifier()
```

The model's predictions are evaluated using:

- accuracy
- classification report

### Important difference

The XGBoost experiment uses `random_state=42`, while the earlier `train_test()` function uses `random_state=40`.

Also, EFPA is called again, so this is not necessarily using the same selected feature set as Cell 13.

---

# 9. Model roles in the project

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Non-linear tree-based classifier |
| Random Forest | Ensemble tree classifier |
| KNN | Instance-based classifier |
| XGBoost | Gradient-boosted tree classifier |

The purpose of using multiple models is to evaluate the selected feature representation across different classification approaches.

---

# 10. Feature-selection roles

| Method | Role in notebook |
|---|---|
| Mutual Information | Select 10 features based on estimated dependency with the target |
| Correlation | Remove highly correlated/redundant features using a 0.95 threshold |
| EFPA | Custom stochastic population-based feature-selection routine |

Only EFPA is used in the main executed classifier comparison and the XGBoost experiment shown in the supplied notebook.

---

# 11. `requirements.txt`

The reconstructed repository lists:

```text
numpy
pandas
scikit-learn
xgboost
```

These correspond to the external Python packages imported by the notebook.

The project does not require a separate package for Jupyter if the notebook is opened through an environment that already provides Jupyter/Colab/Kaggle notebook support.

---

# 12. `data/README.md`

The dataset is intentionally not included.

The notebook expects two files:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

The original notebook used Kaggle-specific paths.

The README therefore documents the expected files without copying the dataset into Git.

This also prevents the repository from becoming unnecessarily large and keeps dataset licensing/distribution considerations separate from the source code.

---

# 13. `results/README.md`

The original notebook prints results to the notebook output rather than saving CSV/JSON result files.

The `results` directory is therefore documentation-only in the reconstruction.

It can later contain:

- classification reports
- comparison tables
- plots
- feature-selection outputs
- experiment logs

Any newly generated result should be clearly distinguished from the original recorded notebook output.

---

# 14. Project workflow

The complete recovered workflow can be represented as:

```text
UNSW-NB15 training CSV
          +
UNSW-NB15 testing CSV
          |
          v
     Concatenate
          |
          v
        Shuffle
          |
          v
    Remove attack_cat
          |
          v
 Identify object columns
          |
          v
     Label encoding
          |
          +-------------------+
          |                   |
          v                   v
 Mutual Information       Correlation
          |                   |
          |              remove > 0.95
          |
          +---------+
                    |
                    v
                  EFPA
                    |
                    v
          Selected feature subset
                    |
                    v
             70/30 split
                    |
                    v
             StandardScaler
                    |
       +------------+-------------+-------------+
       |            |             |             |
       v            v             v             v
    Logistic     Decision      Random        KNN
   Regression      Tree        Forest
       |            |             |             |
       +------------+-------------+-------------+
                    |
                    v
       Accuracy + classification report


Separate path:
EFPA -> 70/30 split -> scaling -> XGBoost -> accuracy/report
```

---

# 15. What is actually recoverable from the notebook

The supplied notebook supports the following facts:

- It is an UNSW-NB15 binary classification workflow.
- It uses `label` as the target.
- `attack_cat` is removed.
- Object columns are label encoded.
- Three feature-selection functions are present.
- EFPA is the feature-selection method used in the main executed experiment.
- Four classifiers are evaluated in the main `train_test()` function.
- XGBoost is evaluated separately.
- The notebook was executed in a Kaggle-style environment.
- The dataset CSVs themselves are not embedded in the notebook.

The notebook does **not** provide enough information to establish:

- the original project title beyond what can be inferred from the notebook filename,
- the original report/documentation,
- the exact GitHub commit history,
- the original dataset download provenance,
- the original author/contributor structure,
- whether commented-out experiments were ever executed elsewhere.

Those should not be invented in the reconstructed repository.

---

# 16. Recommended repository purpose

The new repository should present itself as a **reconstruction/recovered implementation of an UNSW-NB15 intrusion-detection experiment**, rather than claiming that newly generated files were authored at the historical execution date.

The notebook itself retains its original notebook metadata and recorded execution information, while the Git repository records the date on which the reconstruction is actually committed.

---

# 17. Maintenance recommendations

For future maintenance, the project can eventually be separated into:

```text
src/
    preprocessing.py
    feature_selection.py
    models.py
    evaluation.py

notebooks/
    UNSW-NB15.ipynb

data/
    README.md

results/
    README.md
```

However, these files should only be introduced if the goal changes from **faithful recovery** to **software refactoring**. Creating them now would introduce code that did not exist in the recovered notebook.

For the initial repository, preserving the notebook as the primary source is the most faithful reconstruction.
