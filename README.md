# Census Income Classification and Customer Segmentation

A practical machine learning project that does two things on U.S. Census data:

1. Predicts whether income is above or below 50K (classification)
2. Groups people into meaningful customer segments (segmentation)

This repository is structured so a first-time user can run it end-to-end with minimal setup.

## What Is In This Repo

### Main folders

- `data`/`raw`: input files used by the pipelines
- `data`/`processed`: generated datasets and intermediate artifacts (Empty by default. It is filled automatically when the src scripts are executed)
- `src`: runnable Python scripts for preprocessing and model training
- `notebooks`: exploratory and modeling notebooks
- `outputs`/`models`: trained model files (Empty by default. It is filled automatically when the src scripts are executed)
- `outputs`/`figures`: saved charts (Empty by default. It is filled automatically when the src scripts are executed)

### Current source scripts

- src/classify_preprocess.py: builds classification-ready datasets and train/test artifacts
- src/segment_preprocess.py: builds segmentation-ready matrices and files
- src/train_classifier.py: trains and evaluates multiple classification models
- src/train_segmentation.py: trains PCA + KMeans segmentation and saves summaries

## Current Data Files

### Required input files

Place these files in data/raw:

- `census-bureau.data`
- `census-bureau.columns`

### Generated processed files

After preprocessing and training, the project uses and/or creates:

- classification_data.csv
- X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl
- feature_names.pkl
- target_encoder.pkl
- Segmentation.npy
- Segmentation_data.csv
- Segmented_Data_Target.csv
- Raw_Data.csv
- segmentation_X_pca.npy
- segmentation_summary.csv
- segmentation_assignments.pkl

## Current Model Files

In outputs/models, you will see models such as:

- logistic_income_pipeline.joblib
- random_forest_pipeline.joblib
- XGBoost_pipeline.joblib
- LightGBM_pipeline.joblib
- SMOTE_XGBoost_pipeline.joblib
- XGBoost_V2_pipeline.joblib
- final_xgb_classifier.pkl
- target_encoder.pkl
- kmeans_segmentation_model.joblib
- segmentation_pca_model.joblib

## Quick Start (Beginner Friendly)

### Required Python Version

Use **Python 3.13.13** for this project. The scripts, notebook environment, and dependency versions were validated against that version, so using a different Python release may cause package compatibility issues.

### 1. Install Python

Download and install **Python 3.13.13** from the official Python website:

- [Python downloads](https://www.python.org/downloads/)

During installation on Windows, make sure **Add Python to PATH** is selected.

### 2. Install Git

If Git is not already installed, download it here:

- [Git downloads](https://git-scm.com/downloads)

### 3. Install VS Code

If you want to run the notebooks or edit the code comfortably, install Visual Studio Code:

- [Visual Studio Code](https://code.visualstudio.com/)

### 4. Open terminal in project root

Example root path:

`D:/Projects/census-income-jpmc`

### 5. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this once in the same terminal before activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 6. Install dependencies

```powershell
pip install -r requirements.txt
```

### 7. Run preprocessing scripts

```powershell
python src/classify_preprocess.py
python src/segment_preprocess.py
```

### 8. Train models

```powershell
python src/train_classifier.py
python src/train_segmentation.py
```

That is the full reproducible pipeline.

## Recommended Run Order

If you are starting from raw data, run in this exact order:

1. `src`/`classify_preprocess.py`
2. `src`/`segment_preprocess.py`
3. `src`/`train_classifier.py`
4. `src`/`train_segmentation.py`

Notebook-first workflow is also supported. If you want exploratory context, **chain of thought** and visual analysis, run:

1. `notebooks`/`01_eda.ipynb`
2. `notebooks`/`02_classification.ipynb`
3. `notebooks`/`03_segmentation.ipynb`

## What Each Script Does

### classify_preprocess.py

- Loads raw Census files
- Cleans and encodes features for classification
- Creates train/test split artifacts
- Saves target encoder and feature names

### segment_preprocess.py

- Builds segmentation feature set
- Applies mapping, engineered flags, and scaling
- Exports segmentation matrices and target-joined segmentation data

### train_classifier.py

- Loads classification train/test artifacts
- Trains multiple models (Logistic Regression, Random Forest, XGBoost variants, LightGBM, CatBoost, stacking)
- Tunes thresholds and evaluates PR-AUC, ROC-AUC, and minority-class metrics
- Saves trained models

### train_segmentation.py

- Loads segmentation matrix
- Applies PCA
- Selects/uses k=6 for KMeans clustering
- Creates segment summaries and assignments
- Saves KMeans model, PCA model, and summary outputs

## How To Use Saved Models

### Classification

- Load model from outputs/models
- Apply the same preprocessing logic from src/classify_preprocess.py
- Predict using model.predict or model.predict_proba

### Segmentation

- Load segmentation_pca_model.joblib and kmeans_segmentation_model.joblib
- Transform features with PCA model
- Predict cluster with KMeans model
- Map cluster id to business segment labels

## Troubleshooting

### Module not found

Run:

```powershell
pip install -r requirements.txt
```

### Raw data not found

Check that both files exist in data/raw:

- census-bureau.data
- census-bureau.columns

### Script is slow

Model training can take time, especially for boosted models and random search. This is expected.

### Notebook and script outputs differ

Use the script run order above for consistent reproducibility from raw data.

## For New Contributors

If this is your first ML repository:

1. Install Python 3.13.13 first, then create the virtual environment.
2. Run only preprocessing first and inspect `data/processed`.
3. Run the training scripts one by one.
4. Open the notebooks afterward to understand the exploratory reasoning.

Helpful sources:

- [Python official documentation](https://docs.python.org/3/)
- [Git official documentation](https://git-scm.com/doc)
- [VS Code user guide](https://code.visualstudio.com/docs)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- imbalanced-learn
- category-encoders
- matplotlib, seaborn

## License

This repository uses U.S. Census public data. Add a project license file if you plan to distribute publicly.
