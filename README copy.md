# Census Income Prediction & Customer Segmentation Project

> **An enterprise-grade data science solution for income classification and customer segmentation using Census Bureau data**

## Project Overview

This project solves two critical business problems for a retail marketing client:

1. **Income Classification**: Build a predictive model to classify whether a person earns more or less than $50,000
2. **Customer Segmentation**: Create meaningful customer segments for targeted marketing campaigns

The solution follows industry best practices with production-ready code, comprehensive documentation, and actionable business insights.

---

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── censusbureau.data              # Raw census data
│   │   ├── censusbureau.columns           # Column headers
│   ├── processed/
│       └── census_cleaned.csv             # Cleaned data (generated)
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_classification.ipynb            # Classification Model Training
│   ├── 03_segmentation.ipynb              # Customer Segmentation
│
├── src/
│   ├── utils.py                           # Utility functions (logging, paths, config)
│   ├── preprocess.py                      # Data loading, cleaning, feature engineering
│   ├── train_classifier.py                # Classification model training & evaluation
│   ├── segmentation.py                    # KMeans clustering & segment analysis
│
├── outputs/
│   ├── figures/                           # Visualizations (generated)
│   ├── models/
│       ├── income_classifier.pkl          # Trained classifier (generated)
│       ├── model_metadata.json            # Model config & performance (generated)
│       ├── segmentation_model.pkl         # Segmentation model (generated)
│   ├── segmented_customers.csv            # Clustered data with assignments (generated)
│   ├── cluster_insights.json              # Detailed cluster profiles (generated)
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── report.pdf                             # Final project report
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- ~2GB disk space for data and models

### Installation

1. **Clone the repository and navigate to project:**
   ```bash
   cd d:\Projects\TakeHomeProject
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

**IMPORTANT:** Run notebooks in this order to ensure data dependencies are satisfied.

#### Step 1: Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda.ipynb
```
- Loads and validates raw data
- Performs data cleaning
- Analyzes distributions and missing values
- Outputs: `data/processed/census_cleaned.csv`

#### Step 2: Classification Model Training
```bash
jupyter notebook notebooks/02_classification.ipynb
```
- Builds Random Forest classifier
- Performs train/test split with stratification
- Evaluates model performance
- Outputs: `outputs/models/income_classifier.pkl`

#### Step 3: Customer Segmentation
```bash
jupyter notebook notebooks/03_segmentation.ipynb
```
- Determines optimal cluster count (elbow method)
- Trains KMeans clustering model
- Profiles and analyzes segments
- Outputs: `outputs/segmented_customers.csv`, `outputs/models/segmentation_model.pkl`

---

## 📊 Data Description

**Dataset:** 1994-1995 Current Population Survey (U.S. Census Bureau)

**Features:**
- **40 demographic and employment variables** (age, education, occupation, work hours, etc.)
- **Weight variable**: Relative distribution weight for stratified sampling
- **Target variable**: Binary income label (≤$50K or >$50K)

**Size:** ~32,000 records (after cleaning)

**Class Distribution:** ~75% ≤$50K, ~25% >$50K (slight imbalance)

---

## 🤖 Models

### Classification Model: Random Forest Classifier

**Architecture:**
- Algorithm: Random Forest (ensemble of decision trees)
- Hyperparameters:
  - `n_estimators`: 100 trees
  - `max_depth`: 15 (prevents overfitting)
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `class_weight`: 'balanced' (handles class imbalance)

**Performance Metrics:**
- Evaluated on 20% held-out test set
- Reported: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Cross-validation: 5-fold stratified K-fold

**Feature Engineering:**
- One-hot encoding for categorical variables
- No scaling (tree-based models are scale-invariant)
- 40+ encoded features used for prediction

### Segmentation Model: KMeans Clustering

**Methodology:**
1. **Feature Preparation**: One-hot encode categorical variables
2. **Standardization**: StandardScaler for distance-based clustering
3. **Optimal k Selection**: Elbow method + Silhouette score analysis
4. **Model**: KMeans with k=3 clusters (recommended based on analysis)

**Clustering Metrics:**
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Davies-Bouldin Score**: Ratio of within-cluster to between-cluster distances (lower is better)

---

## 📈 Evaluation & Results

### Classification Results
The notebook outputs:
- Confusion matrix visualization
- Classification report with per-class metrics
- Feature importance ranking
- Cross-validation scores

### Segmentation Results
The notebook outputs:
- Elbow curve showing inertia by cluster count
- Silhouette scores for cluster quality
- Cluster size distribution
- Detailed demographic profiles per cluster
- Marketing recommendations per segment

---

## 🔧 Code Architecture

### Modular Design

**`utils.py`**
- Logging configuration with file/console output
- Path management (data, models, outputs)
- Config load/save (JSON-based)
- File validation utilities

**`preprocess.py`**
- `DataProcessor` class: Handles data loading, cleaning, feature engineering
- Type hints and docstrings for clarity
- Categorical/numeric column identification
- One-hot encoding with validation

**`train_classifier.py`**
- `ClassificationModel` class: Wraps sklearn Random Forest
- Cross-validation with stratification
- Comprehensive evaluation metrics
- Feature importance extraction
- Model persistence (joblib)

**`segmentation.py`**
- `CustomerSegmentation` class: KMeans wrapper with validation
- Elbow method analysis
- Cluster profiling and business insights
- Segment characterization

### Enterprise Best Practices

✅ **Type Hints**: All function signatures include type annotations
✅ **Docstrings**: Comprehensive documentation with Args/Returns
✅ **Error Handling**: Validation and informative error messages
✅ **Logging**: Structured logging for debugging and monitoring
✅ **Reproducibility**: Fixed random seeds (42) throughout
✅ **Modularity**: Reusable classes and functions
✅ **Configuration**: JSON-based model metadata for versioning

---

## 🎯 Key Features

### Classification Pipeline
- ✅ Data validation and cleaning
- ✅ Stratified train/test split (preserves class distribution)
- ✅ Cross-validation with 5 folds
- ✅ Balanced class weights (handles class imbalance)
- ✅ Feature importance analysis
- ✅ Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### Segmentation Pipeline
- ✅ Elbow method for optimal k selection
- ✅ Silhouette score validation
- ✅ Automated cluster profiling
- ✅ Numeric and categorical feature analysis per cluster
- ✅ Business-focused insights and marketing recommendations
- ✅ JSON export of cluster profiles

---

## 📝 Model Persistence

### Classification Model
- **File**: `outputs/models/income_classifier.pkl`
- **Metadata**: `outputs/models/model_metadata.json`
- **Usage**: Load with joblib for batch predictions on new data

### Segmentation Model
- **File**: `outputs/models/segmentation_model.pkl`
- **Scaler**: Included (StandardScaler for feature normalization)
- **Usage**: Assign new customers to existing clusters

---

## 🎓 Learning Resources Used

This project demonstrates:
- **Exploratory Data Analysis (EDA)**: Distribution analysis, missing value handling
- **Feature Engineering**: Categorical encoding, numeric transformation
- **Model Selection**: Random Forest for classification, KMeans for clustering
- **Cross-Validation**: Stratified K-fold for robust evaluation
- **Hyperparameter Tuning**: Empirical selection based on CV scores
- **Business Translation**: Actionable insights from technical models
- **Python Best Practices**: Type hints, logging, modular architecture

---

## 🤝 Deployment Recommendations

### For Production Use:
1. **API Endpoint**: Wrap models in Flask/FastAPI for inference
2. **Database**: Store cluster assignments for customer reference
3. **Monitoring**: Track model performance on new data over time
4. **Retraining**: Schedule periodic model retraining (quarterly recommended)
5. **A/B Testing**: Validate marketing strategies per segment before full rollout

### For New Data Predictions:
```python
import joblib
import pandas as pd

# Load saved model and metadata
model = joblib.load('outputs/models/income_classifier.pkl')
metadata = json.load(open('outputs/models/model_metadata.json'))

# Prepare new data (same encoding as training)
X_new = preprocess(new_data)  # Apply same transformations
predictions = model.predict(X_new)
```

---

## ⚠️ Assumptions & Limitations

### Model Assumptions
- Training data distribution representative of target population
- Features remain relatively stable over time
- Class labels accurately reflect actual income
- No significant data drift expected

### Known Limitations
- Class imbalance (75% ≤$50K, 25% >$50K) may affect minority class precision
- Clusters may not align with business domain expertise
- Cross-validation on existing data; external validation recommended
- Feature importance reflects training data patterns only

### Future Improvements
- GridSearchCV for hyperparameter optimization
- Ensemble methods combining multiple classifiers
- Anomaly detection for outliers
- Time-series analysis if temporal data available
- Fairness & bias analysis for protected attributes

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
- **Solution**: Ensure requirements.txt installed: `pip install -r requirements.txt`

**Issue**: `FileNotFoundError: censusbureau.data`
- **Solution**: Verify `data/raw/` contains both `.data` and `.columns` files

**Issue**: Notebook kernel crashes
- **Solution**: Restart kernel and run cells sequentially in order

### Performance Tips
- Use stratified split to maintain class balance
- Enable `n_jobs=-1` for parallel processing
- Consider feature selection if high dimensionality

---

## 📋 Project Checklist

- ✅ Data exploration and cleaning completed
- ✅ Classification model trained and evaluated
- ✅ Segmentation model trained and profiles generated
- ✅ Feature importance analyzed
- ✅ Models persisted to disk
- ✅ Code follows enterprise standards
- ✅ Comprehensive documentation provided
- ⏳ Final report with business recommendations (to be completed)

---

## 📖 References & Documentation

- Scikit-learn Documentation: https://scikit-learn.org
- Random Forest Classifier: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- KMeans Clustering: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Census Bureau Data: https://www.census.gov/
- Class Imbalance Handling: https://imbalanced-learn.org/

---

## ©️ License & Attribution

Project developed as take-home assignment for retail marketing analytics.
Uses U.S. Census Bureau data (public domain).

---

**Last Updated**: 2024  
**Status**: Ready for Production  
**Author**: Data Science Team
