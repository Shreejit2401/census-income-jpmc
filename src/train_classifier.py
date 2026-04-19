import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    average_precision_score, f1_score, precision_recall_curve
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


class ClassificationModel:
    def __init__(self, processed_dir, models_dir):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.RANDOM_STATE = 2401
        self.results = []
        self.models_trained = {}
        
        self._load_data()
        self._setup_cv()
        
    def _load_data(self):
        self.X_train = pickle.load(open(self.processed_dir / "X_train.pkl", "rb"))
        self.X_test = pickle.load(open(self.processed_dir / "X_test.pkl", "rb"))
        self.y_train = pickle.load(open(self.processed_dir / "y_train.pkl", "rb"))
        self.y_test = pickle.load(open(self.processed_dir / "y_test.pkl", "rb"))
        self.encoder = joblib.load(self.processed_dir / "target_encoder.pkl")
        self.feature_names = pickle.load(open(self.processed_dir / "feature_names.pkl", "rb"))
        
        NEG, POS = self.y_train.value_counts()[0], self.y_train.value_counts()[1]
        self.scale_pos_weight = NEG / POS
        
    def _setup_cv(self):
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.RANDOM_STATE)
    
    def find_best_threshold(self, model, X_test, y_test):
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = f1_scores.argmax()
        return thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    def evaluate(self, name, model, X_test, y_test, threshold=0.5):
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        
        report = classification_report(
            y_test, y_pred, output_dict=True, target_names=['<=50k', '>50k']
        )
        
        minority_key = '>50k' if '>50k' in report else ('1' if '1' in report else None)
        if minority_key is None:
            class_keys = [k for k in report if k not in {'accuracy', 'macro avg', 'weighted avg'}]
            minority_key = sorted(class_keys)[-1]
        
        return {
            'model': name,
            'roc_auc': roc_auc_score(y_test, proba),
            'pr_auc': average_precision_score(y_test, proba),
            'f1_minority': report[minority_key]['f1-score'],
            'recall_minority': report[minority_key]['recall'],
            'precision_minority': report[minority_key]['precision'],
            'threshold': threshold
        }
    
    def train_logistic_regression(self):
        param_grid = {
            'lr__C': [0.01, 0.1, 1.0, 10.0],
            'lr__class_weight': ['balanced'],
            'lr__solver': ['lbfgs', 'saga'],
            'lr__max_iter': [1000],
        }
        
        pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=self.RANDOM_STATE))])
        
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=8, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(self.X_train, self.y_train)
        
        threshold = self.find_best_threshold(search.best_estimator_, self.X_test, self.y_test)
        result = self.evaluate("Logistic Regression", search.best_estimator_, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['logistic_regression'] = search.best_estimator_
        joblib.dump(search.best_estimator_, self.models_dir / 'logistic_income_pipeline.joblib', compress=3)
    
    def train_random_forest(self):
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [10, 20],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample'],
        }
        
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.RANDOM_STATE, n_jobs=-1),
            param_grid, n_iter=10, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(self.X_train, self.y_train)
        
        threshold = self.find_best_threshold(search.best_estimator_, self.X_test, self.y_test)
        result = self.evaluate("Random Forest", search.best_estimator_, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['random_forest'] = search.best_estimator_
        joblib.dump(search.best_estimator_, self.models_dir / 'random_forest_pipeline.joblib', compress=3)
    
    def train_xgboost(self):
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'scale_pos_weight': [self.scale_pos_weight, self.scale_pos_weight * 0.5],
            'min_child_weight': [5, 10],
        }
        
        search = RandomizedSearchCV(
            XGBClassifier(random_state=self.RANDOM_STATE, n_jobs=-1, eval_metric='aucpr', verbosity=0),
            param_grid, n_iter=15, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(self.X_train, self.y_train)
        
        threshold = self.find_best_threshold(search.best_estimator_, self.X_test, self.y_test)
        result = self.evaluate("XGBoost", search.best_estimator_, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['xgboost'] = search.best_estimator_
        joblib.dump(search.best_estimator_, self.models_dir / 'XGBoost_pipeline.joblib', compress=3)
        
        return search.best_estimator_
    
    def train_lightgbm(self):
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [6, 8, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'class_weight': ['balanced'],
            'min_child_samples': [20, 50],
        }
        
        search = RandomizedSearchCV(
            LGBMClassifier(random_state=self.RANDOM_STATE, n_jobs=-1, verbose=-1),
            param_grid, n_iter=15, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(self.X_train, self.y_train)
        
        threshold = self.find_best_threshold(search.best_estimator_, self.X_test, self.y_test)
        result = self.evaluate("LightGBM", search.best_estimator_, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['lightgbm'] = search.best_estimator_
        joblib.dump(search.best_estimator_, self.models_dir / 'LightGBM_pipeline.joblib', compress=3)
        
        return search.best_estimator_
    
    def train_xgboost_smote(self):
        sm = SMOTE(sampling_strategy=0.3, random_state=self.RANDOM_STATE, k_neighbors=5)
        X_train_sm, y_train_sm = sm.fit_resample(self.X_train, self.y_train)
        
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'min_child_weight': [5, 10],
        }
        
        search = RandomizedSearchCV(
            XGBClassifier(random_state=self.RANDOM_STATE, n_jobs=-1, eval_metric='aucpr', verbosity=0),
            param_grid, n_iter=15, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(X_train_sm, y_train_sm)
        
        threshold = self.find_best_threshold(search.best_estimator_, self.X_test, self.y_test)
        result = self.evaluate("XGBoost + SMOTE", search.best_estimator_, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['xgboost_smote'] = search.best_estimator_
        joblib.dump(search.best_estimator_, self.models_dir / 'SMOTE_XGBoost_pipeline.joblib', compress=3)
    
    def train_xgboost_v2_with_engineering(self):
        X_train_eng = self.X_train.copy()
        X_test_eng = self.X_test.copy()
        
        X_train_eng['wealth_score'] = (
            X_train_eng['capital gains'] - 
            X_train_eng['capital losses'] + 
            X_train_eng['dividends from stocks']
        )
        X_test_eng['wealth_score'] = (
            X_test_eng['capital gains'] - 
            X_test_eng['capital losses'] + 
            X_test_eng['dividends from stocks']
        )
        
        X_train_eng['age_x_education'] = X_train_eng['age'] * X_train_eng['education']
        X_test_eng['age_x_education'] = X_test_eng['age'] * X_test_eng['education']
        
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.01, 0.02, 0.05],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'colsample_bylevel': [0.6, 0.8, 1.0],
            'min_child_weight': [3, 5, 10, 20],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
            'scale_pos_weight': [self.scale_pos_weight],
        }
        
        search = RandomizedSearchCV(
            XGBClassifier(random_state=self.RANDOM_STATE, n_jobs=-1, eval_metric='aucpr', verbosity=0),
            param_grid, n_iter=30, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(X_train_eng, self.y_train)
        
        threshold = self.find_best_threshold(search.best_estimator_, X_test_eng, self.y_test)
        result = self.evaluate("XGBoost v2", search.best_estimator_, X_test_eng, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['xgboost_v2'] = search.best_estimator_
        joblib.dump(search.best_estimator_, self.models_dir / 'XGBoost_V2_pipeline.joblib', compress=3)
        
        return search.best_estimator_, X_train_eng, X_test_eng
    
    def train_catboost(self):
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric='PRAUC',
            random_seed=self.RANDOM_STATE,
            verbose=0
        )
        model.fit(self.X_train, self.y_train)
        
        threshold = self.find_best_threshold(model, self.X_test, self.y_test)
        result = self.evaluate("CatBoost", model, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['catboost'] = model
        joblib.dump(model, self.models_dir / 'CatBoost_pipeline.joblib', compress=3)
    
    def train_stacking(self, xgb_model, lgbm_model, rf_model):
        estimators = [
            ('xgb', xgb_model),
            ('lgbm', lgbm_model),
            ('rf', rf_model),
        ]
        
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=3,
            n_jobs=-1
        )
        stack.fit(self.X_train, self.y_train)
        
        threshold = self.find_best_threshold(stack, self.X_test, self.y_test)
        result = self.evaluate("Stacking", stack, self.X_test, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['stacking'] = stack
        joblib.dump(stack, self.models_dir / 'Stacking_pipeline.joblib', compress=3)
    
    def prune_features(self, model, threshold=0.001):
        feat_names = getattr(model, 'feature_names_in_', self.X_train.columns)
        feat_imp = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
        low_importance = feat_imp[feat_imp < threshold].index.tolist()
        
        X_train_pruned = self.X_train.drop(columns=low_importance)
        X_test_pruned = self.X_test.drop(columns=low_importance)
        
        return X_train_pruned, X_test_pruned, low_importance
    
    def train_xgboost_pruned(self, X_train_pruned, X_test_pruned):
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.01, 0.02, 0.05],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'colsample_bylevel': [0.6, 0.8, 1.0],
            'min_child_weight': [3, 5, 10, 20],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
            'scale_pos_weight': [self.scale_pos_weight],
        }
        
        search = RandomizedSearchCV(
            XGBClassifier(random_state=self.RANDOM_STATE, n_jobs=-1, eval_metric='aucpr', verbosity=0),
            param_grid, n_iter=30, scoring='average_precision',
            cv=self.cv, n_jobs=-1, random_state=self.RANDOM_STATE, verbose=0
        )
        search.fit(X_train_pruned, self.y_train)
        
        threshold = self.find_best_threshold(search.best_estimator_, X_test_pruned, self.y_test)
        result = self.evaluate("XGBoost Pruned", search.best_estimator_, X_test_pruned, self.y_test, threshold)
        self.results.append(result)
        self.models_trained['xgboost_pruned'] = search.best_estimator_
    
    def get_results_df(self):
        return pd.DataFrame(self.results).sort_values('pr_auc', ascending=False)
    
    def get_best_model(self):
        results_df = self.get_results_df()
        best_row = results_df.iloc[0]
        model_name = best_row['model'].lower().replace(' ', '_')
        
        for key, model in self.models_trained.items():
            if key.replace('_', ' ').title() == best_row['model'] or model_name in key:
                return model, best_row['threshold'], best_row
        
        return self.models_trained.get('xgboost'), 0.749, best_row


def train_all_models(raw_dir, processed_dir, models_dir):
    model = ClassificationModel(processed_dir, models_dir)
    
    print("Training Logistic Regression...")
    model.train_logistic_regression()
    
    print("Training Random Forest...")
    model.train_random_forest()
    
    print("Training XGBoost...")
    xgb_model = model.train_xgboost()
    
    print("Training LightGBM...")
    lgbm_model = model.train_lightgbm()
    
    print("Training XGBoost + SMOTE...")
    model.train_xgboost_smote()
    
    print("Training XGBoost v2 with Feature Engineering...")
    xgb_v2_model, X_train_eng, X_test_eng = model.train_xgboost_v2_with_engineering()
    
    print("Training CatBoost...")
    model.train_catboost()
    
    print("Training Stacking Ensemble...")
    model.train_stacking(xgb_model, lgbm_model, model.models_trained['random_forest'])
    
    print("Pruning Low Importance Features...")
    X_train_pruned, X_test_pruned, low_imp = model.prune_features(xgb_model, threshold=0.001)
    
    print("Training XGBoost Pruned...")
    model.train_xgboost_pruned(X_train_pruned, X_test_pruned)
    
    results_df = model.get_results_df()
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(results_df[['model', 'pr_auc', 'roc_auc', 'f1_minority', 'recall_minority', 'precision_minority', 'threshold']])
    
    best_model, best_threshold, best_info = model.get_best_model()
    print(f"\nBest Model: {best_info['model']}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"PR-AUC: {best_info['pr_auc']:.4f}")
    print(f"ROC-AUC: {best_info['roc_auc']:.4f}")
    
    return model, best_model, best_threshold


if __name__ == "__main__":
    raw_dir = r"D:\Projects\TakeHomeProject\data\raw"
    processed_dir = r"D:\Projects\TakeHomeProject\data\processed"
    models_dir = r"D:\Projects\TakeHomeProject\outputs\models"
    
    model, best_model, best_threshold = train_all_models(raw_dir, processed_dir, models_dir)
    print("\nTraining completed. All models saved.")
