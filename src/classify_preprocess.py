import pandas as pd
import numpy as np
import pickle
import joblib
import category_encoders as ce
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_raw_data(raw_dir):
    raw_dir = Path(raw_dir)
    cols_path = raw_dir / "census-bureau.columns"
    data_path = raw_dir / "census-bureau.data"
    
    columns = [line.strip() for line in cols_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=["?"],
        skipinitialspace=True
    )
    
    return df


def preprocess_classification(raw_dir, output_dir):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data(raw_dir)
    data = df.copy()
    
    data.drop(['year', 'weight'], axis=1, inplace=True)
    
    data['label'] = data['label'].map({
        '- 50000.': 0,
        '50000+.': 1
    })
    
    classification_drop = [
        'migration code-change in msa',
        'migration code-change in reg',
        'migration code-move within reg',
        'migration prev res in sunbelt',
        'country of birth father',
        'country of birth mother',
        'country of birth self',
        'state of previous residence',
        'region of previous residence',
        'fill inc questionnaire for veteran\'s admin',
        'live in this house 1 year ago',
        'hispanic origin',
    ]
    
    data.drop(columns=classification_drop, inplace=True)
    
    data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
    
    data['is_unemployed'] = (
        data['reason for unemployment'] != 'Not in universe'
    ).astype(int)
    
    data['is_enrolled_in_edu'] = (
        data['enroll in edu inst last wk'] != 'Not in universe'
    ).astype(int)
    
    data['has_family_under_18'] = (
        data['family members under 18'] != 'Not in universe'
    ).astype(int)
    
    data.drop(columns=[
        'reason for unemployment',
        'enroll in edu inst last wk',
        'family members under 18'
    ], inplace=True)
    
    education_order = [
        'Children',
        'Less than 1st grade',
        '1st 2nd 3rd or 4th grade',
        '5th or 6th grade',
        '7th and 8th grade',
        '9th grade',
        '10th grade',
        '11th grade',
        '12th grade no diploma',
        'High school graduate',
        'Some college but no degree',
        'Associates degree-occup /vocational',
        'Associates degree-academic program',
        'Bachelors degree(BA AB BS)',
        'Masters degree(MA MS MEng MEd MSW MBA)',
        'Prof school degree (MD DDS DVM LLB JD)',
        'Doctorate degree(PhD EdD)',
    ]
    data['education'] = data['education'].map(
        {v: i for i, v in enumerate(education_order)}
    )
    
    ohe_cols = [
        'race',
        'marital stat',
        'tax filer stat',
        'detailed household summary in household',
        'citizenship',
        'member of a labor union',
    ]
    data = pd.get_dummies(data, columns=ohe_cols, drop_first=True)
    
    data.to_csv(output_dir / 'classification_data.csv', index=False)
    
    X = data.drop(columns=['label'])
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2401,
        stratify=y
    )
    
    target_encode_cols = [
        'class of worker',
        'major industry code',
        'major occupation code',
        'full or part time employment stat',
        'detailed household and family stat',
    ]
    
    encoder = ce.TargetEncoder(cols=target_encode_cols, smoothing=10)
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)
    
    pickle.dump(X_train, open(output_dir / "X_train.pkl", "wb"))
    pickle.dump(X_test, open(output_dir / "X_test.pkl", "wb"))
    pickle.dump(y_train, open(output_dir / "y_train.pkl", "wb"))
    pickle.dump(y_test, open(output_dir / "y_test.pkl", "wb"))
    
    joblib.dump(encoder, output_dir / "target_encoder.pkl")
    
    feature_names = X_train.columns.tolist()
    pickle.dump(feature_names, open(output_dir / "feature_names.pkl", "wb"))


if __name__ == "__main__":
    raw_dir = r"D:\Projects\TakeHomeProject\data\raw"
    output_dir = r"D:\Projects\TakeHomeProject\data\processed"
    
    preprocess_classification(raw_dir, output_dir)
    print("Classification preprocessing completed.")
